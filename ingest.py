"""Notion → ChromaDB ingestion script.

Usage:
    export NOTION_TOKEN=secret_xxx
    uv run python ingest.py              # incremental (skip unchanged, prune deleted)
    uv run python ingest.py --full       # force full re-index
    uv run python ingest.py --status     # print collection stats and exit
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # loads .env from cwd (or any parent dir)

import bm25s
import chromadb
import ollama
from notion_client import AsyncClient
from notion_client.helpers import async_collect_paginated_api

logger = logging.getLogger(__name__)

CHROMA_PATH = "./chroma_db"
BM25_PATH = "./bm25_index"
COLLECTION_NAME = "notion_kb"
EMBED_MODEL = "nomic-embed-text"
VISION_MODEL = "llava"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Block types that contain rich_text content worth indexing
RICH_TEXT_BLOCK_TYPES = {
    "paragraph",
    "heading_1",
    "heading_2",
    "heading_3",
    "bulleted_list_item",
    "numbered_list_item",
    "code",
    "quote",
    "toggle",
    "callout",
}

HEADING_BLOCK_TYPES = {"heading_1", "heading_2", "heading_3"}


def _extract_plain_text(block: dict[str, Any]) -> str:
    """Return plain text from a block's rich_text array, or empty string."""
    block_type = block.get("type", "")
    if block_type not in RICH_TEXT_BLOCK_TYPES:
        return ""
    rich_texts: list[dict[str, Any]] = block.get(block_type, {}).get("rich_text", [])
    return "".join(rt.get("plain_text", "") for rt in rich_texts)


def _get_title(page: dict[str, Any]) -> str:
    """Extract page title from Notion page properties."""
    for prop in page.get("properties", {}).values():
        if prop.get("type") == "title":
            rich_texts = prop.get("title", [])
            return "".join(rt.get("plain_text", "") for rt in rich_texts)
    return "Untitled"


def _emit_chunk(chunks: list[str], heading: str, paras: list[str]) -> None:
    prefix = f"{heading}\n" if heading else ""
    chunks.append(prefix + "\n".join(paras))


def _chunk_text(
    blocks: list[dict[str, str]],
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Paragraph-aware chunking that prepends the last seen heading to each chunk.

    Splits on paragraph boundaries first. Falls back to word-boundary
    character-splitting only for paragraphs that exceed ``size`` on their own.
    The most recent heading is prepended to every chunk so the LLM always has
    section context.
    """
    if not blocks:
        return []

    # Build (heading_context, paragraph_text) pairs
    tagged: list[tuple[str, str]] = []
    current_heading = ""
    for block in blocks:
        block_type = block["type"]
        text = block["text"].strip()
        if not text:
            continue
        if block_type in HEADING_BLOCK_TYPES:
            current_heading = text
        else:
            tagged.append((current_heading, text))

    if not tagged:
        return []

    chunks: list[str] = []
    buf_paras: list[str] = []
    buf_heading = ""
    buf_len = 0

    for heading, para in tagged:
        if heading:
            buf_heading = heading

        # Oversized single paragraph: flush buffer then character-split the para
        if len(para) > size:
            if buf_paras:
                _emit_chunk(chunks, buf_heading, buf_paras)
                buf_paras = []
                buf_len = 0
            start = 0
            while start < len(para):
                end = start + size
                if end < len(para):
                    space = para.rfind(" ", start, end)
                    if space > start:
                        end = space
                sub = para[start:end].strip()
                if sub:
                    _emit_chunk(chunks, buf_heading, [sub])
                start = max(start + 1, end - overlap)
            continue

        # Adding this paragraph would exceed the size limit: flush first
        if buf_paras and buf_len + len(para) + 1 > size:
            _emit_chunk(chunks, buf_heading, buf_paras)
            # Carry the last paragraph forward as overlap context
            last = buf_paras[-1]
            buf_paras = [last] if len(last) + len(para) + 1 <= size else []
            buf_len = len(buf_paras[0]) if buf_paras else 0

        buf_paras.append(para)
        buf_len = sum(len(p) for p in buf_paras) + max(0, len(buf_paras) - 1)

    if buf_paras:
        _emit_chunk(chunks, buf_heading, buf_paras)

    return chunks


async def _caption_image(ollama_client: ollama.AsyncClient, url: str) -> str:
    """Download image and return a text caption via vision model."""
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            image_bytes = resp.read()
        response = await ollama_client.chat(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Extract any text visible in this image. If it's a diagram or chart, describe what it shows. Be concise.",
                    "images": [image_bytes],
                }
            ],
        )
        return response["message"]["content"].strip()
    except Exception as exc:
        logger.warning("Image captioning failed (%s): %s", url, exc)
        return ""


async def _fetch_text_from_blocks(
    notion: AsyncClient,
    ollama_client: ollama.AsyncClient,
    block_id: str,
    depth: int = 0,
    max_depth: int = 10,
) -> list[dict[str, str]]:
    """Recursively extract typed text blocks from all Notion blocks.

    Returns a list of ``{"type": block_type, "text": plain_text}`` dicts so
    that callers can distinguish headings from body text when chunking.
    """
    if depth > max_depth:
        return []

    try:
        blocks: list[dict[str, Any]] = await async_collect_paginated_api(
            notion.blocks.children.list,
            block_id=block_id,
        )
    except Exception as exc:
        logger.warning("Could not fetch blocks for %s: %s", block_id, exc)
        return []

    typed_lines: list[dict[str, str]] = []
    for block in blocks:
        block_type = block.get("type", "")

        text = _extract_plain_text(block)
        if text.strip():
            typed_lines.append({"type": block_type, "text": text})

        if block_type == "image":
            img = block.get("image", {})
            url = img.get("file", {}).get("url") or img.get("external", {}).get(
                "url", ""
            )
            if url:
                caption = await _caption_image(ollama_client, url)
                if caption:
                    typed_lines.append(
                        {"type": "paragraph", "text": f"[Image: {caption}]"}
                    )
            continue

        # child_page: add title as a reference but skip recursing
        # (child pages are indexed separately via notion.search())
        if block_type == "child_page":
            child_title = block.get("child_page", {}).get("title", "")
            if child_title:
                typed_lines.append(
                    {"type": "paragraph", "text": f"[Sub-page: {child_title}]"}
                )
            continue

        # child_database: skip content, just note its existence
        if block_type == "child_database":
            continue

        # Recursively fetch nested children (toggles, callouts, nested lists, etc.)
        if block.get("has_children"):
            child_blocks = await _fetch_text_from_blocks(
                notion, ollama_client, block["id"], depth + 1, max_depth
            )
            typed_lines.extend(child_blocks)

    return typed_lines


def _rebuild_bm25(collection: chromadb.Collection) -> None:
    """Rebuild BM25 index from all documents in the collection."""
    all_docs = collection.get(include=["documents"])
    ids: list[str] = all_docs["ids"] or []
    documents: list[str] = all_docs["documents"] or []
    if not documents:
        logger.warning("BM25: collection is empty — skipping index build")
        return
    tokenized = bm25s.tokenize(documents, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokenized, show_progress=False)
    Path(BM25_PATH).mkdir(exist_ok=True)
    retriever.save(BM25_PATH)
    (Path(BM25_PATH) / "id_map.json").write_text(json.dumps(ids))
    logger.info("BM25 index saved: %d documents", len(documents))


async def ingest(args: argparse.Namespace) -> None:
    token = os.environ.get("NOTION_TOKEN")
    if not token:
        logger.error("NOTION_TOKEN env var is not set")
        sys.exit(1)

    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if args.status:
        all_chunks = collection.get(include=["metadatas"])
        metadatas: list[dict[str, Any]] = all_chunks["metadatas"] or []
        page_ids = {m["page_id"] for m in metadatas if m and "page_id" in m}
        times = sorted(
            m["last_edited_time"] for m in metadatas if m and m.get("last_edited_time")
        )
        print(f"Total chunks  : {len(metadatas)}")
        print(f"Distinct pages: {len(page_ids)}")
        if times:
            print(f"Oldest edit   : {times[0]}")
            print(f"Newest edit   : {times[-1]}")
        return

    notion = AsyncClient(auth=token)
    ollama_client = ollama.AsyncClient()

    logger.info("Fetching all pages from Notion...")
    pages: list[dict[str, Any]] = await async_collect_paginated_api(
        notion.search,
        query="",
        filter={"property": "object", "value": "page"},
    )
    logger.info("Found %d pages", len(pages))

    # Prune chunks for pages that no longer exist in Notion
    all_chunks = collection.get(include=["metadatas"])
    indexed_ids = {
        m["page_id"] for m in all_chunks["metadatas"] if m and "page_id" in m
    }
    live_ids = {p["id"] for p in pages}
    stale_ids = indexed_ids - live_ids
    if stale_ids:
        stale_chunks = collection.get(where={"page_id": {"$in": list(stale_ids)}})
        collection.delete(ids=stale_chunks["ids"])
        logger.info(
            "Pruned %d chunks for %d deleted pages",
            len(stale_chunks["ids"]),
            len(stale_ids),
        )

    total_chunks = 0

    for page in pages:
        page_id: str = page["id"]
        title = _get_title(page)
        page_url: str = page.get("url", f"https://notion.so/{page_id.replace('-', '')}")
        last_edited_time: str = page.get("last_edited_time", "")

        # Skip pages whose content hasn't changed since last index (unless --full)
        if not args.full:
            existing = collection.get(
                where={"page_id": page_id},
                include=["metadatas"],
            )
            if (
                existing["ids"]
                and existing["metadatas"][0].get("last_edited_time") == last_edited_time
            ):
                logger.debug("Page '%s' unchanged — skipping", title)
                continue

        blocks = await _fetch_text_from_blocks(notion, ollama_client, page_id)
        if not blocks:
            logger.debug("Page '%s' produced no indexable text — skipping", title)
            continue
        chunks = _chunk_text(blocks)

        # Embed and upsert each chunk
        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        chunk_metadatas: list[dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            try:
                embed_resp = await ollama_client.embed(model=EMBED_MODEL, input=chunk)
                vector: list[float] = embed_resp["embeddings"][0]
            except Exception as exc:
                logger.warning(
                    "Embedding failed for chunk %d of '%s': %s", i, title, exc
                )
                continue

            ids.append(f"{page_id}_chunk_{i}")
            embeddings.append(vector)
            documents.append(chunk)
            chunk_metadatas.append(
                {
                    "title": title,
                    "source": page_url,
                    "page_id": page_id,
                    "last_edited_time": last_edited_time,
                }
            )

        if ids:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=chunk_metadatas,
            )
            total_chunks += len(ids)
            logger.info("Indexed '%s': %d chunk(s)", title, len(ids))

    _rebuild_bm25(collection)
    print(
        f"Done. Indexed {len(pages)} pages, {total_chunks} chunks into ChromaDB at {CHROMA_PATH!r}."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Notion → ChromaDB ingestion")
    parser.add_argument("--full", action="store_true", help="Force full re-index")
    parser.add_argument(
        "--status", action="store_true", help="Print collection stats and exit"
    )
    args = parser.parse_args()
    asyncio.run(ingest(args))
