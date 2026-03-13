"""One-time Notion → ChromaDB ingestion script.

Usage:
    export NOTION_TOKEN=secret_xxx
    uv run python ingest.py

Re-run any time to refresh the index (upserts are idempotent).
"""

import asyncio
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # loads .env from cwd (or any parent dir)

import chromadb
import ollama
from notion_client import AsyncClient
from notion_client.helpers import async_collect_paginated_api

logger = logging.getLogger(__name__)

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "notion_kb"
EMBED_MODEL = "nomic-embed-text"
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


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    if not text.strip():
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


async def _fetch_text_from_blocks(
    notion: AsyncClient,
    block_id: str,
    depth: int = 0,
    max_depth: int = 10,
) -> list[str]:
    """Recursively extract plain text from all blocks, including nested children."""
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

    lines: list[str] = []
    for block in blocks:
        block_type = block.get("type", "")

        text = _extract_plain_text(block)
        if text.strip():
            lines.append(text)

        # child_page: add title as a reference but skip recursing
        # (child pages are indexed separately via notion.search())
        if block_type == "child_page":
            child_title = block.get("child_page", {}).get("title", "")
            if child_title:
                lines.append(f"[Sub-page: {child_title}]")
            continue

        # child_database: skip content, just note its existence
        if block_type == "child_database":
            continue

        # Recursively fetch nested children (toggles, callouts, nested lists, etc.)
        if block.get("has_children"):
            child_lines = await _fetch_text_from_blocks(
                notion, block["id"], depth + 1, max_depth
            )
            lines.extend(child_lines)

    return lines


async def ingest() -> None:
    token = os.environ.get("NOTION_TOKEN")
    if not token:
        logger.error("NOTION_TOKEN env var is not set")
        sys.exit(1)

    notion = AsyncClient(auth=token)
    ollama_client = ollama.AsyncClient()
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    logger.info("Fetching all pages from Notion...")
    pages: list[dict[str, Any]] = await async_collect_paginated_api(
        notion.search,
        query="",
        filter={"property": "object", "value": "page"},
    )
    logger.info("Found %d pages", len(pages))

    total_chunks = 0

    for page in pages:
        page_id: str = page["id"]
        title = _get_title(page)
        page_url: str = page.get("url", f"https://notion.so/{page_id.replace('-', '')}")

        lines = await _fetch_text_from_blocks(notion, page_id)
        if not lines:
            logger.debug("Page '%s' produced no indexable text — skipping", title)
            continue
        full_text = "\n".join(lines)
        chunks = _chunk_text(full_text)

        # Embed and upsert each chunk
        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for i, chunk in enumerate(chunks):
            try:
                embed_resp = await ollama_client.embed(model=EMBED_MODEL, input=chunk)
                vector: list[float] = embed_resp["embeddings"][0]
            except Exception as exc:
                logger.warning("Embedding failed for chunk %d of '%s': %s", i, title, exc)
                continue

            ids.append(f"{page_id}_chunk_{i}")
            embeddings.append(vector)
            documents.append(chunk)
            metadatas.append(
                {"title": title, "source": page_url, "page_id": page_id}
            )

        if ids:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            total_chunks += len(ids)
            logger.info("Indexed '%s': %d chunk(s)", title, len(ids))

    print(f"Done. Indexed {len(pages)} pages, {total_chunks} chunks into ChromaDB at {CHROMA_PATH!r}.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(ingest())
