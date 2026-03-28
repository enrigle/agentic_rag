"""Notion → ChromaDB ingestion via NotionIngester."""

import asyncio
import functools
import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any, Mapping

import bm25s  # type: ignore[import-untyped]
import chromadb
import ollama
from dotenv import load_dotenv
from notion_client import AsyncClient
from notion_client.helpers import async_collect_paginated_api

from agentic_rag.config import RAGConfig
from agentic_rag.ingestion.base import BaseIngester
from agentic_rag.ingestion.chunker import (
    _chunk_text,
    _extract_plain_text,
    _get_title,
)
from agentic_rag.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class NotionIngester(BaseIngester):
    """Fetches pages from Notion, chunks them, embeds via BaseLLM, and upserts into ChromaDB.

    Image captioning uses ollama.AsyncClient directly because BaseLLM.embed()
    does not support vision/multimodal inputs.
    """

    def __init__(self, config: RAGConfig, llm: BaseLLM) -> None:
        self._config = config
        self._llm = llm
        self._chroma = chromadb.PersistentClient(path=config.chroma_path)
        self._collection = self._chroma.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest(self, full: bool = False) -> int:
        """Fetch all Notion pages, embed their chunks, and upsert into ChromaDB.

        Args:
            full: If True, re-index every page regardless of last_edited_time.

        Returns:
            Total number of chunks successfully upserted.

        Raises:
            RuntimeError: If NOTION_TOKEN environment variable is not set.
        """
        load_dotenv()
        token = os.environ.get("NOTION_TOKEN")
        if not token:
            raise RuntimeError("NOTION_TOKEN environment variable is not set")

        notion = AsyncClient(auth=token)
        # NOTE: ollama.AsyncClient used directly here only for image captioning
        # because BaseLLM.embed() does not support vision/multimodal inputs.
        ollama_client = ollama.AsyncClient()

        logger.info("Fetching all pages from Notion...")
        pages: list[dict[str, Any]] = await async_collect_paginated_api(
            notion.search,
            query="",
            filter={"property": "object", "value": "page"},
        )
        logger.info("Found %d pages", len(pages))

        # Prune chunks for pages that no longer exist in Notion
        all_chunks = self._collection.get(include=["metadatas"])
        _metadatas: list[Mapping[str, Any]] = list(all_chunks["metadatas"] or [])
        indexed_ids = {m["page_id"] for m in _metadatas if m and "page_id" in m}
        live_ids = {p["id"] for p in pages}
        stale_ids = indexed_ids - live_ids
        if stale_ids:
            stale_chunks = self._collection.get(
                where={"page_id": {"$in": list(stale_ids)}}  # type: ignore[dict-item]
            )
            self._collection.delete(ids=stale_chunks["ids"])
            logger.info(
                "Pruned %d chunks for %d deleted pages",
                len(stale_chunks["ids"]),
                len(stale_ids),
            )

        total_chunks = 0

        for page in pages:
            page_id: str = page["id"]
            title = _get_title(page)
            page_url: str = page.get(
                "url", f"https://notion.so/{page_id.replace('-', '')}"
            )
            last_edited_time: str = page.get("last_edited_time", "")

            # Skip unchanged pages unless doing a full re-index
            if not full:
                existing = self._collection.get(
                    where={"page_id": page_id},
                    include=["metadatas"],
                )
                _existing_metas: list[Mapping[str, Any]] = list(
                    existing["metadatas"] or []
                )
                if (
                    existing["ids"]
                    and _existing_metas
                    and _existing_metas[0].get("last_edited_time") == last_edited_time
                ):
                    logger.debug("Page '%s' unchanged — skipping", title)
                    continue

            blocks = await self._fetch_text_from_blocks(notion, ollama_client, page_id)
            if not blocks:
                logger.debug("Page '%s' produced no indexable text — skipping", title)
                continue

            chunks = _chunk_text(
                blocks,
                size=self._config.ingestion.chunk_size,
                overlap=self._config.ingestion.chunk_overlap,
            )

            ids: list[str] = []
            embeddings: list[list[float]] = []
            documents: list[str] = []
            chunk_metadatas: list[dict[str, Any]] = []

            for i, chunk in enumerate(chunks):
                try:
                    vector: list[float] = await self._llm.embed(chunk)
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
                # Delete existing chunks for this page before upserting new ones
                existing_for_page = self._collection.get(where={"page_id": page_id})
                if existing_for_page["ids"]:
                    self._collection.delete(ids=existing_for_page["ids"])
                self._collection.upsert(
                    ids=ids,
                    embeddings=embeddings,  # type: ignore[arg-type]
                    documents=documents,
                    metadatas=chunk_metadatas,  # type: ignore[arg-type]
                )
                total_chunks += len(ids)
                logger.info("Indexed '%s': %d chunk(s)", title, len(ids))

        self._rebuild_bm25()
        logger.info(
            "Done. Indexed %d pages, %d chunks into ChromaDB at %r.",
            len(pages),
            total_chunks,
            self._config.chroma_path,
        )
        return total_chunks

    def status(self) -> dict[str, Any]:
        """Return a summary of what is currently indexed in the collection.

        Returns:
            A dict with keys: total_chunks, distinct_pages, oldest_edit, newest_edit.
        """
        all_chunks = self._collection.get(include=["metadatas"])
        metadatas: list[Mapping[str, Any]] = list(all_chunks["metadatas"] or [])
        page_ids = {m["page_id"] for m in metadatas if m and "page_id" in m}
        times = sorted(
            m["last_edited_time"] for m in metadatas if m and m.get("last_edited_time")
        )
        return {
            "total_chunks": len(metadatas),
            "distinct_pages": len(page_ids),
            "oldest_edit": times[0] if times else None,
            "newest_edit": times[-1] if times else None,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from all documents in the collection."""
        all_docs = self._collection.get(include=["documents"])
        ids: list[str] = all_docs["ids"] or []
        documents: list[str] = all_docs["documents"] or []
        if not documents:
            logger.warning("BM25: collection is empty — skipping index build")
            return
        tokenized = bm25s.tokenize(documents, show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(tokenized, show_progress=False)
        bm25_path = Path(self._config.bm25_path)
        bm25_path.mkdir(exist_ok=True)
        retriever.save(str(bm25_path))
        (bm25_path / "id_map.json").write_text(json.dumps(ids))
        logger.info("BM25 index saved: %d documents", len(documents))

    async def _caption_image(self, ollama_client: ollama.AsyncClient, url: str) -> str:
        """Download image and return a text caption via the configured vision model.

        NOTE: Uses ollama.AsyncClient directly because BaseLLM.embed() does not
        support vision/multimodal inputs.
        """
        try:
            loop = asyncio.get_running_loop()

            def _fetch() -> bytes:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    return resp.read()  # type: ignore[no-any-return]

            image_bytes = await loop.run_in_executor(None, functools.partial(_fetch))
            response = await ollama_client.chat(
                model=self._config.ingestion.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": "Extract any text visible in this image. If it's a diagram or chart, describe what it shows. Be concise.",
                        "images": [image_bytes],
                    }
                ],
            )
            return response.message.content.strip() if response.message.content else ""
        except Exception as exc:
            logger.warning("Image captioning failed (%s): %s", url, exc)
            return ""

    async def _fetch_text_from_blocks(
        self,
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
                    caption = await self._caption_image(ollama_client, url)
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
                child_blocks = await self._fetch_text_from_blocks(
                    notion, ollama_client, block["id"], depth + 1, max_depth
                )
                typed_lines.extend(child_blocks)

        return typed_lines
