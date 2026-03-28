"""ChromaDB vector store implementation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

import chromadb

from agentic_rag.config import RAGConfig
from agentic_rag.models import SearchResult
from agentic_rag.retrieval.base import BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """Persistent ChromaDB vector store with async-safe blocking I/O."""

    def __init__(self, config: RAGConfig) -> None:
        self._min_similarity: float = config.retriever.min_similarity
        client = chromadb.PersistentClient(path=config.chroma_path)
        self._collection = client.get_or_create_collection(
            config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def search(self, query_vec: list[float], top_k: int) -> list[SearchResult]:
        """Query the collection and filter results by min_similarity threshold.

        Args:
            query_vec: Pre-computed embedding vector for the query.
            top_k: Maximum number of candidates to retrieve before filtering.

        Returns:
            Ordered list of SearchResult, highest similarity first.
        """
        loop = asyncio.get_running_loop()
        raw: dict[str, Any] = cast(
            dict[str, Any],
            await loop.run_in_executor(
                None,
                lambda: self._collection.query(
                    query_embeddings=[query_vec],  # type: ignore[arg-type]
                    n_results=top_k,
                    include=["documents", "distances", "metadatas"],
                ),
            ),
        )

        results: list[SearchResult] = []
        dropped = 0
        for doc_id, doc, dist, meta in zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["distances"][0],
            raw["metadatas"][0],
        ):
            score = round(1 - dist, 4)
            if score >= self._min_similarity:
                results.append(
                    SearchResult(
                        id=doc_id,
                        title=meta.get("title", ""),
                        source=meta.get("source", ""),
                        content=doc,
                        score=score,
                    )
                )
            else:
                dropped += 1

        if dropped:
            logger.info(
                "ChromaVectorStore.search: dropped %d results below threshold %.2f",
                dropped,
                self._min_similarity,
            )

        return results

    async def fetch_by_ids(self, ids: list[str]) -> list[SearchResult]:
        """Fetch documents by IDs (used for BM25-only results).

        Args:
            ids: Document IDs to fetch.

        Returns:
            List of SearchResult with score=0.0 (no similarity available).
        """
        if not ids:
            return []

        loop = asyncio.get_running_loop()
        fetched: dict[str, Any] = cast(
            dict[str, Any],
            await loop.run_in_executor(
                None,
                lambda: self._collection.get(
                    ids=ids, include=["documents", "metadatas"]
                ),
            ),
        )

        return [
            SearchResult(
                id=fid,
                title=meta.get("title", ""),
                source=meta.get("source", ""),
                content=doc,
                score=0.0,
            )
            for fid, doc, meta in zip(
                fetched["ids"], fetched["documents"], fetched["metadatas"]
            )
        ]

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Upsert documents into the collection.

        Args:
            ids: Document IDs.
            embeddings: Corresponding embedding vectors.
            documents: Raw document text.
            metadatas: Per-document metadata dicts.
        """
        if not ids:
            logger.warning("ChromaVectorStore.upsert: called with empty ids — no-op")
            return
        if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
            raise ValueError(
                f"upsert: parallel lists must be equal length, got "
                f"ids={len(ids)}, embeddings={len(embeddings)}, "
                f"documents={len(documents)}, metadatas={len(metadatas)}"
            )
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,  # type: ignore[arg-type]
            documents=documents,
            metadatas=metadatas,  # type: ignore[arg-type]
        )
        logger.info("ChromaVectorStore.upsert: upserted %d documents", len(ids))
