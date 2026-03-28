"""Hybrid retriever combining vector search and BM25 via Reciprocal Rank Fusion."""

from __future__ import annotations

import logging

from agentic_rag.config import RAGConfig
from agentic_rag.models import SearchResult
from agentic_rag.retrieval.base import BaseKeywordRetriever, BaseVectorStore
from agentic_rag.retrieval.chroma import ChromaVectorStore

logger = logging.getLogger(__name__)


def _rrf_merge(
    vector_ids: list[str],
    bm25_ids: list[str],
    k: int = 60,
    top_n: int = 5,
) -> tuple[list[str], dict[str, float]]:
    """Reciprocal Rank Fusion over two ranked ID lists.

    Args:
        vector_ids: Doc IDs ordered by vector similarity (best first).
        bm25_ids: Doc IDs ordered by BM25 score (best first).
        k: RRF smoothing constant (default 60).
        top_n: Maximum number of merged IDs to return.

    Returns:
        Tuple of (sorted_ids, rrf_scores) where sorted_ids is ordered
        by descending RRF score.
    """
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(vector_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(bm25_ids):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    merged = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_n]
    return merged, scores


class HybridRetriever:
    """Combines vector search and BM25 keyword search via RRF."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        keyword_retriever: BaseKeywordRetriever | None,
        config: RAGConfig,
    ) -> None:
        self._vector_store = vector_store
        self._keyword_retriever = keyword_retriever
        self._config = config

    async def search(
        self,
        query_vec: list[float],
        query_text: str,
    ) -> list[SearchResult]:
        """Run hybrid search and return merged results.

        Executes vector search and (optionally) BM25 search, merges
        the ranked ID lists via RRF, fetches any missing documents from
        the vector store, and returns the final ordered result list.

        Args:
            query_vec: Pre-computed embedding for the query.
            query_text: Raw query string for BM25 tokenisation.

        Returns:
            Up to ``config.retriever.top_n`` SearchResult objects ordered
            by descending RRF score.
        """
        cfg = self._config.retriever

        # --- Vector search ---
        vector_results = await self._vector_store.search(
            query_vec, top_k=cfg.bm25_top_k
        )
        vector_ids: list[str] = [r.id for r in vector_results]
        vector_data: dict[str, SearchResult] = {r.id: r for r in vector_results}

        dropped = cfg.bm25_top_k - len(vector_ids)
        if dropped > 0:
            logger.info(
                "HybridRetriever.search: dropped %d vector results below threshold",
                dropped,
            )

        # --- BM25 search ---
        bm25_ids: list[str] = []
        if self._keyword_retriever is not None:
            bm25_ids = self._keyword_retriever.search(query_text, top_k=cfg.bm25_top_k)
            logger.info(
                "HybridRetriever.search: BM25 returned %d candidates", len(bm25_ids)
            )

        # --- RRF merge ---
        merged_ids, rrf_scores = _rrf_merge(
            vector_ids, bm25_ids, k=cfg.rrf_k, top_n=cfg.top_n
        )

        if not merged_ids:
            logger.info("HybridRetriever.search: no results after RRF")
            return []

        # --- Fetch data for BM25-only IDs ---
        missing_ids = [fid for fid in merged_ids if fid not in vector_data]
        if missing_ids and isinstance(self._vector_store, ChromaVectorStore):
            fetched = await self._vector_store.fetch_by_ids(missing_ids)
            for result in fetched:
                vector_data[result.id] = result

        # --- Assemble final results with RRF scores ---
        final: list[SearchResult] = []
        for fid in merged_ids:
            if fid not in vector_data:
                logger.warning(
                    "HybridRetriever.search: merged ID %r not found in vector data — skipping",
                    fid,
                )
                continue
            base = vector_data[fid]
            final.append(
                SearchResult(
                    id=base.id,
                    title=base.title,
                    source=base.source,
                    content=base.content,
                    score=round(rrf_scores[fid], 6),
                )
            )

        logger.info(
            "HybridRetriever.search: hybrid returned %d results (vector=%d, bm25=%d)",
            len(final),
            len(vector_ids),
            len(bm25_ids),
        )
        return final
