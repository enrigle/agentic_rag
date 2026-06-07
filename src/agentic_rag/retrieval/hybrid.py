"""Hybrid retriever combining vector search and BM25 via Reciprocal Rank Fusion."""

from __future__ import annotations

import logging

from agentic_rag.config import RAGConfig
from agentic_rag.models import SearchResult
from agentic_rag.retrieval.base import BaseKeywordRetriever, BaseVectorStore

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
    return merged, {fid: scores[fid] for fid in merged}


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
        """Run hybrid search and return merged results."""
        cfg = self._config.retriever

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

        bm25_ids: list[str] = []
        if self._keyword_retriever is not None:
            bm25_ids = self._keyword_retriever.search(query_text, top_k=cfg.bm25_top_k)
            logger.info(
                "HybridRetriever.search: BM25 returned %d candidates", len(bm25_ids)
            )

        # No vector results means nothing cleared min_similarity — signal coordinator
        # to fall through to web search rather than returning BM25-only results.
        if not vector_ids:
            logger.info(
                "HybridRetriever.search: no vector results above min_similarity threshold"
            )
            return []

        merged_ids, rrf_scores = _rrf_merge(
            vector_ids, bm25_ids, k=cfg.rrf_k, top_n=cfg.top_n
        )
        if not merged_ids:
            logger.info("HybridRetriever.search: no results after RRF")
            return []

        await self._fill_vector_data(merged_ids, vector_data)
        merged_ids = self._deduplicate_by_source(merged_ids, vector_data)
        final = self._build_results(merged_ids, vector_data, rrf_scores)

        logger.info(
            "HybridRetriever.search: hybrid returned %d results (vector=%d, bm25=%d)",
            len(final),
            len(vector_ids),
            len(bm25_ids),
        )
        return final

    async def _fill_vector_data(
        self,
        merged_ids: list[str],
        vector_data: dict[str, SearchResult],
    ) -> None:
        missing = [fid for fid in merged_ids if fid not in vector_data]
        if missing:
            for result in await self._vector_store.fetch_by_ids(missing):
                vector_data[result.id] = result

    def _deduplicate_by_source(
        self,
        merged_ids: list[str],
        vector_data: dict[str, SearchResult],
    ) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for fid in merged_ids:
            if fid in vector_data and vector_data[fid].source not in seen:
                seen.add(vector_data[fid].source)
                deduped.append(fid)
        return deduped

    def _build_results(
        self,
        merged_ids: list[str],
        vector_data: dict[str, SearchResult],
        rrf_scores: dict[str, float],
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        for fid in merged_ids:
            if fid not in vector_data:
                logger.warning(
                    "HybridRetriever.search: merged ID %r not found — skipping", fid
                )
                continue
            base = vector_data[fid]
            results.append(
                SearchResult(
                    id=base.id,
                    title=base.title,
                    source=base.source,
                    content=base.content,
                    score=round(rrf_scores[fid], 6),
                )
            )
        return results
