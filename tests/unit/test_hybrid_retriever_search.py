"""Unit tests for agentic_rag.retrieval.hybrid.HybridRetriever.search."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_rag.config import RAGConfig, RetrieverConfig
from agentic_rag.models import SearchResult
from agentic_rag.retrieval.hybrid import HybridRetriever


@pytest.mark.asyncio
async def test_hybrid_search_fetches_bm25_only_ids_and_scores() -> None:
    cfg = RAGConfig(retriever=RetrieverConfig(top_n=3, bm25_top_k=10, rrf_k=60))

    vector_store = MagicMock()
    vector_store.search = AsyncMock(
        return_value=[
            SearchResult(
                id="doc1",
                title="Doc 1",
                source="kb",
                content="one",
                score=0.9,
            )
        ]
    )
    vector_store.fetch_by_ids = AsyncMock(
        return_value=[
            SearchResult(
                id="doc2",
                title="Doc 2",
                source="kb",
                content="two",
                score=0.8,
            )
        ]
    )

    keyword = MagicMock()
    keyword.search = MagicMock(return_value=["doc2", "doc1"])

    retriever = HybridRetriever(vector_store, keyword, cfg)
    results = await retriever.search([0.1, 0.2], "query")

    assert [r.id for r in results] == ["doc1", "doc2"]

    # doc1 appears in both lists: rank 0 (vector) and rank 1 (bm25)
    expected_doc1 = round(1.0 / (60 + 0 + 1) + 1.0 / (60 + 1 + 1), 6)
    expected_doc2 = round(1.0 / (60 + 0 + 1), 6)
    assert results[0].score == expected_doc1
    assert results[1].score == expected_doc2

    vector_store.fetch_by_ids.assert_awaited_once_with(["doc2"])


@pytest.mark.asyncio
async def test_hybrid_search_skips_ids_missing_from_fetch() -> None:
    cfg = RAGConfig(retriever=RetrieverConfig(top_n=3, bm25_top_k=10, rrf_k=60))

    vector_store = MagicMock()
    vector_store.search = AsyncMock(return_value=[])
    vector_store.fetch_by_ids = AsyncMock(return_value=[])

    keyword = MagicMock()
    keyword.search = MagicMock(return_value=["missing"])

    retriever = HybridRetriever(vector_store, keyword, cfg)
    results = await retriever.search([0.0], "query")
    assert results == []

