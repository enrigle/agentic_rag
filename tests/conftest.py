"""Shared fixtures for agentic_rag tests."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from agentic_rag.config import RAGConfig, LLMConfig, RetrieverConfig, IngestionConfig
from agentic_rag.models import SearchResult
from agentic_rag.llm.base import BaseLLM
from agentic_rag.retrieval.base import BaseVectorStore, BaseKeywordRetriever


@pytest.fixture
def sample_config() -> RAGConfig:
    return RAGConfig(
        chroma_path="./test_chroma_db",
        bm25_path="./test_bm25_index",
        collection_name="test_kb",
        max_tool_calls=3,
        llm=LLMConfig(model="test-model", embed_model="test-embed"),
        retriever=RetrieverConfig(min_similarity=0.3, top_n=3),
    )


@pytest.fixture
def mock_llm() -> BaseLLM:
    llm = MagicMock(spec=BaseLLM)
    llm.chat = AsyncMock(return_value='{"needs_web_search": false, "reason": "test"}')
    llm.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    return llm


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    return [
        SearchResult(
            id="doc1",
            title="Test Doc",
            source="http://example.com",
            content="Test content",
            score=0.9,
        ),
    ]


@pytest.fixture
def mock_vector_store(sample_search_results: list[SearchResult]) -> BaseVectorStore:
    store = MagicMock(spec=BaseVectorStore)
    store.search = AsyncMock(return_value=sample_search_results)
    store.fetch_by_ids = AsyncMock(return_value=sample_search_results)
    store.upsert = MagicMock()
    return store


@pytest.fixture
def mock_keyword_retriever() -> BaseKeywordRetriever:
    retriever = MagicMock(spec=BaseKeywordRetriever)
    retriever.search = MagicMock(return_value=["doc1"])
    retriever.rebuild = MagicMock()
    return retriever
