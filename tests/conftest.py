"""Shared fixtures for agentic_rag tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, MagicMock

from agentic_rag.config import RAGConfig, LLMConfig, RetrieverConfig
from agentic_rag.models import SearchResult
from agentic_rag.llm.base import BaseLLM
from agentic_rag.retrieval.base import BaseVectorStore, BaseKeywordRetriever

# Tests must never ship traces to the real Langfuse project. app.py calls
# load_dotenv(), which does not override variables that are already set, so
# blanking them here (before any test imports app) keeps tracing off.
for _key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
    os.environ[_key] = ""


@pytest.fixture
def sample_config() -> RAGConfig:
    return RAGConfig(
        chroma_path=Path("./test_chroma_db"),
        bm25_path=Path("./test_bm25_index"),
        collection_name="test_kb",
        max_tool_calls=3,
        llm=LLMConfig(model="test-model", embed_model="test-embed"),
        retriever=RetrieverConfig(min_similarity=0.3, top_n=3),
    )


@pytest.fixture
def mock_llm() -> BaseLLM:
    llm = MagicMock(spec=BaseLLM)
    llm.chat = AsyncMock(return_value="Here is the answer.")
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
    return store


@pytest.fixture
def mock_keyword_retriever() -> BaseKeywordRetriever:
    retriever = MagicMock(spec=BaseKeywordRetriever)
    retriever.search = MagicMock(return_value=["doc1"])
    retriever.rebuild = MagicMock()
    return retriever
