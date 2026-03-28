"""Integration tests for RAGPipeline.query() using mock dependencies."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from agentic_rag.config import RAGConfig
from agentic_rag.models import QueryResult, SearchResult
from agentic_rag.llm.base import BaseLLM
from agentic_rag.retrieval.base import BaseVectorStore, BaseKeywordRetriever
from agentic_rag.pipeline.rag_pipeline import RAGPipeline


def _make_pipeline(
    mock_llm: BaseLLM,
    mock_vector_store: BaseVectorStore,
    mock_keyword_retriever: BaseKeywordRetriever | None,
    config: RAGConfig,
) -> RAGPipeline:
    return RAGPipeline(mock_llm, mock_vector_store, mock_keyword_retriever, config)


@pytest.mark.asyncio
async def test_query_returns_query_result(
    mock_llm: BaseLLM,
    mock_vector_store: BaseVectorStore,
    mock_keyword_retriever: BaseKeywordRetriever,
    sample_config: RAGConfig,
) -> None:
    # LLM: analyze returns no web search; synthesize returns an answer
    mock_llm.chat = AsyncMock(
        side_effect=[
            '{"needs_web_search": false, "reason": "static knowledge"}',
            "Here is the answer based on context.",
        ]
    )

    pipeline = _make_pipeline(
        mock_llm, mock_vector_store, mock_keyword_retriever, sample_config
    )
    result = await pipeline.query("What is the capital of France?")

    assert isinstance(result, QueryResult)
    assert isinstance(result.answer, str)
    assert isinstance(result.sources, list)
    assert isinstance(result.tool_calls_used, int)
    assert isinstance(result.latency_ms, float)


@pytest.mark.asyncio
async def test_query_empty_string_raises_value_error(
    mock_llm: BaseLLM,
    mock_vector_store: BaseVectorStore,
    mock_keyword_retriever: BaseKeywordRetriever,
    sample_config: RAGConfig,
) -> None:
    pipeline = _make_pipeline(
        mock_llm, mock_vector_store, mock_keyword_retriever, sample_config
    )
    with pytest.raises(ValueError):
        await pipeline.query("")


@pytest.mark.asyncio
async def test_query_no_web_search_when_llm_says_false(
    mock_llm: BaseLLM,
    mock_vector_store: BaseVectorStore,
    mock_keyword_retriever: BaseKeywordRetriever,
    sample_config: RAGConfig,
) -> None:
    """When LLM returns needs_web_search=false, web_search node is skipped."""
    call_log: list[str] = []

    async def chat_side_effect(prompt: str) -> str:
        call_log.append(prompt)
        if "query classifier" in prompt:
            return '{"needs_web_search": false, "reason": "has KB data"}'
        return "Answer from RAG context."

    mock_llm.chat = AsyncMock(side_effect=chat_side_effect)

    pipeline = _make_pipeline(
        mock_llm, mock_vector_store, mock_keyword_retriever, sample_config
    )
    result = await pipeline.query("Tell me about Python.")

    # Should have answer from synthesize
    assert result.answer != ""
    # Sources come from RAG (mock vector store returns sample_search_results)
    assert len(result.sources) >= 1


@pytest.mark.asyncio
async def test_query_sources_have_correct_types(
    mock_llm: BaseLLM,
    mock_vector_store: BaseVectorStore,
    mock_keyword_retriever: BaseKeywordRetriever,
    sample_config: RAGConfig,
) -> None:
    mock_llm.chat = AsyncMock(
        side_effect=[
            '{"needs_web_search": false, "reason": "test"}',
            "Final answer.",
        ]
    )
    pipeline = _make_pipeline(
        mock_llm, mock_vector_store, mock_keyword_retriever, sample_config
    )
    result = await pipeline.query("Test question")

    for source in result.sources:
        assert isinstance(source, SearchResult)
        assert isinstance(source.id, str)
        assert isinstance(source.title, str)
        assert isinstance(source.source, str)
        assert isinstance(source.content, str)
        assert isinstance(source.score, float)


@pytest.mark.asyncio
async def test_query_tool_calls_used_is_positive(
    mock_llm: BaseLLM,
    mock_vector_store: BaseVectorStore,
    mock_keyword_retriever: BaseKeywordRetriever,
    sample_config: RAGConfig,
) -> None:
    mock_llm.chat = AsyncMock(
        side_effect=[
            '{"needs_web_search": false, "reason": "test"}',
            "Final answer.",
        ]
    )
    pipeline = _make_pipeline(
        mock_llm, mock_vector_store, mock_keyword_retriever, sample_config
    )
    result = await pipeline.query("Some question")
    assert result.tool_calls_used > 0


@pytest.mark.asyncio
async def test_query_without_keyword_retriever(
    mock_llm: BaseLLM,
    mock_vector_store: BaseVectorStore,
    sample_config: RAGConfig,
) -> None:
    """Pipeline works correctly when keyword_retriever is None."""
    mock_llm.chat = AsyncMock(
        side_effect=[
            '{"needs_web_search": false, "reason": "test"}',
            "Answer without BM25.",
        ]
    )
    pipeline = _make_pipeline(mock_llm, mock_vector_store, None, sample_config)
    result = await pipeline.query("Question without BM25")

    assert isinstance(result, QueryResult)
    assert result.answer != ""
