"""Integration tests for PipelineCoordinator.query() using mock dependencies."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from agentic_rag.config import RAGConfig
from agentic_rag.models import QueryResult, SearchResult
from agentic_rag.llm.base import BaseLLM
from agentic_rag.pipeline.coordinator import PipelineCoordinator
from agentic_rag.pipeline.memory import ConversationMemory
from agentic_rag.pipeline.synthesizer import Synthesizer
from agentic_rag.retrieval.reranker import CrossEncoderReranker


def _make_mock_source(name: str, results: list[dict], score: float = 0.9):
    source = MagicMock()
    source.name = name
    source.search = AsyncMock(return_value=results)
    return source


def _make_coordinator(
    mock_llm: BaseLLM,
    config: RAGConfig,
    source_results: list[dict] | None = None,
) -> PipelineCoordinator:
    if source_results is None:
        source_results = [
            {
                "id": "doc1",
                "title": "Test Doc",
                "source": "http://example.com",
                "content": "Test content",
                "score": 0.9,
            }
        ]
    mock_source = _make_mock_source("rag", source_results)
    reranker = MagicMock(spec=CrossEncoderReranker)
    reranker.rerank = MagicMock(return_value=source_results)
    return PipelineCoordinator(
        sources=[mock_source],
        reranker=reranker,
        synthesizer=Synthesizer(mock_llm),
        memory=ConversationMemory(),
        threshold=config.retriever.web_search_fallback_score,
        max_tool_calls=config.max_tool_calls,
    )


@pytest.mark.asyncio
async def test_query_returns_query_result(
    mock_llm: BaseLLM,
    sample_config: RAGConfig,
) -> None:
    coordinator = _make_coordinator(mock_llm, sample_config)
    result = await coordinator.query("What is the capital of France?")

    assert isinstance(result, QueryResult)
    assert isinstance(result.answer, str)
    assert isinstance(result.sources, list)
    assert isinstance(result.tool_calls_used, int)
    assert isinstance(result.latency_ms, float)


@pytest.mark.asyncio
async def test_query_empty_string_raises_value_error(
    mock_llm: BaseLLM,
    sample_config: RAGConfig,
) -> None:
    coordinator = _make_coordinator(mock_llm, sample_config)
    with pytest.raises(ValueError):
        await coordinator.query("")


@pytest.mark.asyncio
async def test_query_sources_have_correct_types(
    mock_llm: BaseLLM,
    sample_config: RAGConfig,
) -> None:
    coordinator = _make_coordinator(mock_llm, sample_config)
    result = await coordinator.query("Test question")

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
    sample_config: RAGConfig,
) -> None:
    coordinator = _make_coordinator(mock_llm, sample_config)
    result = await coordinator.query("Some question")
    assert result.tool_calls_used > 0


@pytest.mark.asyncio
async def test_query_stops_at_high_score_source(
    mock_llm: BaseLLM,
    sample_config: RAGConfig,
) -> None:
    """When first source returns results above threshold, second source is not called."""
    high_score_results = [
        {"id": "1", "title": "T", "source": "s", "content": "c", "score": 0.99}
    ]
    first_source = _make_mock_source("rag", high_score_results)
    second_source = _make_mock_source("web", [])
    reranker = MagicMock(spec=CrossEncoderReranker)
    reranker.rerank = MagicMock(return_value=high_score_results)

    coordinator = PipelineCoordinator(
        sources=[first_source, second_source],
        reranker=reranker,
        synthesizer=Synthesizer(mock_llm),
        memory=ConversationMemory(),
        threshold=sample_config.retriever.web_search_fallback_score,
        max_tool_calls=sample_config.max_tool_calls,
    )
    await coordinator.query("question")
    second_source.search.assert_not_called()


@pytest.mark.asyncio
async def test_query_falls_through_to_second_source_on_low_score(
    mock_llm: BaseLLM,
    sample_config: RAGConfig,
) -> None:
    """When first source score is below threshold, second source is called."""
    low_score_results = [
        {"id": "1", "title": "T", "source": "s", "content": "c", "score": 0.01}
    ]
    web_results = [
        {
            "id": "2",
            "title": "Web",
            "source": "http://web.com",
            "content": "web content",
            "score": 0.8,
        }
    ]
    first_source = _make_mock_source("rag", low_score_results)
    second_source = _make_mock_source("web", web_results)
    reranker = MagicMock(spec=CrossEncoderReranker)
    reranker.rerank = MagicMock(return_value=low_score_results + web_results)

    coordinator = PipelineCoordinator(
        sources=[first_source, second_source],
        reranker=reranker,
        synthesizer=Synthesizer(mock_llm),
        memory=ConversationMemory(),
        threshold=sample_config.retriever.web_search_fallback_score,
        max_tool_calls=sample_config.max_tool_calls,
    )
    await coordinator.query("question about karpathy")
    second_source.search.assert_called_once()


@pytest.mark.asyncio
async def test_query_memory_persists_across_calls(
    mock_llm: BaseLLM,
    sample_config: RAGConfig,
) -> None:
    coordinator = _make_coordinator(mock_llm, sample_config)
    await coordinator.query("first question", thread_id="t1")
    await coordinator.query("second question", thread_id="t1")

    history = coordinator._memory.get("t1")
    assert any(m["content"] == "first question" for m in history)
    assert any(m["content"] == "second question" for m in history)


@pytest.mark.asyncio
async def test_query_thread_isolation(
    mock_llm: BaseLLM,
    sample_config: RAGConfig,
) -> None:
    coordinator = _make_coordinator(mock_llm, sample_config)
    await coordinator.query("thread A question", thread_id="A")
    await coordinator.query("thread B question", thread_id="B")

    history_a = coordinator._memory.get("A")
    history_b = coordinator._memory.get("B")
    assert all(m["content"] != "thread B question" for m in history_a)
    assert all(m["content"] != "thread A question" for m in history_b)
