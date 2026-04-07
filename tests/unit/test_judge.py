import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agentic_rag.feedback.judge import classify_failure


@pytest.mark.asyncio
async def test_classify_retrieval_miss() -> None:
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_ctx.chat = AsyncMock(
        return_value={"message": {"content": '{"category": "retrieval_miss"}'}}
    )
    with patch("agentic_rag.feedback.judge.ollama.AsyncClient", return_value=mock_ctx):
        result = await classify_failure(
            query="What is the capital of France?",
            answer="I don't know.",
            sources=[
                {
                    "title": "Python docs",
                    "content": "Python is a language.",
                    "score": 0.01,
                }
            ],
        )
    assert result == "retrieval_miss"


@pytest.mark.asyncio
async def test_classify_synthesis_failure() -> None:
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_ctx.chat = AsyncMock(
        return_value={"message": {"content": '{"category": "synthesis_failure"}'}}
    )
    with patch("agentic_rag.feedback.judge.ollama.AsyncClient", return_value=mock_ctx):
        result = await classify_failure(
            query="How does RRF work?",
            answer="RRF stands for...",
            sources=[
                {
                    "title": "RRF paper",
                    "content": "Reciprocal Rank Fusion...",
                    "score": 0.05,
                }
            ],
        )
    assert result == "synthesis_failure"


@pytest.mark.asyncio
async def test_classify_invalid_json_returns_unknown() -> None:
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_ctx.chat = AsyncMock(return_value={"message": {"content": "not json at all"}})
    with patch("agentic_rag.feedback.judge.ollama.AsyncClient", return_value=mock_ctx):
        result = await classify_failure(query="q", answer="a", sources=[])
    assert result == "unknown"


@pytest.mark.asyncio
async def test_classify_exception_returns_unknown() -> None:
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_ctx.chat = AsyncMock(side_effect=RuntimeError("connection refused"))
    with patch("agentic_rag.feedback.judge.ollama.AsyncClient", return_value=mock_ctx):
        result = await classify_failure(query="q", answer="a", sources=[])
    assert result == "unknown"


@pytest.mark.asyncio
async def test_classify_invalid_category_returns_unknown() -> None:
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_ctx.chat = AsyncMock(
        return_value={"message": {"content": '{"category": "hallucination"}'}}
    )
    with patch("agentic_rag.feedback.judge.ollama.AsyncClient", return_value=mock_ctx):
        result = await classify_failure(query="q", answer="a", sources=[])
    assert result == "unknown"


@pytest.mark.asyncio
async def test_classify_no_braces_returns_unknown() -> None:
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_ctx.chat = AsyncMock(
        return_value={"message": {"content": "Sorry, I cannot classify this."}}
    )
    with patch("agentic_rag.feedback.judge.ollama.AsyncClient", return_value=mock_ctx):
        result = await classify_failure(query="q", answer="a", sources=[])
    assert result == "unknown"
