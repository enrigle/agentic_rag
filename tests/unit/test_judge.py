import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agentic_rag.feedback.judge import classify_failure


def _mock_llm(response: str) -> MagicMock:
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=response)
    return llm


@pytest.mark.asyncio
async def test_classify_retrieval_miss() -> None:
    with patch(
        "agentic_rag.feedback.judge._build_synth_llm",
        return_value=_mock_llm('{"category": "retrieval_miss"}'),
    ):
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
    with patch(
        "agentic_rag.feedback.judge._build_synth_llm",
        return_value=_mock_llm('{"category": "synthesis_failure"}'),
    ):
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
    with patch(
        "agentic_rag.feedback.judge._build_synth_llm",
        return_value=_mock_llm("not json at all"),
    ):
        result = await classify_failure(query="q", answer="a", sources=[])
    assert result == "unknown"


@pytest.mark.asyncio
async def test_classify_exception_returns_unknown() -> None:
    llm = MagicMock()
    llm.chat = AsyncMock(side_effect=RuntimeError("connection refused"))
    with patch("agentic_rag.feedback.judge._build_synth_llm", return_value=llm):
        result = await classify_failure(query="q", answer="a", sources=[])
    assert result == "unknown"


@pytest.mark.asyncio
async def test_classify_invalid_category_returns_unknown() -> None:
    with patch(
        "agentic_rag.feedback.judge._build_synth_llm",
        return_value=_mock_llm('{"category": "hallucination"}'),
    ):
        result = await classify_failure(query="q", answer="a", sources=[])
    assert result == "unknown"


@pytest.mark.asyncio
async def test_classify_no_braces_returns_unknown() -> None:
    with patch(
        "agentic_rag.feedback.judge._build_synth_llm",
        return_value=_mock_llm("Sorry, I cannot classify this."),
    ):
        result = await classify_failure(query="q", answer="a", sources=[])
    assert result == "unknown"
