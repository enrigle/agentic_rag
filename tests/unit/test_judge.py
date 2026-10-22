import pytest
from unittest.mock import AsyncMock, MagicMock
from agentic_rag.feedback.judge import classify_failure
from agentic_rag.feedback.store import FeedbackSource


def _mock_llm(response: str) -> MagicMock:
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=response)
    return llm


@pytest.mark.asyncio
async def test_classify_retrieval_miss() -> None:
    result = await classify_failure(
        _mock_llm('{"category": "retrieval_miss"}'),
        query="What is the capital of France?",
        answer="I don't know.",
        sources=[
            FeedbackSource(
                title="Python docs", content="Python is a language.", score=0.01
            )
        ],
    )
    assert result == "retrieval_miss"


@pytest.mark.asyncio
async def test_classify_synthesis_failure() -> None:
    result = await classify_failure(
        _mock_llm('{"category": "synthesis_failure"}'),
        query="How does RRF work?",
        answer="RRF stands for...",
        sources=[
            FeedbackSource(
                title="RRF paper", content="Reciprocal Rank Fusion...", score=0.05
            )
        ],
    )
    assert result == "synthesis_failure"


@pytest.mark.asyncio
async def test_classify_invalid_json_returns_unknown() -> None:
    result = await classify_failure(
        _mock_llm("not json at all"), query="q", answer="a", sources=[]
    )
    assert result == "unknown"


@pytest.mark.asyncio
async def test_classify_exception_returns_unknown() -> None:
    llm = MagicMock()
    llm.chat = AsyncMock(side_effect=RuntimeError("connection refused"))
    result = await classify_failure(llm, query="q", answer="a", sources=[])
    assert result == "unknown"


@pytest.mark.asyncio
async def test_classify_invalid_category_returns_unknown() -> None:
    result = await classify_failure(
        _mock_llm('{"category": "hallucination"}'), query="q", answer="a", sources=[]
    )
    assert result == "unknown"


@pytest.mark.asyncio
async def test_classify_no_braces_returns_unknown() -> None:
    result = await classify_failure(
        _mock_llm("Sorry, I cannot classify this."), query="q", answer="a", sources=[]
    )
    assert result == "unknown"
