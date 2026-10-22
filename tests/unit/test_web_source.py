"""WebSource converts Tavily JSON into SearchResult and drops unusable rows."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_rag.models import SearchResult
from agentic_rag.pipeline.sources import WebSource


@pytest.mark.asyncio
async def test_web_results_become_search_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    client = MagicMock()
    client.search = AsyncMock(
        return_value={
            "results": [
                {"url": "https://a", "title": "A", "content": "body", "score": 0.7},
                {"url": "https://b", "content": "no title, no score"},
                {"url": "", "content": "missing url"},
                {"url": "https://c", "content": ""},
            ]
        }
    )
    with patch("agentic_rag.pipeline.sources.AsyncTavilyClient", return_value=client):
        results = await WebSource().search("q", MagicMock())

    assert results == [
        SearchResult(
            id="https://a", title="A", source="https://a", content="body", score=0.7
        ),
        SearchResult(
            id="https://b",
            title="",
            source="https://b",
            content="no title, no score",
            score=1.0,
        ),
    ]
    assert all(r.similarity is None for r in results)


@pytest.mark.asyncio
async def test_no_api_key_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert await WebSource().search("q", MagicMock()) == []
