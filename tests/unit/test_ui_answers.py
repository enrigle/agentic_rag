"""Pure helpers behind the chat UI: result shape, source links, feedback."""

from __future__ import annotations

from unittest.mock import patch

from agentic_rag.feedback.store import FeedbackSource
from agentic_rag.models import QueryResult, SearchResult
from agentic_rag.ui import answers


def _src(score: float, similarity: float | None) -> SearchResult:
    return SearchResult(
        id="i", title="t", source="s", content="c", score=score, similarity=similarity
    )


def test_result_dict_top_similarity_ignores_sources_without_cosine() -> None:
    kb = QueryResult(
        answer="a",
        sources=[_src(-1.0, 0.6), _src(-3.0, 0.8)],
        tool_calls_used=1,
        latency_ms=1.0,
    )
    web = QueryResult(
        answer="a", sources=[_src(0.9, None)], tool_calls_used=2, latency_ms=1.0
    )
    assert answers.to_result_dict(kb)["top_similarity"] == 0.8
    assert answers.to_result_dict(web)["top_similarity"] is None


def test_source_markdown_rejects_non_http_urls() -> None:
    rendered = answers.source_markdown(1, "Notes", "javascript:alert(1)")
    assert "javascript:" not in rendered
    assert "](" not in rendered


def test_source_markdown_escapes_markdown_in_title() -> None:
    rendered = answers.source_markdown(
        1, "](javascript:alert(1))", "https://example.com"
    )
    assert rendered == "[1. \\]\\(javascript:alert\\(1\\)\\)](<https://example.com>)"


def test_score_feedback_maps_rating_to_trace_score() -> None:
    with patch("agentic_rag.ui.answers.score_trace") as score:
        answers.score_feedback({"trace_id": "t-9"}, -1, "wrong page")
        answers.score_feedback({"trace_id": None}, 1)
    score.assert_called_once_with(
        trace_id="t-9", name="user_feedback", value=0, comment="wrong page"
    )


def test_feedback_entry_copies_sources_and_scores() -> None:
    result = answers.to_result_dict(
        QueryResult(
            answer="ans", sources=[_src(-2.0, 0.7)], tool_calls_used=1, latency_ms=1.0
        )
    )
    entry = answers.feedback_entry("q", result, rating=-1, note="off")
    assert (entry.query, entry.answer, entry.rating, entry.note) == (
        "q",
        "ans",
        -1,
        "off",
    )
    assert entry.sources == [FeedbackSource(title="t", content="c", score=-2.0)]
    assert (entry.top_score, entry.top_similarity) == (-2.0, 0.7)
