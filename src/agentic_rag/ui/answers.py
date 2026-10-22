"""Shape answers for the chat UI and turn thumbs ratings into feedback."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from agentic_rag.feedback.store import FeedbackEntry, FeedbackSource, Rating
from agentic_rag.models import QueryResult
from agentic_rag.observability.langfuse import score_trace

_MD_SPECIAL = re.compile(r"([\\`*_\[\]()<>#+\-!])")


def to_result_dict(result: QueryResult) -> dict[str, Any]:
    """QueryResult → the dict kept in Streamlit session state and rendered."""
    return {
        "answer": result.answer,
        "sources": [
            {
                "index": i + 1,
                "url": s.source,
                "title": s.title,
                "content": s.content,
                "score": s.score,
            }
            for i, s in enumerate(result.sources)
        ],
        "tool_calls_used": result.tool_calls_used,
        "latency_ms": result.latency_ms,
        "trace_id": result.trace_id,
        "top_score": max((s.score for s in result.sources), default=0.0),
        # Cosine of the best KB chunk; None for web answers. Feeds min_similarity tuning.
        "top_similarity": max(
            (s.similarity for s in result.sources if s.similarity is not None),
            default=None,
        ),
    }


def source_markdown(index: int, title: str, url: str) -> str:
    """One source as a markdown link, or inert text if the URL is not http(s).

    Titles and URLs come from ingested Notion content, so both are untrusted:
    escape markdown and reject non-http schemes (javascript:, data:).
    """
    label = f"{index}. {_MD_SPECIAL.sub(r'\\\1', title)}"
    if urlparse(url).scheme in ("http", "https"):
        return f"[{label}](<{url}>)"
    return label


def feedback_entry(
    query: str, result: dict[str, Any], rating: Rating, note: str = ""
) -> FeedbackEntry:
    """Build the feedback row for a rated answer (result from to_result_dict)."""
    return FeedbackEntry(
        query=query,
        answer=result["answer"],
        sources=[
            FeedbackSource(
                title=s["title"],
                content=s.get("content", ""),
                score=s.get("score", 0.0),
            )
            for s in result["sources"]
        ],
        top_score=result.get("top_score", 0.0),
        top_similarity=result.get("top_similarity"),
        rating=rating,
        note=note,
    )


def score_feedback(result: dict[str, Any], rating: Rating, note: str = "") -> None:
    """Mirror a saved thumbs rating onto the answer's Langfuse trace (no-op if untraced)."""
    if result.get("trace_id"):
        score_trace(
            trace_id=result["trace_id"],
            name="user_feedback",
            value=1 if rating == 1 else 0,
            comment=note or None,
        )
