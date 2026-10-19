import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import app


def test_run_ingest_success() -> None:
    mock_ingester = MagicMock()
    mock_ingester.ingest = AsyncMock(return_value=42)

    with (
        patch("app.load_config", return_value=MagicMock()),
        patch("app.make_embed_llm", return_value=MagicMock()),
        patch("app.NotionIngester", return_value=mock_ingester),
    ):
        app._sync_state["status"] = "idle"
        app._sync_state["chunks"] = 0
        app._sync_state["error"] = ""
        app._run_ingest()

    assert app._sync_state["status"] == "done"
    assert app._sync_state["chunks"] == 42
    assert app._sync_state["error"] == ""


def test_run_ingest_error_does_not_leak_exception() -> None:
    with (
        patch("app.load_config", side_effect=RuntimeError("bad config at /srv/secret")),
    ):
        app._sync_state["status"] = "idle"
        app._sync_state["error"] = ""
        app._run_ingest()

    assert app._sync_state["status"] == "error"
    assert "/srv/secret" not in app._sync_state["error"]


def test_maybe_start_ingest_starts_only_one() -> None:
    app._sync_state["status"] = "idle"
    with patch("app.threading.Thread") as mock_thread:
        assert app._maybe_start_ingest() is True
        assert app._maybe_start_ingest() is False  # already "syncing"
    assert mock_thread.call_count == 1


def test_render_source_rejects_non_http_urls() -> None:
    with patch("app.st.markdown") as mock_md:
        app._render_source(1, "Notes", "javascript:alert(1)")
    rendered = mock_md.call_args[0][0]
    assert "javascript:" not in rendered
    assert "](" not in rendered


def test_render_source_escapes_markdown_in_title() -> None:
    with patch("app.st.markdown") as mock_md:
        app._render_source(1, "](javascript:alert(1))", "https://example.com")
    rendered = mock_md.call_args[0][0]
    assert rendered == "[1. \\]\\(javascript:alert\\(1\\)\\)](<https://example.com>)"


def test_result_dict_top_similarity_ignores_sources_without_cosine() -> None:
    from agentic_rag.models import QueryResult, SearchResult

    def src(score: float, similarity: float | None) -> SearchResult:
        return SearchResult(
            id="i",
            title="t",
            source="s",
            content="c",
            score=score,
            similarity=similarity,
        )

    kb = QueryResult(
        answer="a",
        sources=[src(-1.0, 0.6), src(-3.0, 0.8)],
        tool_calls_used=1,
        latency_ms=1.0,
    )
    web = QueryResult(
        answer="a", sources=[src(0.9, None)], tool_calls_used=2, latency_ms=1.0
    )
    assert app._to_result_dict(kb)["top_similarity"] == 0.8
    assert app._to_result_dict(web)["top_similarity"] is None


def test_importing_app_does_not_start_ingest() -> None:
    # Tests import app as a module; only `streamlit run` (__name__ == "__main__")
    # may kick off a real Notion ingest that writes to data/chroma_db.
    import streamlit as st

    st.session_state["sync_started"] = False
    app._sync_state["status"] = "idle"
    with patch("threading.Thread") as mock_thread:
        importlib.reload(app)
    mock_thread.assert_not_called()


def test_score_feedback_maps_rating_to_trace_score() -> None:
    with patch("app.score_trace") as score:
        app._score_feedback({"trace_id": "t-9"}, -1, "wrong page")
        app._score_feedback({"trace_id": None}, 1)
    score.assert_called_once_with(
        trace_id="t-9", name="user_feedback", value=0, comment="wrong page"
    )


def test_tests_run_with_tracing_off() -> None:
    from agentic_rag.observability.langfuse import langfuse_enabled

    assert not langfuse_enabled()  # .env credentials must not leak into tests
