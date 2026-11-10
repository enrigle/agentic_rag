"""app.py is Streamlit layout only; these guard what importing it may do."""

import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import app
from agentic_rag.models import QueryResult


def test_importing_app_does_not_start_ingest() -> None:
    # Tests import app as a module; only `streamlit run` (__name__ == "__main__")
    # may kick off a real Notion ingest that writes to data/chroma_db.
    import streamlit as st

    st.session_state["sync_started"] = False
    with patch("agentic_rag.ui.ingest.BackgroundIngest.start") as start:
        importlib.reload(app)
    start.assert_not_called()


def test_tests_run_with_tracing_off() -> None:
    from agentic_rag.observability.langfuse import langfuse_enabled

    assert not langfuse_enabled()  # .env credentials must not leak into tests


def _run_app_with_query(check_web: bool) -> AsyncMock:
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.cache_resource.clear()
    pipeline = MagicMock()
    pipeline.query = AsyncMock(
        return_value=QueryResult(
            answer="a", sources=[], tool_calls_used=0, latency_ms=1.0
        )
    )
    with (
        patch("agentic_rag.health.run_checks", AsyncMock(return_value=[])),
        patch(
            "agentic_rag.pipeline.rag_pipeline.create_pipeline", return_value=pipeline
        ),
        patch("agentic_rag.ui.ingest.BackgroundIngest.start"),
    ):
        at = AppTest.from_file("app.py", default_timeout=30).run()
        assert not at.exception
        box = at.checkbox(key="use_web")
        assert box.label == "Also search the web"
        assert box.value is False
        if check_web:
            at = box.check().run()
        at.chat_input[0].set_value("hi").run()
        assert not at.exception
    query: AsyncMock = pipeline.query
    return query


def test_web_toggle_off_by_default() -> None:
    query = _run_app_with_query(check_web=False)
    assert query.call_args.kwargs["use_web"] is False


def test_web_toggle_checked_passes_true() -> None:
    query = _run_app_with_query(check_web=True)
    assert query.call_args.kwargs["use_web"] is True
