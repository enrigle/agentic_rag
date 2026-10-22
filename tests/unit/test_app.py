"""app.py is Streamlit layout only; these guard what importing it may do."""

import importlib
from unittest.mock import patch

import app


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
