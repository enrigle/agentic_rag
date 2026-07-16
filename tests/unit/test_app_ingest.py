from unittest.mock import AsyncMock, MagicMock, patch

import app


def test_run_ingest_success() -> None:
    mock_ingester = MagicMock()
    mock_ingester.ingest = AsyncMock(return_value=42)

    with (
        patch("app.load_config", return_value=MagicMock()),
        patch("app.OllamaLLM", return_value=MagicMock()),
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
