import asyncio
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


def test_run_ingest_error() -> None:
    with (
        patch("app.load_config", side_effect=RuntimeError("bad config")),
    ):
        app._sync_state["status"] = "idle"
        app._sync_state["error"] = ""
        app._run_ingest()

    assert app._sync_state["status"] == "error"
    assert "bad config" in app._sync_state["error"]
