"""BackgroundIngest: one ingest at a time, status reported, errors not leaked."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from agentic_rag.ui.ingest import BackgroundIngest


def test_run_success_records_chunks() -> None:
    ingester = MagicMock()
    ingester.ingest = AsyncMock(return_value=42)
    bg = BackgroundIngest(make_ingester=lambda: ingester)

    bg.run()

    assert (bg.state.status, bg.state.chunks, bg.state.error) == ("done", 42, "")
    ingester.ingest.assert_awaited_once_with(full=False)


def test_run_error_does_not_leak_exception() -> None:
    bg = BackgroundIngest(
        make_ingester=MagicMock(side_effect=RuntimeError("bad config at /srv/secret"))
    )

    bg.run()

    assert bg.state.status == "error"
    assert "/srv/secret" not in bg.state.error


def test_start_runs_only_one_ingest() -> None:
    bg = BackgroundIngest(make_ingester=MagicMock())
    with patch("agentic_rag.ui.ingest.threading.Thread") as mock_thread:
        assert bg.start() is True
        assert bg.start() is False  # already "syncing"
    assert mock_thread.call_count == 1
    assert bg.state.status == "syncing"
