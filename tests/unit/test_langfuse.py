"""Tracing is best-effort, but its failures must be logged, not swallowed."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from agentic_rag.observability import langfuse as lf


@pytest.fixture
def enabled(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    lf.get_client.cache_clear()
    yield
    lf.get_client.cache_clear()


def test_client_init_failure_is_logged(
    enabled: None, caplog: pytest.LogCaptureFixture
) -> None:
    with (
        patch("langfuse.Langfuse", side_effect=RuntimeError("bad host")),
        caplog.at_level(logging.WARNING),
    ):
        assert lf.get_client() is None
    assert "bad host" in caplog.text


def test_score_failure_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    client = MagicMock()
    client.create_score.side_effect = RuntimeError("401")
    with (
        patch.object(lf, "langfuse_enabled", return_value=True),
        patch.object(lf, "get_client", return_value=client),
        caplog.at_level(logging.WARNING),
    ):
        lf.score_trace(trace_id="t", name="n", value=1)
    assert "401" in caplog.text


def test_observation_start_failure_is_logged_and_yields_none(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = MagicMock()
    client.start_as_current_observation.side_effect = RuntimeError("otel")
    with (
        patch.object(lf, "get_client", return_value=client),
        caplog.at_level(logging.WARNING),
    ):
        with lf.observation("x") as obs:
            assert obs is None
    assert "otel" in caplog.text
