"""Langfuse v4 tracing helpers.

Uses the official Langfuse SDK (v4+) with OpenTelemetry-based context propagation.
All helpers are no-ops if credentials are absent or the package is unavailable.
"""

from __future__ import annotations

import contextlib
import os
from functools import lru_cache
from typing import Any


def langfuse_enabled() -> bool:
    """Return True if Langfuse credentials are present in the environment."""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


@lru_cache(maxsize=1)
def get_client() -> Any | None:
    """Return a cached Langfuse client, or None if disabled/unavailable."""
    if not langfuse_enabled():
        return None
    try:
        from langfuse import Langfuse  # type: ignore[import-not-found]

        kwargs: dict[str, Any] = {
            "public_key": os.getenv("LANGFUSE_PUBLIC_KEY"),
            "secret_key": os.getenv("LANGFUSE_SECRET_KEY"),
        }
        host = os.getenv("LANGFUSE_HOST")
        if host:
            kwargs["host"] = host
        return Langfuse(**kwargs)
    except Exception:  # noqa: BLE001
        return None


@contextlib.contextmanager
def observation(
    name: str,
    *,
    as_type: str = "span",
    input: Any | None = None,  # noqa: A002
    metadata: dict[str, Any] | None = None,
    model: str | None = None,
) -> Any:
    """Context manager that creates a Langfuse span/generation/agent, or no-ops.

    Args:
        name: Observation name shown in the Langfuse UI.
        as_type: Observation type — e.g. "agent", "generation", "embedding", "span".
        input: Input payload to attach to the observation.
        metadata: Arbitrary metadata dict.
        model: Model name (used when as_type is "generation" or "embedding").
    """
    client = get_client()
    if client is None:
        yield None
        return

    kwargs: dict[str, Any] = {"name": name, "as_type": as_type}
    if input is not None:
        kwargs["input"] = input
    if metadata:
        kwargs["metadata"] = metadata
    if model is not None:
        kwargs["model"] = model

    try:
        with client.start_as_current_observation(**kwargs) as obs:
            yield obs
    except Exception:  # noqa: BLE001
        yield None


def score_trace(
    *,
    trace_id: str,
    name: str,
    value: float | int,
    comment: str | None = None,
) -> None:
    """Attach a score to an existing trace id (best-effort).

    Compatible with Langfuse v4 ``create_score()`` API.
    """
    if not trace_id or not langfuse_enabled():
        return
    client = get_client()
    if client is None:
        return
    try:
        score_kwargs: dict[str, Any] = {
            "trace_id": trace_id,
            "name": name,
            "value": float(value),
        }
        if comment is not None:
            score_kwargs["comment"] = comment
        client.create_score(**score_kwargs)
        client.flush()
    except Exception:  # noqa: BLE001
        pass
