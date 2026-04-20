from __future__ import annotations

import logging
from typing import Any, Literal, Mapping


class ErrorHandler:
    """Centralizes exception logging + fallback return construction.

    This avoids repeating `except Exception as exc: logger.exception(...); return ...`
    throughout the pipeline code.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def log(
        self,
        where: str,
        exc: BaseException,
        *,
        level: Literal["exception", "warning"] = "exception",
    ) -> None:
        if level == "exception":
            self._logger.exception("%s: %s", where, exc)
            return
        if level == "warning":
            self._logger.warning("%s: %s", where, exc)
            return
        raise ValueError(f"Unsupported level: {level!r}")

    def state_from_exception(
        self,
        state: Mapping[str, Any],
        where: str,
        exc: BaseException,
        *,
        updates: Mapping[str, Any] | None = None,
        set_error: bool = True,
        level: Literal["exception", "warning"] = "exception",
    ) -> dict[str, Any]:
        """Log and return a new state dict with standard error/tool_calls handling."""
        self.log(where, exc, level=level)

        tool_calls = 0
        try:
            tool_calls = int(state["tool_calls"])  # type: ignore[index]
        except Exception:
            tool_calls = 0

        next_state: dict[str, Any] = {**state, "tool_calls": tool_calls + 1}
        if set_error:
            next_state["error"] = str(exc)
        if updates:
            next_state.update(dict(updates))
        return next_state
