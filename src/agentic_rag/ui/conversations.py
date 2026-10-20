"""Process-wide chat transcripts, so a browser reconnect restores its thread."""

from __future__ import annotations

from typing import Any


class ConversationStore:
    """thread_id → UI messages, keeping only the max_threads most recently written.

    thread_id comes from the URL, so an unbounded store would let anyone grow
    process memory without limit.
    """

    def __init__(self, max_threads: int = 100) -> None:
        self._max_threads = max_threads
        self._threads: dict[str, list[Any]] = {}

    def get(self, thread_id: str) -> list[Any]:
        return list(self._threads.get(thread_id, []))

    def put(self, thread_id: str, messages: list[Any]) -> None:
        # Re-insert so an active thread moves to the back of the eviction order.
        self._threads.pop(thread_id, None)
        self._threads[thread_id] = messages
        while len(self._threads) > self._max_threads:
            self._threads.pop(next(iter(self._threads)))
