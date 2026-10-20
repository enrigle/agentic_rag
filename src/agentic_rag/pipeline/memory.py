"""In-memory conversation history store, keyed by thread_id."""

from __future__ import annotations

from agentic_rag.models import ChatMessage


class ConversationMemory:
    MAX_MESSAGES: int = 10

    def __init__(self) -> None:
        self._store: dict[str, list[ChatMessage]] = {}

    def get(self, thread_id: str) -> list[ChatMessage]:
        return list(self._store.get(thread_id, []))

    def append(self, thread_id: str, query: str, answer: str) -> None:
        history = self._store.setdefault(thread_id, [])
        history.append(ChatMessage("user", query))
        history.append(ChatMessage("assistant", answer))
        if len(history) > self.MAX_MESSAGES:
            self._store[thread_id] = history[-self.MAX_MESSAGES :]
