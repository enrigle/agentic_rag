"""ConversationStore keeps at most max_threads threads, evicting least recent."""

from __future__ import annotations

from agentic_rag.ui.conversations import ConversationStore


def test_get_unknown_thread_is_empty() -> None:
    assert ConversationStore().get("nope") == []


def test_evicts_least_recently_written_thread() -> None:
    store = ConversationStore(max_threads=2)
    store.put("a", [1])
    store.put("b", [2])
    store.put("a", [1, 1])  # touch a → b is now oldest
    store.put("c", [3])

    assert store.get("b") == []
    assert store.get("a") == [1, 1]
    assert store.get("c") == [3]
