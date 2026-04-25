"""Unit tests for ConversationMemory."""

from __future__ import annotations

import pytest

from agentic_rag.pipeline.memory import ConversationMemory


@pytest.fixture
def memory() -> ConversationMemory:
    return ConversationMemory()


# ── get ───────────────────────────────────────────────────────────────────────


def test_get_unknown_thread_returns_empty(memory: ConversationMemory) -> None:
    assert memory.get("unknown") == []


def test_get_returns_copy(memory: ConversationMemory) -> None:
    memory.append("t1", "hello", "world")
    copy = memory.get("t1")
    copy.clear()
    assert len(memory.get("t1")) == 2  # original unaffected


# ── append ────────────────────────────────────────────────────────────────────


def test_append_adds_user_and_assistant_turns(memory: ConversationMemory) -> None:
    memory.append("t1", "hello", "hi there")
    history = memory.get("t1")
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "hello"}
    assert history[1] == {"role": "assistant", "content": "hi there"}


def test_append_multiple_turns(memory: ConversationMemory) -> None:
    memory.append("t1", "q1", "a1")
    memory.append("t1", "q2", "a2")
    history = memory.get("t1")
    assert len(history) == 4
    assert history[2]["content"] == "q2"
    assert history[3]["content"] == "a2"


# ── thread isolation ──────────────────────────────────────────────────────────


def test_thread_isolation(memory: ConversationMemory) -> None:
    memory.append("A", "q-A", "a-A")
    memory.append("B", "q-B", "a-B")

    history_a = memory.get("A")
    history_b = memory.get("B")

    assert all(m["content"] != "q-B" for m in history_a)
    assert all(m["content"] != "q-A" for m in history_b)


# ── max messages ──────────────────────────────────────────────────────────────


def test_history_capped_at_max_messages(memory: ConversationMemory) -> None:
    many = ConversationMemory.MAX_MESSAGES + 4
    for i in range(many):
        memory.append("t1", f"q{i}", f"a{i}")
    assert len(memory.get("t1")) <= ConversationMemory.MAX_MESSAGES


def test_most_recent_messages_kept(memory: ConversationMemory) -> None:
    many = ConversationMemory.MAX_MESSAGES + 2
    for i in range(many):
        memory.append("t1", f"q{i}", f"a{i}")
    history = memory.get("t1")
    last_q = f"q{many - 1}"
    assert any(m["content"] == last_q for m in history)
