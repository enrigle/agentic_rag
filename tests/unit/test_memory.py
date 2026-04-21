"""Unit tests for AgenticRAGSystem conversational memory."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from main import AgenticRAGSystem, AgentState


def _fake_state(query: str = "hello", answer: str = "The answer.") -> AgentState:
    return {
        "query": query,
        "chat_history": [],
        "tool_calls": 2,
        "max_tool_calls": 5,
        "rag_results": [
            {"source": "http://x.com", "title": "Doc", "content": "ctx", "score": 0.9}
        ],
        "web_results": [],
        "needs_web_search": False,
        "final_answer": answer,
        "error": None,
    }


@pytest.fixture
def system() -> AgenticRAGSystem:
    with (
        patch("main.chromadb.PersistentClient"),
        patch.object(AgenticRAGSystem, "_build_graph", return_value=MagicMock()),
        patch("main._lf_obs", lambda *a, **kw: nullcontext()),
        patch("main._lf_client", return_value=None),
    ):
        return AgenticRAGSystem()


# ── _trim_chat_history ────────────────────────────────────────────────────────


def test_trim_empty_returns_empty(system: AgenticRAGSystem) -> None:
    assert system._trim_chat_history([]) == []


def test_trim_within_limit_unchanged(system: AgenticRAGSystem) -> None:
    history = [{"role": "user", "content": "hi"}]
    assert system._trim_chat_history(history) == history


def test_trim_exceeds_limit_keeps_tail(system: AgenticRAGSystem) -> None:
    history = [{"role": "user", "content": str(i)} for i in range(20)]
    trimmed = system._trim_chat_history(history)
    assert len(trimmed) == AgenticRAGSystem.MEMORY_MAX_MESSAGES
    assert trimmed[-1]["content"] == "19"  # most recent entry kept


# ── _format_chat_history ──────────────────────────────────────────────────────


def test_format_empty_returns_empty_string(system: AgenticRAGSystem) -> None:
    assert system._format_chat_history([]) == ""


def test_format_produces_role_prefixed_lines(system: AgenticRAGSystem) -> None:
    history = [
        {"role": "user", "content": "what is X?"},
        {"role": "assistant", "content": "X is Y."},
    ]
    out = system._format_chat_history(history)
    assert "user: what is X?" in out
    assert "assistant: X is Y." in out


def test_format_skips_empty_content(system: AgenticRAGSystem) -> None:
    history = [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": "answer"},
    ]
    out = system._format_chat_history(history)
    assert "user:" not in out
    assert "assistant: answer" in out


def test_format_clips_to_max_messages(system: AgenticRAGSystem) -> None:
    history = [{"role": "user", "content": str(i)} for i in range(20)]
    out = system._format_chat_history(history)
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) <= 8  # default max_messages=8


# ── query() memory accumulation ───────────────────────────────────────────────


async def test_query_stores_turn_in_memory(system: AgenticRAGSystem) -> None:
    system.graph.ainvoke = AsyncMock(return_value=_fake_state("hello", "The answer."))
    await system.query("hello", thread_id="t1")

    history = system._chat_memory["t1"]
    assert any(m["role"] == "user" and m["content"] == "hello" for m in history)
    assert any(m["role"] == "assistant" and "answer" in m["content"].lower() for m in history)


async def test_query_thread_isolation(system: AgenticRAGSystem) -> None:
    system.graph.ainvoke = AsyncMock(return_value=_fake_state())
    await system.query("question A", thread_id="thread-A")
    await system.query("question B", thread_id="thread-B")

    thread_a = system._chat_memory.get("thread-A", [])
    thread_b = system._chat_memory.get("thread-B", [])
    assert all(m["content"] != "question B" for m in thread_a)
    assert all(m["content"] != "question A" for m in thread_b)


async def test_query_history_injected_on_second_call(system: AgenticRAGSystem) -> None:
    system.graph.ainvoke = AsyncMock(return_value=_fake_state("q1", "answer 1"))
    await system.query("q1", thread_id="t1")

    system.graph.ainvoke = AsyncMock(return_value=_fake_state("q2", "answer 2"))
    await system.query("q2", thread_id="t1")

    initial_state: dict[str, Any] = system.graph.ainvoke.call_args[0][0]
    chat_history = initial_state["chat_history"]
    assert any(m["content"] == "q1" for m in chat_history)
    assert any(m["content"] == "answer 1" for m in chat_history)


async def test_query_memory_respects_max_window(system: AgenticRAGSystem) -> None:
    many = AgenticRAGSystem.MEMORY_MAX_MESSAGES + 4
    for i in range(many):
        system.graph.ainvoke = AsyncMock(return_value=_fake_state(f"q{i}", f"a{i}"))
        await system.query(f"q{i}", thread_id="t1")

    assert len(system._chat_memory["t1"]) <= AgenticRAGSystem.MEMORY_MAX_MESSAGES


async def test_query_empty_raises(system: AgenticRAGSystem) -> None:
    with pytest.raises(ValueError, match="user_query cannot be empty"):
        await system.query("", thread_id="t1")


async def test_query_graph_failure_returns_error_answer(system: AgenticRAGSystem) -> None:
    system.graph.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
    result = await system.query("anything", thread_id="t1")
    assert "Pipeline failed" in result["answer"]
    assert result["sources"] == []
