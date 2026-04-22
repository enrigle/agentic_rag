"""Shared data models for agentic_rag."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict


@dataclass
class SearchResult:
    id: str
    title: str
    source: str
    content: str
    score: float


@dataclass
class QueryResult:
    answer: str
    sources: list[SearchResult]
    tool_calls_used: int
    latency_ms: float


class AgentState(TypedDict):
    """LangGraph state dict — must remain TypedDict."""

    query: str
    chat_history: list[dict[str, Any]]
    tool_calls: int
    max_tool_calls: int
    rag_results: list[dict[str, Any]] | None
    web_results: list[dict[str, Any]] | None
    reranked_results: list[dict[str, Any]] | None
    needs_web_search: bool
    final_answer: str | None
    error: str | None
