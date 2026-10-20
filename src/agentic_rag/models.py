"""Shared data models for agentic_rag."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    id: str
    title: str
    source: str
    content: str
    score: float  # ranking score; its scale depends on the stage (cosine, RRF, cross-encoder)
    similarity: float | None = (
        None  # query-chunk cosine from the vector store; None if unknown
    )


@dataclass
class QueryResult:
    answer: str
    sources: list[SearchResult]
    tool_calls_used: int
    latency_ms: float
    trace_id: str | None = (
        None  # Langfuse trace for score_trace(); None when tracing is off
    )


@dataclass
class PipelineContext:
    query: str
    chat_history: list[dict[str, Any]]
    results: list[dict[str, Any]]  # single accumulator; all sources append here
    final_answer: str | None
    error: str | None
    tool_calls: int
    max_tool_calls: int
    query_vec: list[float] | None = None  # embedded once, reused by cache + retrieval
