# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run
uv run python test.py

# Lint & format
ruff format . && ruff check . --fix

# Type check
mypy . --strict

# Run tests
pytest -x
```

## Prerequisites

- [Ollama](https://ollama.com) installed and running: `ollama serve`
- Model pulled: `ollama pull llama3.2`

## Architecture

Single-file skeleton: `test.py` implements an agentic RAG system with two retrieval paths (internal KB + web) orchestrated via LangGraph, with Ollama (llama3.2) as the local LLM backend.

### State machine (`AgentState` → `AgenticRAGSystem`)

Data flows through a LangGraph `StateGraph` via shared `AgentState` (TypedDict):

```
query → analyze_query → rag_search → [should_web_search?] → synthesize → END
                                              ↓ yes
                                         web_search → synthesize → END
```

**Nodes:**
- `analyze_query` — Ollama call to classify intent, sets `needs_web_search`
- `rag_search` — Mock internal knowledge base search
- `web_search` — Mock external search (conditional)
- `synthesize` — Ollama call to generate final answer with citations

**Routing:**
- `should_web_search()` is the conditional edge after `rag_search`; returns `"web_search"` or `"synthesize"`

**Circuit breaker:** `tool_calls` counter checked against `max_tool_calls` (default 5) before each node increments.

**Checkpointing:** `MemorySaver` provides conversation memory keyed by `thread_id`.

### Ollama integration

- Client: `ollama.AsyncClient()` (connects to local daemon at `http://localhost:11434`)
- Default model: `llama3.2`
- All LLM calls are local — no cloud credentials required

### `query()` return shape

```python
{"answer": str, "sources": list, "tool_calls_used": int, "latency_ms": float}
```

## Implementation status

All methods are fully implemented. Run `uv run python test.py` to execute the pipeline end-to-end.
