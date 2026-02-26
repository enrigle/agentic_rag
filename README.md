# Agentic RAG

A local agentic Retrieval-Augmented Generation system using LangGraph for orchestration and Ollama (llama3.2) for inference. No cloud credentials required.

## Architecture

```
query → analyze_query → rag_search → [needs web?] → synthesize → answer
                                            ↓ yes
                                       web_search → synthesize
```

- **analyze_query** — classifies whether the query needs live web data
- **rag_search** — searches internal knowledge base
- **web_search** — searches external sources (conditional)
- **synthesize** — generates a cited answer from retrieved context
- **Circuit breaker** — caps LLM calls at `max_tool_calls` (default: 5)
- **Checkpointing** — `MemorySaver` persists conversation state per `thread_id`

## Prerequisites

1. **Python 3.12+**
2. **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — fast Python package manager
3. **[Ollama](https://ollama.com)** — local LLM runtime

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/enrigle/agentic_rag.git
cd agentic_rag

# 2. Install dependencies
uv sync

# 3. Pull the model (one-time)
ollama pull llama3.2

# 4. Start Ollama daemon (keep this running in a separate terminal)
ollama serve
```

## Run

```bash
uv run python main.py
```

Expected output:

```json
{
  "answer": "...",
  "sources": [...],
  "tool_calls_used": 3,
  "latency_ms": 17135.78
}
```

## Use as a library

```python
import asyncio
from test import AgenticRAGSystem

system = AgenticRAGSystem(model="llama3.2", max_tool_calls=5)
result = asyncio.run(system.query("What is our API endpoint format?"))
print(result["answer"])
```

## Development

```bash
# Lint & format
ruff format . && ruff check . --fix

# Type check
mypy . --strict

# Tests
pytest -x
```
