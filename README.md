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

# 3. Pull the LLM and embedding models (one-time)
ollama pull llama3.2
ollama pull nomic-embed-text

# 4. Start Ollama daemon (keep this running in a separate terminal)
ollama serve
```

### Notion knowledge base (RAG)

The RAG path queries a ChromaDB vector store populated from your Notion workspace.

**One-time Notion setup:**

1. Go to [https://www.notion.so/my-integrations](https://www.notion.so/my-integrations) and create a new **Internal Integration**. Copy the **Internal Integration Secret** (starts with `secret_`).
2. For each Notion page or database you want indexed: open the page → click **"..."** (top-right) → **"Connect to"** → select your integration.
3. Export the token:

```bash
export NOTION_TOKEN=secret_xxx
```

**Ingest (run once, re-run to refresh):**

```bash
uv run python ingest.py
# Expected: "Done. Indexed N pages, M chunks into ChromaDB at './chroma_db'."
```

This writes a local `./chroma_db/` directory used by `main.py` at query time. No re-ingestion is needed unless your Notion content changes.

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
