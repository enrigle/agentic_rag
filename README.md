# Agentic RAG

Local agentic RAG system using LangGraph, Ollama (llama3.2), and a Notion knowledge base. No cloud credentials required.

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)**
- **[Ollama](https://ollama.com)**

## Quickstart

```bash
# 1. Install dependencies
uv sync

# 2. Pull models (one-time)
ollama pull llama3.2
ollama pull nomic-embed-text

# 3. Start Ollama (keep running in a separate terminal)
ollama serve

# 4. Set your Notion token
export NOTION_TOKEN=secret_xxx

# 5. Index your Notion workspace
uv run python ingest.py

# 6. Run a query
uv run python main.py
```

### Notion setup (step 4)

1. Go to [notion.so/my-integrations](https://www.notion.so/my-integrations) → create an **Internal Integration** → copy the secret.
2. For each page to index: open the page → **"..."** → **"Connect to"** → select your integration.

## Ingestion

```bash
uv run python ingest.py           # incremental — skips unchanged pages, prunes deleted ones
uv run python ingest.py --full    # force re-embed everything
uv run python ingest.py --status  # print chunk/page counts and exit
```

Incremental mode (default) uses `last_edited_time` to skip unchanged pages and removes chunks for pages deleted from Notion. Use `--full` after changing chunking settings.

Ingestion builds both the ChromaDB vector index and a BM25 index (`./bm25_index/`). Queries use both via Reciprocal Rank Fusion — BM25 catches exact keyword matches that vector search can miss, and vector search handles semantic similarity.

## Eval

```bash
# Edit evals/queries.json with your test queries, then:
uv run python eval.py           # run queries and rate answers interactively
uv run python eval.py --report  # print pass-rate summary from saved results
```

Results are saved to `evals/results.jsonl`.

## Architecture

```
query → analyze_query → rag_search → [needs web?] → synthesize → answer
                                           ↓ yes
                                      web_search → synthesize
```

## Development

```bash
ruff format . && ruff check . --fix  # lint
mypy . --strict                       # types
pytest -x                             # tests
```
