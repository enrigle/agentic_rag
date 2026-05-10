# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run Streamlit UI
uv run streamlit run app.py

# Ingest Notion workspace
uv run python scripts/ingest.py
uv run python scripts/ingest.py --full    # force re-embed everything
uv run python scripts/ingest.py --status  # print index stats

# Run a query without the UI
uv run python scripts/main.py

# Evaluate
uv run python scripts/eval.py
uv run python scripts/eval.py --report

# Lint & format
ruff format . && ruff check . --fix

# Type check
mypy . --strict

# Run tests
pytest -x
```

## Prerequisites

- [Ollama](https://ollama.com) installed and running: `ollama serve`
- Models pulled: `ollama pull llama3.2 && ollama pull nomic-embed-text`
- `NOTION_TOKEN` env var set (see `.env.example`)
