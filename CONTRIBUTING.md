# Contributing

## Running Locally

```bash
cp .env.example .env   # add NOTION_TOKEN
uv sync
ollama serve           # separate terminal
uv run streamlit run app.py
```

App runs at `http://localhost:8501`. ChromaDB is embedded — no separate server needed.

### Ingest (local)

```bash
uv run python scripts/ingest.py           # incremental
uv run python scripts/ingest.py --full    # force re-embed everything
uv run python scripts/ingest.py --status  # print stats
```

## Running with Docker

Ollama must run on the host — the container reaches it via `host.docker.internal:11434`.

```bash
cp .env.example .env   # add NOTION_TOKEN
docker compose up --build
```

App runs at `http://localhost:8502`. Redis is included and wired automatically.

### Ingest (Docker)

```bash
docker compose exec app uv run python scripts/ingest.py
docker compose exec app uv run python scripts/ingest.py --full
docker compose exec app uv run python scripts/ingest.py --status
```

Data is written to `./data/` on the host (mounted volume) and persists across restarts.

## Development

After cloning, wire up the pre-commit hook once:

```bash
uv run pre-commit install
```

This runs ruff and pytest automatically on every `git commit`. To run the checks manually at any time:

```bash
uv run pre-commit run --all-files
```

Type checking is not part of the hook and must be run separately:

```bash
mypy . --strict
```

## Collaborating

### Reporting issues

Open a GitHub issue with a clear description of the problem, steps to reproduce, and the relevant config (local vs. Docker, model names).

### Submitting a PR

1. Fork the repo and create a branch from `master`.
2. Branch names: `feat/<short-description>`, `fix/<short-description>`, `refactor/<short-description>`.
3. Keep commits focused — one logical change per commit.
4. Commit format: `<type>: <description>` (types: `feat`, `fix`, `refactor`, `test`, `docs`).
5. Run lint, type check, and tests before opening the PR.
6. Reference any related issues in the PR description.

### What belongs where

| Change | Notes |
|--------|-------|
| New retrieval strategy | Add under `src/agentic_rag/retrieval/` |
| New LLM backend | Add under `src/agentic_rag/llm/` |
| New config key | Add to `config/default.yaml` and `config/docker.yaml`, update `src/agentic_rag/config.py` |
| New script | Add under `scripts/`, keep it a thin CLI wrapper over library code |
