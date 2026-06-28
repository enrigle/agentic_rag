# Contributing

## Running Locally

```bash
cp .env.example .env   # add NOTION_TOKEN
uv sync
ollama serve           # separate terminal
uv run streamlit run app.py
```

App runs at `http://localhost:8501`. ChromaDB embedded — no separate server needed.

### Ingest (local)

```bash
uv run python scripts/ingest.py           # incremental
uv run python scripts/ingest.py --full    # force re-embed everything
uv run python scripts/ingest.py --status  # print stats
```

## Running with Docker

Ollama run on host — container reaches via `host.docker.internal:11434`.

```bash
cp .env.example .env   # add NOTION_TOKEN
docker compose up --build
```

App runs at `http://localhost:8502`. Redis included, wired automatically.

### Ingest (Docker)

```bash
docker compose exec app uv run python scripts/ingest.py
docker compose exec app uv run python scripts/ingest.py --full
docker compose exec app uv run python scripts/ingest.py --status
```

Data written to `./data/` on host (mounted volume), persists across restarts.

## Development

After cloning, wire pre-commit hook once:

```bash
uv run pre-commit install
```

Runs ruff and pytest on every `git commit`. Manual check:

```bash
uv run pre-commit run --all-files
```

Type check not in hook, run separately:

```bash
mypy . --strict
```

## Collaborating

### Reporting issues

Open GitHub issue: description, repro steps, config (local vs Docker, model names).

### Submitting a PR

1. Fork repo, branch from `master`.
2. Branch names: `feat/<short-description>`, `fix/<short-description>`, `refactor/<short-description>`.
3. Keep commits focused — one logical change per commit.
4. Commit format: `<type>: <description>` (types: `feat`, `fix`, `refactor`, `test`, `docs`).
5. Run lint, type check, tests before PR.
6. Reference related issues in PR.

### What belongs where

| Change | Notes |
|--------|-------|
| New retrieval strategy | Add under `src/agentic_rag/retrieval/` |
| New LLM backend | Add under `src/agentic_rag/llm/` |
| New config key | Add to `config/default.yaml` and `config/docker.yaml`, update `src/agentic_rag/config.py` |
| New script | Add under `scripts/`, thin CLI wrapper over library code |