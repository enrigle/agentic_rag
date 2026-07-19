# Contributing

## Running the app

See [README → Quickstart](README.md#quickstart) for local and Docker setup, run, and ingest commands.

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