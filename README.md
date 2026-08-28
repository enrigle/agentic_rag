# Agentic RAG

Local agentic RAG system using Ollama (llama3.2) + Notion knowledge base. Optionally uses Groq or Azure OpenAI for fast cloud synthesis and Redis for semantic caching.

## Prerequisites

- **Python 3.12+** and **[uv](https://docs.astral.sh/uv/getting-started/installation/)**
- **[Ollama](https://ollama.com)** — runs on host in both local and Docker setups
- **[Tesseract](https://github.com/tesseract-ocr/tesseract)** — OCR for image blocks (`brew install tesseract` on macOS)
- **Groq** *(optional)* — cloud LLM for fast synthesis; set `GROQ_API_KEY` in `.env`
- **Azure OpenAI** *(optional)* — alternative cloud LLM; set `AZURE_OPENAI_API_KEY` + endpoint in `.env`
- **Redis** *(optional)* — semantic cache; hits return in < 5 ms (`brew install redis && brew services start redis` on macOS)

## Quickstart

### 0. Setup (both modes)

```bash
git clone https://github.com/enrigle/agentic_rag.git
cd agentic_rag
cp .env.example .env                             # then set NOTION_TOKEN (see Notion setup below)
ollama pull llama3.2 && ollama pull nomic-embed-text
```

Ollama always runs on the host — start it if it isn't running:

```bash
ollama serve   # separate terminal
```

### Option A — local

```bash
uv sync
uv run streamlit run app.py
```

App at `http://localhost:8501`. ChromaDB is embedded — no separate server. Redis optional (`brew install redis && brew services start redis`); without it the app runs with caching disabled.

### Option B — Docker

```bash
docker compose up --build
```

App at `http://localhost:8502` (bound to loopback only — the app has no auth). Redis is included and wired automatically; the container reaches host Ollama via `host.docker.internal:11434`. Config comes from `config/docker.yaml`; index data persists in `./data/` on the host.

### First run

On launch the app starts an incremental Notion ingest in the background — the sidebar shows `⟳ Syncing...` then `✓ Synced · N chunks indexed`. Until sync finishes, answers may come from web search only. You can also ingest manually:

```bash
# local
uv run python scripts/ingest.py            # incremental
uv run python scripts/ingest.py --full     # force re-embed everything
uv run python scripts/ingest.py --status   # print index stats

# Docker
docker compose exec app uv run python scripts/ingest.py
```

---

## UI (Streamlit)

Sidebar shows live **service health** (Ollama, Redis, Groq, ChromaDB) and **Chunking** tool to paste text and preview chunk counts for different `ingestion.chunk_size` / `ingestion.chunk_overlap` values.

### Notion setup

1. Go to [notion.so/my-integrations](https://www.notion.so/my-integrations) → create **Internal Integration** → copy secret.
2. Each page to index: open page → **"..."** → **"Connect to"** → select integration.

Store token in `.env` at project root:
```
NOTION_TOKEN=secret_xxx
```

## Ingestion

Ingestion builds ChromaDB vector index and BM25 index (paths set by `chroma_path` / `bm25_path` in config). Queries use both via Reciprocal Rank Fusion — BM25 catches exact keyword matches vector search can miss. Incremental mode (default) uses `last_edited_time` to skip unchanged pages, prunes deleted ones. Use `--full` after changing chunking settings.

Image blocks processed with OCR (Tesseract), optionally captioned via `llava`.

Ingest commands are in [Quickstart → First run](#first-run).

## Configuration

Settings live in [`config/default.yaml`](config/default.yaml) (local) and [`config/docker.yaml`](config/docker.yaml) (Docker: paths under `/app/data/`, host-gateway Ollama URL, `redis://redis:6379`), loaded into typed dataclasses (`src/agentic_rag/config.py`) at startup. Those files are the authoritative option list; key options:

```yaml
embed_backend: ollama         # or sentence_transformers; used by both queries and ingest (always local)

retriever:
  min_similarity: 0.50        # cosine similarity cutoff for vector candidates
  reranker_min_score: -8.0    # cross-encoder gate (KB-calibrated); null disables
  reranker_top_k: 5           # results returned after reranking

ingestion:
  chunk_size: 800
  chunk_overlap: 100
  vision_model: ""            # Ollama model for image captioning (e.g. llava); empty disables
```

Optional `groq`, `azure_openai`, and `redis` sections enable cloud synthesis and semantic caching; API keys come from env vars (`GROQ_API_KEY`, `AZURE_OPENAI_API_KEY`), never the config file.

### Groq setup

```bash
# 1. Create an account at console.groq.com and generate an API key
# 2. Add to .env (never commit this file):
GROQ_API_KEY=gsk_...
```

### Azure OpenAI setup (alternative to Groq)

```bash
# Add to .env:
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
```

## Eval

```bash
# Edit evals/queries.json with your test queries, then:
uv run python scripts/eval.py           # run queries and rate answers interactively
uv run python scripts/eval.py --report  # print pass-rate summary from saved results
```

Results saved to `evals/results.jsonl`.

## Optional: Langfuse tracing

Optional Langfuse tracing for LangGraph runs + Ollama calls. When enabled:

- `main.AgenticRAGSystem.query()` returns `trace_id`
- `eval.py` logs `[y/n]` rating back to Langfuse as `human_rating` score

```bash
uv add langfuse
export LANGFUSE_PUBLIC_KEY=...
export LANGFUSE_SECRET_KEY=...
export LANGFUSE_HOST=...   # optional (cloud or self-hosted)
```

## Conversation memory (follow-ups)

App keeps rolling in-memory chat history per `thread_id`, uses it as extra context for retrieval + synthesis. Pass stable `thread_id` when calling `AgenticRAGSystem.query()`. Streamlit UI auto-generates per-session `thread_id`.

## Architecture

```mermaid
flowchart TD
  subgraph Online["Online: answer a user query"]
    UI["Client / UI\napp.py or your code"] --> Q["PipelineCoordinator.query()"]
    Q -->|cache hit| RC["Redis SemanticCache\nless than 5 ms"]
    RC --> A
    Q -->|cache miss| H["HybridRetriever\nRRF merge"]
    H -->|vector| V["ChromaDB\nchroma_path"]
    H -->|keyword| K["BM25\nbm25_path"]
    H --> R["CrossEncoderReranker"]
    R --> S["Synthesizer\nGroq / Ollama / Azure OpenAI"]
    S -->|store result| RC
    S --> A["Final answer + sources"]
    A --> UI
    Q -->|no KB results| W["Tavily web search"]
    W --> R
  end

  subgraph Offline["Offline: ingest, eval, feedback loop"]
    N["Notion workspace"] --> I["NotionIngester\ningest.py"]
    I --> C["Chunk + embed\nOllama"]
    C --> V
    C --> K

    E["Evaluator\neval.py"] --> QF["evals/queries.json"]
    E --> RF["evals/results.jsonl"]

    FB["Feedback\napp.py"] --> ST["feedback.store\nfeedback.db"]
    ST --> J["feedback.judge\nfailure classifier"]
    ST --> O["feedback.optimizer"]
    O --> CFG["config/default.yaml\nfeedback_config.json"]
  end
```

`PipelineCoordinator` runs sources in priority order: `RAGSource` first, `WebSource` only if KB returns no vector results above `min_similarity`. Conversation memory is per `thread_id`.

```
src/agentic_rag/
├── config.py          # RAGConfig dataclasses + YAML loader
├── models.py
├── cache/             # SemanticCache (Redis, cosine similarity)
├── ingestion/         # Notion fetching, chunking, embedding (+ image captioning)
├── retrieval/         # ChromaDB, BM25, hybrid RRF, cross-encoder reranker
├── pipeline/          # PipelineCoordinator, sources, synthesizer, memory
├── llm/               # BaseLLM, OllamaLLM, OpenAICompatLLM (Groq + Azure)
├── health.py          # startup dependency checks
├── feedback/          # Store + judge + optimizer (feedback loop)
├── observability/     # Langfuse tracing/scoring
├── evaluation/        # Evaluator logic (reads/writes evals/)
└── utils/             # shared helpers (errors, etc.)
```

```
repo/
├── app.py             # Streamlit UI (query + feedback + background ingest)
├── scripts/
│   ├── ingest.py      # CLI wrapper for NotionIngester
│   ├── eval.py        # CLI wrapper for Evaluator
│   └── main.py        # non-UI query entrypoint
├── config/
│   ├── default.yaml   # local config
│   └── docker.yaml    # Docker config (Ollama + Redis URLs differ)
├── chroma_db/         # ChromaDB vector index (generated; `chroma_path` in config)
├── bm25_index/        # BM25 index (generated; `bm25_path` in config)
├── data/
│   ├── feedback_config.json  # few-shot examples (auto-written by optimizer)
│   └── feedback.db        # SQLite feedback store (gitignored)
├── evals/
│   ├── queries.json   # eval inputs
│   └── results.jsonl  # eval outputs (generated)
├── tests/
│   ├── unit/
│   └── integration/
└── src/agentic_rag/    # library code (see above)
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md).