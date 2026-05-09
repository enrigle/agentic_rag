# Agentic RAG

## Langfuse (optional tracing + evals)

This repo has optional Langfuse tracing for LangGraph runs + Ollama calls. When enabled:

- `main.AgenticRAGSystem.query()` returns a `trace_id`
- `eval.py` logs your `[y/n]` rating back to Langfuse as a `human_rating` score

```bash
uv add langfuse
export LANGFUSE_PUBLIC_KEY=...
export LANGFUSE_SECRET_KEY=...
export LANGFUSE_HOST=...   # optional (cloud or self-hosted)
```

## Conversation memory (follow-ups)

To answer follow-up questions, the app keeps a rolling in-memory chat history per `thread_id` and uses it as extra context for retrieval + synthesis. Pass a stable `thread_id` when calling `AgenticRAGSystem.query()`. The Streamlit UI automatically generates a per-session `thread_id`.

Local agentic RAG system using Ollama (llama3.2) and a Notion knowledge base. Optionally uses Groq or Azure OpenAI for fast cloud synthesis and Redis for semantic caching.

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)**
- **[Ollama](https://ollama.com)**
- **[Tesseract](https://github.com/tesseract-ocr/tesseract)** — for OCR on image blocks (`brew install tesseract` on macOS)
- **Groq** *(optional)* — cloud LLM for fast synthesis; set `GROQ_API_KEY` in `.env`; falls back to Ollama if absent
- **Azure OpenAI** *(optional)* — alternative cloud LLM; set `AZURE_OPENAI_API_KEY` + endpoint in `.env`
- **Redis** *(optional)* — semantic cache; cache hits return in < 5 ms (`brew install redis && brew services start redis`)

## Quickstart

```bash
# 1. Install dependencies
uv sync

# 2. Pull models (one-time)
ollama pull llama3.2
ollama pull nomic-embed-text
ollama pull llava          # for image captioning in Notion pages

# 3. Start Ollama (keep running in a separate terminal)
ollama serve

# 4. Set your Notion token (or add to a .env file)
export NOTION_TOKEN=secret_xxx

# 5. Index your Notion workspace
uv run python ingest.py

# 6. Run a query
uv run python main.py
```

## Docker

Runs the Streamlit app and Redis in containers. Ollama stays on your host machine so it keeps GPU/Metal access.

**Prerequisites:**
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Ollama running on your host: `ollama serve`
- Models pulled (one-time): `ollama pull llama3.2 && ollama pull nomic-embed-text`

```bash
# 1. Copy the secrets template and fill in your keys
cp .env.example .env

# 2. Build the image and start all services
docker compose up --build

# 3. Open the app
open http://localhost:8501
```

On subsequent starts (no code changes), skip `--build`:

```bash
docker compose up
```

**Run ingestion inside the container** (indexes your Notion workspace into the Docker volume):

```bash
docker compose run --rm app uv run python ingest.py
```

**Stop everything:**

```bash
docker compose down
```

Your ChromaDB and BM25 indexes are stored in `./data/` on your machine and survive restarts. To wipe them and start fresh: `rm -rf ./data/`.

---

## UI (Streamlit)

```bash
uv run streamlit run app.py
```

The sidebar shows live **service health** (Ollama, Redis, Groq, ChromaDB) and a **Chunking** tool to paste text and preview chunk counts for different `ingestion.chunk_size` / `ingestion.chunk_overlap` values.

### Notion setup (step 4)

1. Go to [notion.so/my-integrations](https://www.notion.so/my-integrations) → create an **Internal Integration** → copy the secret.
2. For each page to index: open the page → **"..."** → **"Connect to"** → select your integration.

You can also store the token in a `.env` file at the project root:
```
NOTION_TOKEN=secret_xxx
```

## Ingestion

```bash
uv run python ingest.py           # incremental — skips unchanged pages, prunes deleted ones
uv run python ingest.py --full    # force re-embed everything
uv run python ingest.py --status  # print chunk/page counts and exit
```

Incremental mode (default) uses `last_edited_time` to skip unchanged pages and removes chunks for pages deleted from Notion. Use `--full` after changing chunking settings.

Ingestion builds both the ChromaDB vector index and a BM25 index (`./bm25_index/`). Queries use both via Reciprocal Rank Fusion — BM25 catches exact keyword matches that vector search can miss, and vector search handles semantic similarity.

Image blocks in Notion pages are processed with OCR (Tesseract) and optionally captioned via `llava`.

## Configuration

Settings live in `config/default.yaml` and are loaded into typed dataclasses at startup:

```yaml
chroma_path: ./chroma_db
bm25_path: ./bm25_index
collection_name: notion_kb
max_tool_calls: 5

llm:
  model: llama3.2
  embed_model: nomic-embed-text
  base_url: http://localhost:11434

retriever:
  min_similarity: 0.50        # cosine similarity cutoff for vector candidates
  top_n: 20                   # RRF candidates passed to reranker
  rrf_k: 60                   # RRF damping constant
  bm25_top_k: 10              # BM25 candidates before merge
  reranker_model: cross-encoder/ms-marco-MiniLM-L-2-v2
  reranker_top_k: 5           # results returned after reranking
  few_shot_max: 3             # max thumbs-up examples injected into the prompt

ingestion:
  chunk_size: 800
  chunk_overlap: 100
  vision_model: llava         # Ollama model used for image captioning

# Optional: Groq for fast cloud synthesis (falls back to Ollama if absent)
groq:
  model: "llama-3.1-8b-instant"
  # api_key: set via GROQ_API_KEY env var (never in config file)

# Optional: Azure OpenAI (alternative to Groq)
azure_openai:
  endpoint: ""                # set via AZURE_OPENAI_ENDPOINT env var
  deployment: "gpt-4o-mini"
  api_version: "2024-02-01"
  # api_key: set via AZURE_OPENAI_API_KEY env var

# Optional: Redis semantic cache
redis:
  url: "redis://localhost:6379"
  ttl_seconds: 3600
  similarity_threshold: 0.95  # cosine similarity required for a cache hit
```

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

### Redis setup (local dev)

```bash
brew install redis && brew services start redis
```

## Eval

```bash
# Edit evals/queries.json with your test queries, then:
uv run python eval.py           # run queries and rate answers interactively
uv run python eval.py --report  # print pass-rate summary from saved results
```

Results are saved to `evals/results.jsonl`.

## Architecture

```mermaid
flowchart TD
  %% ───────────────────────── Online query path ─────────────────────────
  subgraph Online["Online: answer a user query"]
    UI["Client/UI\n`app.py` (Streamlit) or your code"] --> Q["`PipelineCoordinator.query()`"]
    Q -->|cache hit| RC["Redis SemanticCache\n(< 5 ms)"]
    RC --> A
    Q -->|cache miss| H["`HybridRetriever` (RRF merge)"]
    H -->|vector| V["ChromaDB `chroma_db/`"]
    H -->|keyword| K["BM25 `bm25_index/`"]
    H --> R["`CrossEncoderReranker`"]
    R --> S["`Synthesizer`\nGroqLLM · OllamaLLM · AzureOpenAILLM"]
    S -->|store result| RC
    S --> A["Final answer + sources"]
    A --> UI
    Q -->|no KB results| W["Tavily web search"]
    W --> R
  end

  %% ───────────────────────── Offline/ops workflows ─────────────────────
  subgraph Offline["Offline: ingest, eval, feedback loop"]
    N["Notion workspace"] --> I["Ingest\n`scripts/ingest.py` → `NotionIngester`"]
    I --> C["Chunk + embed (Ollama)"]
    C --> V
    C --> K

    E["Eval\n`scripts/eval.py` → `agentic_rag.evaluation.Evaluator`"] --> QF["`evals/queries.json`"]
    E --> RF["writes `evals/results.jsonl`"]

    FB["Feedback (in `app.py`)"] --> ST["`agentic_rag.feedback.store` (`feedback.db`)"]
    ST --> J["Judge failures\n`agentic_rag.feedback.judge`"]
    ST --> O["Optimize\n`agentic_rag.feedback.optimizer`"]
    O --> CFG["updates `config/default.yaml`\n+ writes `feedback_config.json`"]
  end
```

`PipelineCoordinator` runs sources in priority order: `RAGSource` first, `WebSource` only if the KB returns no vector results above `min_similarity`. Conversation memory is per `thread_id`.

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
├── evaluation/        # Evaluator logic (reads/writes `evals/`)
└── utils/             # shared helpers (errors, etc.)
```

```
repo/
├── app.py             # Streamlit UI (query + feedback + background ingest)
├── scripts/
│   ├── ingest.py      # CLI wrapper for NotionIngester
│   └── eval.py        # CLI wrapper for Evaluator
├── evals/
│   ├── queries.json   # eval inputs
│   └── results.jsonl  # eval outputs (generated)
├── tests/
│   ├── unit/
│   └── integration/
└── src/agentic_rag/    # library code (see above)
```

## Development

```bash
ruff format . && ruff check . --fix  # lint
mypy . --strict                       # types
pytest -x                             # tests
```
