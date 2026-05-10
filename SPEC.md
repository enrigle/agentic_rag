# Agentic RAG — System Specification

> **Purpose of this file:** Machine-readable architecture spec. An LLM reading this file should be
> able to reconstruct the application from scratch. Every design decision that is non-obvious is
> annotated with a "Why:" line.

---

## 1. What This System Is

A local-first, agentic Retrieval-Augmented Generation system. It answers natural-language queries
by searching an internal Notion knowledge base first, then falling back to live web search (Tavily)
only when the KB has nothing relevant. Answers are synthesised by an LLM with inline source
citations.

**Core properties:**

- Embeddings are always local (Ollama). Cloud LLMs are used only for synthesis, never for privacy-
  sensitive embedding.
- The system is entirely async (`asyncio`).
- All external dependencies (Redis, Ollama, Tavily, Langfuse) fail open — the pipeline degrades
  gracefully rather than crashing.
- Configuration is YAML-driven; the same codebase runs locally and in Docker with no code changes.

---

## 2. Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| Language | Python 3.12 | `from __future__ import annotations` everywhere |
| Package manager | `uv` | `pyproject.toml`, `uv.lock` |
| Vector DB | ChromaDB (persistent) | cosine space, local on-disk |
| Keyword search | `bm25s` | Saved to disk; rebuilt after every ingestion |
| Embeddings | Ollama (`nomic-embed-text`) | Local, never sent to cloud |
| Synthesis (default) | Ollama (`llama3.2`) | Local fallback |
| Synthesis (fast) | Groq (`llama-3.1-8b-instant`) | Cloud; enabled via `GROQ_API_KEY` |
| Synthesis (enterprise) | Azure OpenAI (`gpt-4o-mini`) | Cloud; enabled via endpoint + key |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-2-v2` | sentence-transformers |
| Cache | Redis 7 (semantic, embedding-based) | Fails open when unreachable |
| Web search | Tavily | Enabled via `TAVILY_API_KEY` |
| Observability | Langfuse v4 (OpenTelemetry) | No-op if credentials absent |
| Feedback store | SQLite (`feedback.db`) | Thumbs-up/down from UI |
| Failure classifier | Ollama LLM-as-judge | Classifies thumbs-down into 3 categories |
| UI | Streamlit | `app.py` |
| Ingestion source | Notion API | `notion-client` async |
| Containerisation | Docker + Compose | App + Redis; Ollama stays on host |

---

## 3. Directory Layout

```
agentic_rag/
├── src/agentic_rag/
│   ├── config.py            # RAGConfig + loader
│   ├── models.py            # SearchResult, QueryResult, PipelineContext
│   ├── health.py            # Startup probes (Ollama, Redis, Groq, ChromaDB)
│   ├── cache/
│   │   └── semantic_cache.py  # Redis-backed embedding-similarity cache
│   ├── evaluation/
│   │   └── evaluator.py     # CLI evaluator over evals/queries.json
│   ├── feedback/
│   │   ├── judge.py         # LLM-as-judge: classifies failure categories
│   │   ├── optimizer.py     # Tunes config from feedback; writes few-shot examples
│   │   └── store.py         # SQLite CRUD for FeedbackEntry
│   ├── ingestion/
│   │   ├── base.py          # BaseIngester protocol
│   │   ├── chunker.py       # _chunk_text, _extract_plain_text, _get_title
│   │   └── notion.py        # NotionIngester: Notion → ChromaDB + BM25
│   ├── llm/
│   │   ├── base.py          # BaseLLM (abstract: chat + embed)
│   │   ├── ollama.py        # OllamaLLM — local, used for embeddings always
│   │   └── openai_compat.py # OpenAICompatLLM base + AzureOpenAILLM + GroqLLM
│   ├── observability/
│   │   └── langfuse.py      # observation() context manager + score_trace()
│   ├── pipeline/
│   │   ├── coordinator.py   # PipelineCoordinator: orchestrates query flow
│   │   ├── memory.py        # ConversationMemory (in-memory, keyed by thread_id)
│   │   ├── rag_pipeline.py  # create_pipeline() factory
│   │   ├── sources.py       # BaseSource protocol, RAGSource, WebSource
│   │   └── synthesizer.py   # Synthesizer: formats prompt + calls LLM
│   ├── retrieval/
│   │   ├── base.py          # BaseVectorStore + BaseKeywordRetriever protocols
│   │   ├── bm25.py          # BM25Retriever (loads saved bm25s index)
│   │   ├── chroma.py        # ChromaVectorStore
│   │   ├── hybrid.py        # HybridRetriever: RRF merge of vector + BM25
│   │   └── reranker.py      # CrossEncoderReranker
│   └── utils/
│       └── errors.py        # Custom exception types
├── config/
│   ├── default.yaml         # Local dev defaults
│   └── docker.yaml          # Docker-specific overrides (host.docker.internal, redis URL)
├── app.py                   # Streamlit UI entry point
├── scripts/
│   ├── ingest.py            # CLI wrapper for NotionIngester
│   ├── eval.py              # CLI wrapper for Evaluator
│   └── main.py              # Non-UI query entrypoint
├── data/
│   ├── chroma_db/           # ChromaDB vector index (generated)
│   ├── bm25_index/          # BM25 index (generated)
│   ├── feedback_config.json # Auto-written few-shot examples (by optimizer)
│   └── feedback.db          # SQLite feedback store (gitignored)
├── Dockerfile               # python:3.12-slim-bookworm, two-stage layer cache
├── docker-compose.yml       # app + redis services
├── .env.example             # Required env var template
└── evals/
    ├── queries.json         # Evaluation queries + expected keywords
    └── results.jsonl        # Append-only rated results
```

---

## 4. Query Data Flow

```
user query
    │
    ▼
PipelineCoordinator.query()
    │
    ├─► SemanticCache.get()         — cosine similarity over all Redis keys
    │       └─ cache hit → return immediately (< 5 ms)
    │
    ├─► RAGSource.search()          — HybridRetriever → ChromaDB + BM25 → RRF merge
    │       └─ results found? → stop source chain
    │
    ├─► WebSource.search()          — Tavily (only if RAGSource returned nothing)
    │
    ├─► CrossEncoderReranker.rerank()  — re-scores all candidates with cross-encoder
    │
    ├─► Synthesizer.synthesize()    — builds prompt with context + chat history → LLM
    │
    ├─► ConversationMemory.append() — stores (query, answer) for thread_id
    │
    └─► SemanticCache.set()         — stores result + embedding in Redis
```

**Why sources stop early:** The first source that returns any results ends the chain. This is an
explicit design choice over score-based thresholds. The old threshold approach (`web_search_fallback_score`)
failed because RRF scores cap at ~0.033 — the threshold was permanently unreachable. Result-count
check is reliable: if ChromaDB finds nothing above `min_similarity`, `HybridRetriever` returns `[]`,
which lets `WebSource` run.

**Why `min_similarity` gates the whole hybrid result, not just the vector leg:** BM25 always
produces candidates (it has no similarity threshold). Without the vector gate, BM25-only results
would pass through for queries completely outside the KB. If `ChromaVectorStore` returns no results
above `min_similarity`, `HybridRetriever` returns `[]` regardless of BM25 hits.

---

## 5. Config System

### 5.1 Structure

`RAGConfig` is a frozen hierarchy of Python `dataclass`es:

```python
RAGConfig
├── LLMConfig          (model, embed_model, base_url)
├── RetrieverConfig    (min_similarity, top_n, rrf_k, bm25_top_k, reranker_model, reranker_top_k, few_shot_max)
├── IngestionConfig    (chunk_size, chunk_overlap, vision_model)
├── GroqConfig         (model, api_key) + is_configured()
├── AzureOpenAIConfig  (endpoint, api_key, deployment, api_version) + is_configured()
└── RedisConfig        (url, ttl_seconds, similarity_threshold)
```

### 5.2 Loading Order

`load_config(path=None)`:
1. If `path` is explicitly passed, use it.
2. Else check `RAG_CONFIG_PATH` env var.
3. Else fall back to `config/default.yaml` (relative to package root).

Unknown YAML keys are silently ignored (`_parse_sub` filters to known dataclass fields).
Missing keys use dataclass defaults. Invalid YAML falls back to all defaults with a warning.

**Why `RAG_CONFIG_PATH`:** Docker Compose sets this env var to `config/docker.yaml` so the
containerised app uses container-appropriate paths (`/app/data/...`) and the Redis URL
(`redis://redis:6379`) without any code changes or build-time baking.

### 5.3 Provider Detection

Cloud LLM selection in `create_pipeline()` uses explicit `is_configured()` checks:

```
GROQ_API_KEY set?        → GroqLLM for synthesis
AZURE_OPENAI_API_KEY set + endpoint set?  → AzureOpenAILLM for synthesis
otherwise                → OllamaLLM for synthesis (same as embeddings)
```

**Why `is_configured()` instead of try/except:** Previous code caught `ValueError` to detect
missing credentials. That is exception-driven control flow — slow and unpredictable. `is_configured()`
checks env vars + config fields directly.

**Embeddings always use OllamaLLM** regardless of which synthesis provider is active.
`OpenAICompatLLM.embed()` raises `NotImplementedError` by design — embeddings must stay local.

---

## 6. Retrieval Pipeline

### 6.1 Ingestion (write path)

`NotionIngester.ingest()`:
1. Fetch all pages from Notion workspace via `notion.search()` (paginated).
2. Prune chunks for pages deleted from Notion (diff indexed page IDs vs live IDs).
3. For each changed page (detected via `last_edited_time`): fetch block tree recursively up to depth 10.
4. Chunk text with `_chunk_text()` (size=800, overlap=100 by default).
5. Embed each chunk with `OllamaLLM.embed()`.
6. Delete existing chunks for the page, then upsert new ones into ChromaDB.
7. After all pages: rebuild BM25 index from all ChromaDB documents and save to disk.

Image blocks: if `ingestion.vision_model` is set, download and caption via Ollama vision model
(e.g. `llava`). Uses `ollama.AsyncClient` directly — not `BaseLLM` — because `BaseLLM.embed()` does
not support multimodal inputs.

Chunk IDs: `{page_id}_chunk_{i}`. Metadata stored per chunk: `title`, `source` (URL), `page_id`,
`last_edited_time`.

### 6.2 Hybrid Search (read path)

`HybridRetriever.search(query_vec, query_text)`:
1. Vector search: `ChromaVectorStore.search()` — cosine similarity, returns results above `min_similarity`.
2. BM25 search: `BM25Retriever.search()` — tokenises query, returns top-k doc IDs.
3. If no vector results → return `[]` (blocks web fallback from being skipped).
4. RRF merge: `_rrf_merge(vector_ids, bm25_ids, k=rrf_k, top_n=top_n)`.
   Score formula: `sum(1 / (k + rank + 1))` over both lists. Default `k=60` (standard RRF constant).
5. Fetch documents for BM25-only IDs from ChromaDB (they have no embedding score, need content).
6. Return `SearchResult` list with RRF scores.

### 6.3 Reranking

`CrossEncoderReranker.rerank(query, candidates)`:
- Takes all results from all sources (post-coordinator accumulation).
- Scores `(query, doc_content)` pairs with cross-encoder.
- Returns top `reranker_top_k` (default 5) by cross-encoder score, replacing the RRF score.

**Why cross-encoder after RRF:** RRF and vector similarity are bi-encoder — query and doc are
encoded independently. Cross-encoder attends to both simultaneously, much better at relevance
ranking. Running it over a small candidate set (top_n=20 → reranker_top_k=5) is fast enough.

---

## 7. LLM Backends

### 7.1 `BaseLLM` (abstract)

Two methods: `async chat(prompt: str) -> str` and `async embed(text: str) -> list[float]`.
Errors: `ValueError` for empty/null response, `RuntimeError` for network/backend failure.

### 7.2 `OllamaLLM`

Wraps `ollama.AsyncClient`. Used for both embedding and synthesis when no cloud key is set.
`embed()` calls `ollama.embed()`, `chat()` calls `ollama.chat()`.

### 7.3 `OpenAICompatLLM` (base for cloud providers)

Shared `chat()` implementation using `openai.AsyncOpenAI`. `embed()` raises `NotImplementedError`.

- `AzureOpenAILLM`: `openai.AsyncAzureOpenAI(azure_endpoint=..., api_version=...)`.
- `GroqLLM`: `openai.AsyncOpenAI(base_url="https://api.groq.com/openai/v1")`.
  No `groq` package — uses the OpenAI SDK with a custom base URL.

**Why no `groq` package:** Groq exposes an OpenAI-compatible API. Using `openai>=1.0` keeps the
dependency count low and the client identical to Azure.

---

## 8. Semantic Cache

`SemanticCache` stores `(embedding, QueryResult)` in Redis hashes under keys `cache:{sha256_of_vec}`.

**Lookup:** Scan all `cache:*` keys; compute cosine similarity between query embedding and stored
embedding; return best match if similarity ≥ `similarity_threshold` (default 0.95).

**Why cosine similarity instead of exact key match:** Semantically equivalent queries ("what is X?"
vs "explain X") produce slightly different embeddings. Exact match would miss these. 0.95 threshold
is tight enough to avoid false positives while catching near-duplicates.

**Circuit breaker:** `_available` flag. First Redis error sets it to `False` for the session.
Subsequent calls skip Redis silently (one `WARNING` log, no repeated tracebacks).

**TTL:** `redis.expire(key, ttl_seconds)` applied after every `set()`. Default 1 hour.

---

## 9. Feedback Loop

### 9.1 Collection

Streamlit UI shows thumbs-up/thumbs-down per answer. Each rating creates a `FeedbackEntry` in
SQLite (`data/feedback.db`) with: `query`, `answer`, `sources`, `top_score`, `rating` (1 or -1),
optional `note`, `category`.

### 9.2 LLM-as-Judge (`feedback/judge.py`)

On thumbs-down, `classify_failure()` calls Ollama locally to classify the failure into one of:
- `retrieval_miss` — sources retrieved are irrelevant
- `synthesis_failure` — sources are relevant but answer is wrong/incomplete
- `missing_content` — topic does not exist in the KB

Result is written back to the SQLite row via `update_category()`.

### 9.3 Optimizer (`feedback/optimizer.py`)

`apply_optimization(entries)` runs three steps:
1. **Retrieval param tuning:** Computes median top_score for thumbs-up and thumbs-down entries.
   Suggests `min_similarity = (median_up + median_down) / 2`. Writes to `config/default.yaml`.
2. **Few-shot examples:** Takes the most recent thumbs-up `(query, answer)` pairs (up to
   `few_shot_max`). Writes to `data/feedback_config.json`. These are loaded at synthesis time.
3. **KB gaps:** Returns queries classified as `missing_content` for the user to act on.

**Why median, not mean:** Score distributions can be skewed by outliers. Median is more robust.

---

## 10. Observability

`observability/langfuse.py` provides a single `observation()` context manager. If Langfuse
credentials are absent (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`), every call is a no-op
(yields `None`). No code changes needed to enable/disable tracing.

Traced spans in the query path:
- `pipeline.query` (type: span)
- `rag_search` (type: retriever)
- `web_search` (type: span)
- `synthesize` (type: generation)

`score_trace()` attaches numeric scores to a trace ID — used by the UI after feedback is saved.

**Why `@lru_cache` on `get_client()`:** The Langfuse client is expensive to construct and holds
connection state. One instance per process is correct.

---

## 11. Conversation Memory

`ConversationMemory` is an in-memory dict: `thread_id → list[{role, content}]`.

`get(thread_id)` returns the history list (empty if new thread). `append(thread_id, query, answer)`
appends both user and assistant turns. No persistence — memory is lost on process restart.

The `Synthesizer` includes the last N messages from chat history in the prompt under
"Conversation so far:". No summary or truncation — full history is passed each time.

**Why no persistence:** Conversation memory is session-scoped. For production, replace with a
Redis- or PostgreSQL-backed store.

---

## 12. Startup Health Checks

`health.py` exports `run_checks(ollama_url, redis_url, chroma_path) -> list[ServiceStatus]`.
Runs Ollama and Redis probes in parallel (`asyncio.gather`). Groq and ChromaDB are sync checks
(env var presence + path existence). Results are rendered in the Streamlit sidebar as coloured
status indicators.

`ServiceStatus` fields: `name`, `ok: bool`, `detail: str` (human-readable fix hint when `ok=False`).

---

## 13. Evaluation

`evaluation/evaluator.py`:
- Reads `evals/queries.json`: list of `{id, query, expected_keywords}`.
- For each query: runs `PipelineCoordinator.query()`, prints answer + sources.
- User rates interactively: `y` / `n` / `s` (skip).
- Appends `{id, query, answer, sources, rating, note, timestamp}` to `evals/results.jsonl`.
- `Evaluator.report()` prints pass-rate summary per query ID.

---

## 14. Docker & Infrastructure

### 14.1 Dockerfile

Base: `python:3.12-slim-bookworm`. System deps: `tesseract-ocr` (for OCR), `libgomp1` (OpenMP for
ONNX reranker), `curl`.

Two-stage layer cache:
1. Copy `pyproject.toml` only → `uv sync --no-install-project` (installs third-party deps).
2. Copy full source → `uv sync` (installs local package). Re-runs only when source changes.

PyTorch CPU index used (`https://download.pytorch.org/whl/cpu`) to avoid 2 GB CUDA binaries.

Entrypoint: `uv run --no-sync streamlit run app.py --server.address=0.0.0.0 --server.port=8501`

### 14.2 docker-compose.yml

Two services:
- `app`: built from `Dockerfile`. Mounts `./data/chroma_db`, `./data/bm25_index`, `./evals`.
  Sets `RAG_CONFIG_PATH=config/docker.yaml`. `extra_hosts: host.docker.internal:host-gateway`
  so Ollama on the Mac host is reachable at `http://host.docker.internal:11434`.
- `redis`: `redis:7-alpine`. Named volume `redis_data` (survives `docker compose down`).

Port: host `8502` → container `8501` (avoids conflict with local Streamlit).

### 14.3 config/docker.yaml

Key differences from `config/default.yaml`:
- `chroma_path: /app/data/chroma_db` (absolute, maps to host volume)
- `bm25_path: /app/data/bm25_index`
- `llm.base_url: http://host.docker.internal:11434`
- `redis.url: redis://redis:6379`

### 14.4 Required Environment Variables

```
NOTION_TOKEN          Notion integration token (required for ingestion)
TAVILY_API_KEY        Web search (optional — web fallback disabled if absent)
GROQ_API_KEY          Fast cloud synthesis (optional)
AZURE_OPENAI_API_KEY  Azure synthesis (optional, also needs endpoint in config)
LANGFUSE_PUBLIC_KEY   Tracing (optional)
LANGFUSE_SECRET_KEY   Tracing (optional)
LANGFUSE_HOST         Tracing host override (optional, for self-hosted)
RAG_CONFIG_PATH       Config file override (set by Docker Compose automatically)
```

---

## 15. Extension Points

### Add a new LLM backend

1. Subclass `BaseLLM` in `src/agentic_rag/llm/`.
2. Implement `async chat()` and `async embed()` (or raise `NotImplementedError` for embed if cloud-only).
3. Add a config dataclass to `config.py` and a field on `RAGConfig`.
4. Add an `is_configured()` method to the config dataclass.
5. Wire it in `create_pipeline()` in `pipeline/rag_pipeline.py`.

### Add a new retrieval source

1. Implement the `BaseSource` protocol (`name: str`, `async search(query, ctx) -> list[dict]`).
   Each dict must have keys: `id`, `source`, `title`, `content`, `score`.
2. Add an instance to the `sources` list in `create_pipeline()`.
3. Sources run in order; the first to return non-empty results stops the chain.

### Add a new ingestion source

1. Implement `BaseIngester` protocol (`async ingest() -> int`, `status() -> dict`).
2. After ingestion, call `_rebuild_bm25()` or its equivalent to keep BM25 in sync.

### Add a new vector store

1. Implement `BaseVectorStore` protocol: `async search(vec, top_k) -> list[SearchResult]`,
   `async fetch_by_ids(ids) -> list[SearchResult]`.
2. Pass it to `HybridRetriever` in `create_pipeline()`.

### Tune retrieval quality

- `retriever.min_similarity` (0.0–1.0): raise to reduce false KB matches; lower to widen recall.
- `retriever.top_n`: candidate pool size after RRF merge, before reranking.
- `retriever.reranker_top_k`: final answer window shown to synthesizer.
- `retriever.bm25_top_k`: candidates fetched from each of vector and BM25.
- `retriever.rrf_k`: RRF smoothing constant (60 is the field-standard default).

The `feedback/optimizer.py` auto-tunes `min_similarity` from thumbs-up/down signal.

---

## 16. Key Invariants

- **Embeddings never leave the machine.** `OllamaLLM` is always used for `embed()`.
  `OpenAICompatLLM.embed()` raises `NotImplementedError` by design.
- **Sources are ordered.** `RAGSource` always runs before `WebSource`. The first non-empty result
  wins. Order in `create_pipeline()` is `[RAGSource, WebSource]`.
- **Cache failures are invisible.** `SemanticCache._available = False` after first error; all
  subsequent calls return `None` / no-op silently.
- **Config missing keys use dataclass defaults.** `_parse_sub` ignores unknown YAML keys.
  `load_config()` falls back to `RAGConfig()` (all defaults) if YAML is missing or malformed.
- **BM25 is rebuilt from ChromaDB, not maintained separately.** After ingestion, BM25 is always
  derived from the full ChromaDB document set, so they are always in sync.
