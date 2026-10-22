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

- Embeddings are always local (Ollama or sentence-transformers, per `embed_backend`). Cloud LLMs
  are used only for synthesis, never for privacy-sensitive embedding.
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
| Embeddings | Ollama (`nomic-embed-text`) or sentence-transformers | Selected by `embed_backend`; local, never sent to cloud |
| Synthesis (default) | Ollama (`llama3.2`) | Local fallback |
| Synthesis (fast) | Groq (`openai/gpt-oss-20b`) | Cloud; enabled via `GROQ_API_KEY` |
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
│   │   ├── base.py          # Embedder + ChatModel protocols
│   │   ├── ollama.py        # OllamaLLM — local; default embed backend
│   │   ├── sentence_transformers_llm.py  # SentenceTransformersLLM — embed-only backend
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
│   │   ├── bm25.py          # BM25Retriever (loads saved bm25s index; reloads if rebuilt on disk)
│   │   ├── chroma.py        # ChromaVectorStore
│   │   ├── hybrid.py        # HybridRetriever: RRF merge of vector + BM25
│   │   └── reranker.py      # CrossEncoderReranker
│   └── ui/
│       ├── answers.py       # QueryResult → UI dict, safe source links, feedback rows/scores
│       ├── conversations.py # ConversationStore: thread transcripts, 100-thread cap
│       └── ingest.py        # BackgroundIngest: one ingest thread at a time + status
├── config/
│   ├── default.yaml         # Local dev defaults
│   ├── docker.yaml          # Docker-specific overrides (host.docker.internal, redis URL)
│   └── azure.yaml           # Cloud profile: in-process embeddings, Azure OpenAI, no Ollama
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
    │       └─ CrossEncoderReranker.rerank() with the relevance gate (reranker_min_score)
    │           └─ anything left? → stop source chain
    │
    ├─► WebSource.search()          — Tavily (only if nothing relevant came from the KB)
    │       └─ CrossEncoderReranker.rerank() without the gate (reorder only)
    │
    ├─► Synthesizer.synthesize()    — builds prompt with context + chat history → LLM
    │
    ├─► ConversationMemory.append() — stores (query, answer) for thread_id
    │
    └─► SemanticCache.set()         — stores result + embedding in Redis
```

**Why sources stop early:** The first source whose results survive reranking ends the chain. Each
source is reranked on its own, right after its search. Sources with `gated = True` (the KB) drop
candidates whose cross-encoder logit is below `retriever.reranker_min_score` (default −8.0,
calibrated on this KB: real matches score about −1…−7, off-topic queries −10…−12; `null` disables
the gate). An emptied list falls through to the next source. Ungated sources (web) are only
reordered: their scores are on another scale and they have no fallback. The gate is keyed on
`source.gated`, not on list position. Result count, not a raw score, decides the fallback.
This is an explicit design choice over score-based thresholds on retrieval scores. The old threshold approach (`web_search_fallback_score`)
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
├── LLMConfig          (model, embed_model, base_url, request_timeout)
├── RetrieverConfig    (min_similarity, top_n, rrf_k, bm25_top_k, reranker_model, reranker_top_k,
│                       reranker_min_score, few_shot_max)
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

**Embeddings are always local**, regardless of which synthesis provider is active.
`make_embed_llm()` selects the backend from `embed_backend`: `OllamaLLM` (default) or
`SentenceTransformersLLM`. It is shared by the pipeline and the in-app ingest so both sides
write and query vectors in the same embedding space. `OpenAICompatLLM.embed()` raises
`NotImplementedError` by design — embeddings must stay local.

---

## 6. Retrieval Pipeline

### 6.1 Ingestion (write path)

`NotionIngester.ingest()`:
1. Fetch all pages from Notion workspace via `notion.search()` (paginated).
2. Prune chunks for pages deleted from Notion (diff indexed page IDs vs live IDs).
3. For each changed page (detected via `last_edited_time`): fetch block tree recursively up to depth 10.
4. Chunk text with `_chunk_text()` (size=800, overlap=100 by default).
5. Embed each chunk with the configured embed LLM (`make_embed_llm()` — see §5.3).
6. Delete existing chunks for the page, then upsert new ones into ChromaDB.
7. After all pages: rebuild BM25 index from all ChromaDB documents via `BM25Retriever.rebuild()`
   (saves to disk; live retrievers detect the rewrite via `id_map.json` mtime and reload on next
   `search()` — no process restart needed after an in-app sync).

Image blocks: if `ingestion.vision_model` is set, download and caption via Ollama vision model
(e.g. `llava`); otherwise extract visible text with Tesseract OCR. Uses `ollama.AsyncClient` directly — not `Embedder` — because `Embedder.embed()` does
not support multimodal inputs.

Stored document text is `"{page title}\n\n{chunk}"`, so a page title is searchable by both vector
and BM25 even when the chunk body never repeats it.

Chunk IDs: `{page_id}_chunk_{i}`. Metadata stored per chunk: `title`, `source` (URL), `page_id`,
`last_edited_time`, `doc_format`. `doc_format` is `INDEX_DOC_FORMAT` (currently 2 = title-prefixed
chunks). Incremental ingest re-embeds a page whose stored `doc_format` differs, even if its
`last_edited_time` is unchanged; bump the constant whenever the stored text format changes.

`status()` returns `IngestStats(total_chunks, distinct_pages, oldest_edit, newest_edit)`
(`scripts/ingest.py --status`).

### 6.2 Hybrid Search (read path)

`HybridRetriever.search(query_vec, query_text)`:
1. Vector search: `ChromaVectorStore.search()` — cosine similarity, returns results above `min_similarity`.
2. BM25 search: `BM25Retriever.search()` — tokenises query, returns top-k doc IDs.
3. If no vector results → return `[]` (blocks web fallback from being skipped).
4. RRF merge: `_rrf_merge(vector_ids, bm25_ids, k=rrf_k, top_n=top_n)`.
   Score formula: `sum(1 / (k + rank + 1))` over both lists. Default `k=60` (standard RRF constant).
5. Fetch documents for BM25-only IDs from ChromaDB (they have no embedding score, need content).
6. Keep one chunk per source page (the best-ranked), so several chunks of one page don't crowd out
   other pages in the reranker's top-k.
7. Return `SearchResult` list with RRF scores in `score` and the vector cosine (if any) in
   `similarity`.

### 6.3 Reranking

`CrossEncoderReranker.rerank(query, candidates)`:
- Called once per source, on that source's results (see §4).
- Scores `(query, doc_content)` pairs with cross-encoder.
- Returns top `reranker_top_k` (default 5) by cross-encoder score, as copies whose `score` is the
  logit. With a gate set, candidates below `reranker_min_score` are dropped.
- Skips the model when there is no gate and the candidates already fit in `top_k` (it could not
  change which documents are kept).

**Why cross-encoder after RRF:** RRF and vector similarity are bi-encoder — query and doc are
encoded independently. Cross-encoder attends to both simultaneously, much better at relevance
ranking. Running it over a small candidate set (top_n=20 → reranker_top_k=5) is fast enough.

---

## 7. LLM Backends

### 7.1 `Embedder` and `ChatModel` (Protocols)

One Protocol per job: `Embedder.embed(text: str) -> list[float]` and
`ChatModel.chat(prompt: str) -> str` (both async). A backend implements only the jobs it can do;
callers type against the job they use, so mypy rejects pairing an embed-only backend with
synthesis. Errors: `ValueError` for empty/null response, `RuntimeError` for network/backend failure.

| Backend | Embedder | ChatModel |
|---|---|---|
| `OllamaLLM` | yes | yes |
| `SentenceTransformersLLM` | yes | no |
| `GroqLLM`, `AzureOpenAILLM` | no (embeddings stay local) | yes |

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

**Scope:** The key is the query embedding alone, so the coordinator uses the cache only for the
first question of a thread (empty history); follow-ups depend on context and always run the
pipeline. A hit is still appended to memory. The fixed error reply is never cached.

---

## 9. Feedback Loop

### 9.1 Collection

Streamlit UI shows thumbs-up/thumbs-down per answer. Each rating creates a `FeedbackEntry` in
SQLite (`data/feedback.db`) with: `query`, `answer`, `sources`, `top_score`, `rating` (1 or -1),
optional `note`, `category`.

### 9.2 LLM-as-Judge (`feedback/judge.py`)

On thumbs-down, `classify_failure(llm, ...)` asks the pipeline's synthesis LLM (`make_synth_llm`:
Groq → Azure OpenAI → Ollama) to classify the failure. It sends KB content nowhere synthesis does
not already. Categories:
- `retrieval_miss` — sources retrieved are irrelevant
- `synthesis_failure` — sources are relevant but answer is wrong/incomplete
- `missing_content` — topic does not exist in the KB

Result is written back to the SQLite row via `update_category()`.

### 9.3 Optimizer (`feedback/optimizer.py`)

`apply_optimization(entries)` runs three steps:
1. **Retrieval param tuning:** Computes median `top_similarity` (best query–chunk cosine, the same
   scale as `min_similarity`) for thumbs-up and thumbs-down entries; entries without one (web
   answers, legacy rows) are skipped. `top_score` is not used: after reranking it is a
   cross-encoder logit. Suggests `min_similarity = (median_up + median_down) / 2` and edits that
   one line in the active config file (`RAG_CONFIG_PATH` or `config/default.yaml`), keeping
   comments. The UI then rebuilds the cached pipeline.
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

The `Synthesizer` includes chat history in the prompt under "Conversation so far:". Memory keeps
the last `MAX_MESSAGES` (10) messages per thread to bound prompt size; there is no summarisation.

**Why no persistence:** Conversation memory is session-scoped. For production, replace with a
Redis- or PostgreSQL-backed store.

---

## 12. Startup Health Checks

`health.py` exports `run_checks(ollama_url, redis_url, chroma_path, embed_backend,
synth_uses_ollama) -> list[ServiceStatus]`. Ollama is probed whenever it embeds or synthesizes.
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

### 14.4 config/azure.yaml

Profile for a cloud deployment with no Ollama. Select it with `RAG_CONFIG_PATH=config/azure.yaml`.
- `embed_backend: sentence_transformers` with `BAAI/bge-small-en-v1.5` (in-process embeddings).
- `llm.base_url: ""`: no Ollama; answers come from Azure OpenAI (or Groq if `GROQ_API_KEY` is set).
- `azure_openai.endpoint` and `redis.url` (`rediss://…` Azure Cache) are placeholders to fill in.
- Data paths under `/app/data` as in `docker.yaml`.

### 14.5 Required Environment Variables

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

1. Add a class in `src/agentic_rag/llm/` (no base class needed; Protocols are structural).
2. Implement `async chat()` to be a `ChatModel`, `async embed()` to be an `Embedder`, or both.
3. Add a config dataclass to `config.py` and a field on `RAGConfig`.
4. Add an `is_configured()` method to the config dataclass.
5. Wire it in `make_synth_llm()` (chat) or `make_embed_llm()` (embeddings) in `pipeline/rag_pipeline.py`.

### Add a new retrieval source

1. Implement the `BaseSource` protocol (`name: str`, `gated: bool`, `async search(query, ctx) -> list[SearchResult]`).
   Each dict must have keys: `id`, `source`, `title`, `content`, `score`.
2. Add an instance to the `sources` list in `create_pipeline()`.
3. Sources run in order; the first to return non-empty results stops the chain.

### Add a new ingestion source

1. Subclass `BaseIngester` (`async ingest(full) -> int`, `status() -> IngestStats`).
2. After ingestion, call `BM25Retriever.rebuild()` to keep BM25 in sync.

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

- **Embeddings never leave the machine.** `make_embed_llm()` only returns local backends
  (`OllamaLLM` or `SentenceTransformersLLM`), and the pipeline and in-app ingest share it so
  both use the same embedding space. `OpenAICompatLLM.embed()` raises `NotImplementedError`
  by design.
- **Sources are ordered.** `RAGSource` always runs before `WebSource`. The first non-empty result
  wins. Order in `create_pipeline()` is `[RAGSource, WebSource]`.
- **Cache failures are invisible.** `SemanticCache._available = False` after first error; all
  subsequent calls return `None` / no-op silently.
- **Config missing keys use dataclass defaults.** `_parse_sub` ignores unknown YAML keys.
  `load_config()` falls back to `RAGConfig()` (all defaults) if YAML is missing or malformed.
- **BM25 is rebuilt from ChromaDB, not maintained separately.** After ingestion, BM25 is always
  derived from the full ChromaDB document set, so they are always in sync.
