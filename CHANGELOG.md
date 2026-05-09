# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.9.0] - 2026-05-09

### Added
- **Docker support** — `Dockerfile` (python:3.12-slim-bookworm, two-stage layer cache for deps), `docker-compose.yml` (app + Redis 7), `.env.example`, `.dockerignore`
- **`config/docker.yaml`** — Docker-specific config; Ollama pointed at `host.docker.internal:11434` so the host GPU/Metal is used; Redis URL set to `redis://redis:6379` for the compose-managed service
- **`RAG_CONFIG_PATH` env var** — `load_config()` checks this env var before falling back to `config/default.yaml`; `docker-compose.yml` injects `config/docker.yaml` via this var
- Docker quickstart section in README (build, run, ingestion inside container, data volume persistence)

### Changed
- **`few_shot_max` now configurable** — removed hardcoded `3` from `get_few_shot_examples()`; exposed as `retriever.few_shot_max` in `config/default.yaml` and passed through `apply_optimization()`
- Updated few-shot examples in `feedback_config.json` — replaced Claude hooks / weather / Anthropic founder queries with autoresearch, Datadog, and agentic scaling

## [0.8.0] - 2026-05-01

### Added
- **Groq LLM support** — `GroqLLM` backed by Groq's OpenAI-compatible API (`llama-3.1-8b-instant` default); no separate `groq` package, uses `openai` client with `base_url`
- **`OpenAICompatLLM` base class** — merged `azure_openai.py` and `groq_llm.py` into `llm/openai_compat.py`; shared `chat()` and `embed()` logic; `AzureOpenAILLM` and `GroqLLM` are thin subclasses with only `__init__`
- **`is_configured()` on `GroqConfig` and `AzureOpenAIConfig`** — checks config field and env var; replaces exception-driven provider detection in `rag_pipeline.py`
- **Startup health checks** — `health.py` probes Ollama, Redis, Groq, and ChromaDB at launch; results shown in Streamlit sidebar as coloured status indicators
- **`SemanticCache` circuit breaker** — `_available` flag flips `False` on first Redis failure; subsequent calls skip Redis silently instead of logging a full traceback on every query
- `load_dotenv()` called at app startup so `.env` credentials are always loaded

### Changed
- **Web search fallback logic overhauled** — previous score-based threshold (`web_search_fallback_score: 0.4`) was permanently unreachable because RRF scores cap at ~0.033; replaced with result-count check: any non-empty RAGSource result stops the source chain
- **`HybridRetriever` returns empty when no vector results** — BM25 always returns candidates, which previously blocked web fallback for out-of-KB queries; now if ChromaDB finds nothing above `min_similarity`, the retriever returns `[]` regardless of BM25 hits
- **`min_similarity` raised from `0.35` to `0.50`** — reduces false-positive KB matches for loosely-related queries
- **`PipelineCoordinator`** — removed `threshold` parameter; web fallback is now `if new_results: break`
- **Langfuse `observation` context manager** — fixed double-yield bug that caused `RuntimeError: generator didn't stop after throw()` when an exception occurred inside the `with` block
- `GroqConfig` added to `RAGConfig`; `web_search_fallback_score` removed from `RetrieverConfig`
- `groq` package removed from dependencies (functionality covered by `openai>=1.0`)

### Fixed
- `SemanticCache` logged a full Redis traceback on every query when Redis was unreachable; now logs one `WARNING` line and goes silent for the session
- Provider selection in `rag_pipeline.py` used `try/except ValueError` to detect missing credentials; replaced with explicit `is_configured()` checks

## [0.7.0] - 2026-04-29

### Added
- `AzureOpenAILLM` — new `BaseLLM` implementation backed by Azure OpenAI (`llm/azure_openai.py`); used for synthesis while embeddings remain local via `OllamaLLM`
- `SemanticCache` — Redis-backed cache keyed by embedding vector; `get()` uses cosine similarity (SCAN + best-match) so near-identical queries skip the full pipeline (`cache/semantic_cache.py`)
- `AzureOpenAIConfig` and `RedisConfig` dataclasses in `config.py`; corresponding sections in `config/default.yaml`
- `PipelineCoordinator` now accepts an optional `cache: SemanticCache | None` — cache hit returns in < 5 ms; cache miss stores the result after synthesis

### Changed
- `pyproject.toml` adds `openai>=1.0`, `redis[asyncio]>=5.0`, `numpy>=1.26` as explicit dependencies

### Performance
- Expected latency without cache: 2–4 s (down from 20–40 s on CPU) when Azure OpenAI synthesis is configured
- Expected latency on cache hit: < 5 ms for any previously seen or semantically similar query

## [0.6.0] - 2026-04-25

### Added
- `PipelineCoordinator` — plain async coordinator replacing LangGraph `StateGraph`
- `BaseSource` protocol with `RAGSource` and `WebSource` implementations (`pipeline/sources.py`)
- `Synthesizer` class with chat history injection (`pipeline/synthesizer.py`)
- `ConversationMemory` replacing LangGraph `MemorySaver` (`pipeline/memory.py`)
- `PipelineContext` dataclass — single `results` accumulator, no per-source fields

### Changed
- Routing is now score-based: `RAGSource` runs first; `WebSource` (Tavily) is called only when best score falls below `web_search_fallback_score` threshold
- `rag_pipeline.py` reduced to a thin `create_pipeline()` factory
- `AgentState` TypedDict (10 fields) replaced by `PipelineContext` dataclass (7 fields)
- Adding a new retrieval source now requires only a new `BaseSource` class — no state fields, no graph edges

### Removed
- LangGraph dependency (`StateGraph`, `MemorySaver`, conditional edges)
- `analyze_query` LLM classifier node — eliminated unreliable LLM-based routing
- `needs_web_search` flag and per-source result fields (`rag_results`, `web_results`, `reranked_results`)
- `RAGPipeline` class

## [0.5.0] - 2026-04-22

### Added
- Cross-encoder reranking node (`rerank`) using `cross-encoder/ms-marco-MiniLM-L-2-v2`
- `CrossEncoderReranker` class in `src/agentic_rag/retrieval/reranker.py`
- `reranked_results` field in `AgentState`
- Unit tests for `CrossEncoderReranker` (empty input, top-k, score ordering, fewer candidates than top-k)

### Changed
- RAG candidate pool widened from `top_n=5` to `top_n=20` to give the reranker more signal
- Both retrieval paths (RAG-only and RAG+web) now converge at `rerank` before `synthesize`
- `synthesize` and `query()` sources now reflect reranked results only

## [0.4.0] - 2026-04-20

### Added
- Memory layer with tests
- Observability integration via Langfuse
- LLM-as-judge for failure classification
- Feedback optimizer (retrieval tuning, few-shot, KB gaps)
- SQLite feedback store
- Feedback UI in Streamlit app (rating buttons, sidebar)
- Few-shot injection into synthesize step; `content`/`score`/`top_score` in `query()` return

### Changed
- Web search backend switched to Tavily

### Fixed
- Factual questions + `web_search_fallback_score` handling
- Feedback store validation and type safety
- Judge JSON guard and test coverage
- Optimizer error handling and score filter
- Per-example few-shot resilience; anchored `feedback_config` path

## [0.3.0] - 2026-04-10

### Added
- Streamlit UI with background ingest thread and sidebar ingest status
- `_sync_state` / `_run_ingest` for background ingestion
- `enter` key support in search UI
- Initial few-shot examples from optimizer output

### Changed
- `_sync_state` narrowed to TypedDict for type safety

### Fixed
- Idle sync state handling in sidebar; cast chunks to `int`
- Unused imports; mypy cast; sync state reset on run
- Minimum similarity score set to 0.05
- Web search routing and Ollama async client lifecycle
- `ollama.AsyncClient` created per-call as async context manager

## [0.2.0] - 2026-03-15

### Added
- RAGPipeline, Evaluator, and CLI scripts
- Ingestion layer: `BaseIngester` ABC, chunker pure functions, `NotionIngester`
- Retrieval layer: `BaseVectorStore`, `BM25Retriever`, `ChromaVectorStore`, `HybridRetriever`
- LLM layer: `BaseLLM` ABC and `OllamaLLM` implementation
- Image text extraction (pytesseract)
- Hybrid BM25 search
- Evaluations and `MIN_SIMILARITY` threshold
- Notion recursive ingestion

### Fixed
- ABC contract and correctness in retrieval layer
- `BaseVectorStore.search` return type corrected to `list[SearchResult]`
- RAGConfig constructor guard; `_parse_sub` types; dev deps consolidated
- Circuit breaker abort paths, web_search `KeyError`, lazy pipeline init

## [0.1.0] - 2026-02-01

### Added
- Foundation layer: `src/` layout, config, models, directory skeleton
- Agentic RAG with LangGraph + Ollama
- `analyze_query` → `rag_search` → conditional `web_search` → `synthesize` state machine
- `MemorySaver` checkpointing keyed by `thread_id`
