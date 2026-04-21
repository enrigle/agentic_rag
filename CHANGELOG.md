# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
