# Agentic RAG
##TODO: Add docker
##TODO: Add minikube
##TODO: Add some cloud options to expose the RAG app
##TODO: Add Langfuse for evaluations. Possible to do in the front-end?
##TODO: Is there a way to test the chunking length from the front-end?


Local agentic RAG system using LangGraph, Ollama (llama3.2), and a Notion knowledge base. No cloud credentials required.

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)**
- **[Ollama](https://ollama.com)**
- **[Tesseract](https://github.com/tesseract-ocr/tesseract)** — for OCR on image blocks (`brew install tesseract` on macOS)

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
  min_similarity: 0.35   # cosine similarity cutoff for vector candidates
  top_n: 5               # results returned after RRF merge
  rrf_k: 60              # RRF damping constant
  bm25_top_k: 10         # BM25 candidates before merge

ingestion:
  chunk_size: 800
  chunk_overlap: 100
  vision_model: llava    # Ollama model used for image captioning
```

## Eval

```bash
# Edit evals/queries.json with your test queries, then:
uv run python eval.py           # run queries and rate answers interactively
uv run python eval.py --report  # print pass-rate summary from saved results
```

Results are saved to `evals/results.jsonl`.

## Architecture

```
query → analyze → rag_search → [needs web?] → synthesize → answer
                                    ↓ yes
                               web_search → synthesize
```

The pipeline is a LangGraph `StateGraph` with a circuit breaker (`max_tool_calls`) and `MemorySaver` checkpointing for conversation memory.

```
src/agentic_rag/
├── config.py          # RAGConfig dataclasses + YAML loader
├── models.py
├── ingestion/         # Notion fetching, chunking, OCR, embedding
├── retrieval/         # ChromaDB, BM25, hybrid RRF
├── pipeline/          # LangGraph agent (rag_pipeline.py)
├── llm/               # Ollama LLM abstraction
└── evaluation/        # Evaluator logic
```

## Development

```bash
ruff format . && ruff check . --fix  # lint
mypy . --strict                       # types
pytest -x                             # tests
```
