# Feedback Loop & Quality Improvement — Design Spec
_Date: 2026-04-06_

## Context

The system is a local, single-user agentic RAG (Retrieval-Augmented Generation) app backed by Notion as a knowledge base, ChromaDB + BM25 for retrieval, and Ollama (llama3.2) as the LLM. A manual CLI-based eval system (`eval.py`) exists but is unused. There is no in-app feedback mechanism.

**Goal:** Add a simple, end-to-end feedback loop that (1) collects 👍/👎 ratings in the Streamlit UI, (2) uses an LLM-as-judge to categorize failures, and (3) uses accumulated feedback to permanently improve retrieval parameters, synthesize prompts, and surface KB content gaps — without adding heavy dependencies.

---

## Architecture

### 1. Feedback Storage — SQLite

**New file:** `src/agentic_rag/feedback/store.py`

Single table in `feedback.db` at the project root:

```sql
CREATE TABLE IF NOT EXISTS feedback (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT    NOT NULL,
    query     TEXT    NOT NULL,
    answer    TEXT    NOT NULL,
    sources   TEXT    NOT NULL,  -- JSON: [{title, content, score}]
    top_score REAL,              -- top RRF score, used for param tuning
    rating    INTEGER NOT NULL,  -- 1 = thumbs up, -1 = thumbs down
    note      TEXT,              -- optional user note (thumbs down only)
    category  TEXT               -- retrieval_miss | synthesis_failure | missing_content
)
```

API: `save(entry: FeedbackEntry) -> None` and `get_all() -> list[FeedbackEntry]`.

Supersedes the existing `evals/results.jsonl` system.

---

### 2. Streamlit UI — `app.py`

Minimal additions using `st.session_state`:

- After a response renders, show 👍 / 👎 buttons
- On 👎: show optional text input for a note, then "Submit"
- On submission: write to SQLite, run LLM-as-judge synchronously with a spinner (single call, ~2-3s)
- Buttons disappear after rating (session state flag)
- **Sidebar:**
  - "KB Gaps" list — queries categorized as `missing_content`
  - "Optimize" button — disabled until 10+ ratings; runs optimizer on click

---

### 3. LLM-as-Judge — `src/agentic_rag/feedback/judge.py`

Triggered on every 👎 submission. Single Ollama call with this prompt:

```
You are evaluating a RAG system response. Classify why it failed.

Query: {query}
Answer: {answer}
Sources retrieved:
  1. "{title}" — "{content[:350]}..."
  2. ...

Pick exactly one failure category:
- retrieval_miss: sources retrieved are irrelevant to the query
- synthesis_failure: sources are relevant but the answer is wrong or incomplete
- missing_content: the topic doesn't exist in the knowledge base

Respond with JSON only: {"category": "..."}
```

Result is written back to the `category` column for that feedback row.

Note: scores are **not** passed to the judge (BM25 scores are unnormalized; RRF scores are tiny decimals not interpretable without context). Source snippets (first 150 chars) provide sufficient signal.

---

### 4. Optimizer — `src/agentic_rag/feedback/optimizer.py`

Three independent functions, all called from the "Optimize" button:

#### A. Retrieval param tuning
- Reads all rated feedback with stored `top_score`
- Computes the median `top_score` for 👍 vs 👎 responses
- If the 👎 median is above `min_similarity`, suggests raising the threshold
- If the 👍 median is below `min_similarity`, suggests lowering it
- Writes the suggested value back to `config/default.yaml`

#### B. Prompt improvement (few-shot injection)
- Takes the 3 most recent 👍 `(query, answer)` pairs
- Injects them as few-shot examples into the `synthesize` node prompt
- Stored as a `few_shot_examples` key in `config/default.yaml`
- The synthesize node reads this key at startup and prepends examples to the prompt

#### C. KB gap report
- Queries all rows where `category = 'missing_content'`
- Returns them as a list for display in the sidebar
- No automation — user decides what to add to Notion

---

## File Changes Summary

| File | Change |
|------|--------|
| `src/agentic_rag/feedback/store.py` | **New** — SQLite CRUD |
| `src/agentic_rag/feedback/judge.py` | **New** — LLM-as-judge |
| `src/agentic_rag/feedback/optimizer.py` | **New** — 3 optimization functions |
| `src/agentic_rag/feedback/__init__.py` | **New** — package init |
| `app.py` | **Modified** — add feedback UI + sidebar |
| `src/agentic_rag/pipeline/rag_pipeline.py` | **Modified** — read few-shot examples from config |
| `config/default.yaml` | **Modified** — add `few_shot_examples: []` key |

---

## Verification

1. Run `uv run streamlit run app.py`
2. Submit a query → confirm 👍/👎 buttons appear after response
3. Click 👎 → add a note → submit → confirm row appears in `feedback.db`
4. Check sidebar for KB gaps after a `missing_content` categorization
5. After 10+ ratings, click "Optimize" → confirm `default.yaml` is updated with new `min_similarity` and `few_shot_examples`
6. Submit another query → verify synthesize prompt includes few-shot examples (visible in logs)
