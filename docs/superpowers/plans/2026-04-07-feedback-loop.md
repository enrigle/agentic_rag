# Feedback Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 👍/👎 feedback UI to the Streamlit app that stores ratings in SQLite, classifies failures with an LLM-as-judge, and uses accumulated data to improve retrieval params, synthesize prompts (via few-shot examples), and surface KB content gaps.

**Architecture:** A new `src/agentic_rag/feedback/` package handles storage (`store.py`), LLM-as-judge (`judge.py`), and optimization (`optimizer.py`). `main.py`'s `query()` return dict is extended with `content` + `top_score`. `app.py` gains rating buttons and a sidebar. `synthesize()` in `main.py` reads few-shot examples from `feedback_config.json` at call time.

**Tech Stack:** Python stdlib `sqlite3`, `ollama` (already a dep), `statistics`, `yaml` (already a dep), Streamlit session state.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/agentic_rag/feedback/__init__.py` | Create | Package marker |
| `src/agentic_rag/feedback/store.py` | Create | SQLite CRUD for `FeedbackEntry` |
| `src/agentic_rag/feedback/judge.py` | Create | LLM-as-judge via Ollama |
| `src/agentic_rag/feedback/optimizer.py` | Create | Retrieval tuning, few-shot selection, KB gap reporting |
| `tests/unit/test_feedback_store.py` | Create | Unit tests for store |
| `tests/unit/test_judge.py` | Create | Unit tests for judge (mocked Ollama) |
| `tests/unit/test_optimizer.py` | Create | Unit tests for optimizer |
| `main.py` | Modify | Add `content`, `score`, `top_score` to `query()` return; add few-shot injection to `synthesize()` |
| `app.py` | Modify | Add 👍/👎 buttons, note input, sidebar |
| `feedback_config.json` | Created at runtime | Few-shot examples written by optimizer |

---

## Task 1: Feedback Store

**Files:**
- Create: `src/agentic_rag/feedback/__init__.py`
- Create: `src/agentic_rag/feedback/store.py`
- Create: `tests/unit/test_feedback_store.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_feedback_store.py
import json
import pytest
from pathlib import Path
from agentic_rag.feedback.store import FeedbackEntry, save, get_all, update_category


def test_save_and_get_all(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    entry = FeedbackEntry(
        query="What is ML?",
        answer="Machine learning is...",
        sources=[{"title": "ML Basics", "content": "ML intro.", "score": 0.05}],
        top_score=0.05,
        rating=1,
    )
    entry_id = save(entry, db_path=db)
    assert entry_id == 1
    all_entries = get_all(db_path=db)
    assert len(all_entries) == 1
    assert all_entries[0].query == "What is ML?"
    assert all_entries[0].rating == 1
    assert all_entries[0].sources == [{"title": "ML Basics", "content": "ML intro.", "score": 0.05}]


def test_update_category(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    entry = FeedbackEntry(query="q", answer="a", sources=[], top_score=0.0, rating=-1)
    entry_id = save(entry, db_path=db)
    update_category(entry_id, "retrieval_miss", db_path=db)
    entries = get_all(db_path=db)
    assert entries[0].category == "retrieval_miss"


def test_save_empty_sources(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    entry = FeedbackEntry(query="q", answer="a", sources=[], top_score=0.0, rating=1)
    save(entry, db_path=db)
    entries = get_all(db_path=db)
    assert entries[0].sources == []


def test_multiple_entries_oldest_first(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    for i in range(3):
        save(FeedbackEntry(query=f"q{i}", answer="a", sources=[], top_score=0.0, rating=1), db_path=db)
    entries = get_all(db_path=db)
    assert [e.query for e in entries] == ["q0", "q1", "q2"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_feedback_store.py -v
```

Expected: `ModuleNotFoundError: No module named 'agentic_rag.feedback'`

- [ ] **Step 3: Create the package and store**

```python
# src/agentic_rag/feedback/__init__.py
```
(empty file)

```python
# src/agentic_rag/feedback/store.py
"""SQLite-backed feedback store."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("feedback.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS feedback (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT    NOT NULL,
    query     TEXT    NOT NULL,
    answer    TEXT    NOT NULL,
    sources   TEXT    NOT NULL,
    top_score REAL    NOT NULL,
    rating    INTEGER NOT NULL,
    note      TEXT    NOT NULL DEFAULT '',
    category  TEXT    NOT NULL DEFAULT ''
)
"""


@dataclass
class FeedbackEntry:
    query: str
    answer: str
    sources: list[dict]        # [{title, content, score}]
    top_score: float
    rating: int                # 1 = thumbs up, -1 = thumbs down
    note: str = ""
    category: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    id: int | None = None


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(_CREATE_TABLE)
    conn.commit()


def save(entry: FeedbackEntry, db_path: Path = DB_PATH) -> int:
    """Persist a FeedbackEntry. Returns the new row id."""
    with sqlite3.connect(db_path) as conn:
        _ensure_table(conn)
        cursor = conn.execute(
            "INSERT INTO feedback (timestamp, query, answer, sources, top_score, rating, note, category) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry.timestamp,
                entry.query,
                entry.answer,
                json.dumps(entry.sources),
                entry.top_score,
                entry.rating,
                entry.note,
                entry.category,
            ),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]


def get_all(db_path: Path = DB_PATH) -> list[FeedbackEntry]:
    """Return all entries ordered oldest first."""
    with sqlite3.connect(db_path) as conn:
        _ensure_table(conn)
        rows = conn.execute(
            "SELECT id, timestamp, query, answer, sources, top_score, rating, note, category "
            "FROM feedback ORDER BY id ASC"
        ).fetchall()
    return [
        FeedbackEntry(
            id=row[0],
            timestamp=row[1],
            query=row[2],
            answer=row[3],
            sources=json.loads(row[4]),
            top_score=row[5],
            rating=row[6],
            note=row[7],
            category=row[8],
        )
        for row in rows
    ]


def update_category(entry_id: int, category: str, db_path: Path = DB_PATH) -> None:
    """Write the judge's classification back to the row."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE feedback SET category = ? WHERE id = ?",
            (category, entry_id),
        )
        conn.commit()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_feedback_store.py -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agentic_rag/feedback/__init__.py src/agentic_rag/feedback/store.py tests/unit/test_feedback_store.py
git commit -m "feat: add SQLite feedback store"
```

---

## Task 2: LLM-as-Judge

**Files:**
- Create: `src/agentic_rag/feedback/judge.py`
- Create: `tests/unit/test_judge.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_judge.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agentic_rag.feedback.judge import classify_failure


@pytest.mark.asyncio
async def test_classify_retrieval_miss() -> None:
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_ctx.chat = AsyncMock(
        return_value={"message": {"content": '{"category": "retrieval_miss"}'}}
    )
    with patch("agentic_rag.feedback.judge.ollama.AsyncClient", return_value=mock_ctx):
        result = await classify_failure(
            query="What is the capital of France?",
            answer="I don't know.",
            sources=[{"title": "Python docs", "content": "Python is a language.", "score": 0.01}],
        )
    assert result == "retrieval_miss"


@pytest.mark.asyncio
async def test_classify_synthesis_failure() -> None:
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_ctx.chat = AsyncMock(
        return_value={"message": {"content": '{"category": "synthesis_failure"}'}}
    )
    with patch("agentic_rag.feedback.judge.ollama.AsyncClient", return_value=mock_ctx):
        result = await classify_failure(
            query="How does RRF work?",
            answer="RRF stands for...",
            sources=[{"title": "RRF paper", "content": "Reciprocal Rank Fusion...", "score": 0.05}],
        )
    assert result == "synthesis_failure"


@pytest.mark.asyncio
async def test_classify_invalid_json_returns_unknown() -> None:
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_ctx.chat = AsyncMock(return_value={"message": {"content": "not json at all"}})
    with patch("agentic_rag.feedback.judge.ollama.AsyncClient", return_value=mock_ctx):
        result = await classify_failure(query="q", answer="a", sources=[])
    assert result == "unknown"


@pytest.mark.asyncio
async def test_classify_exception_returns_unknown() -> None:
    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)
    mock_ctx.chat = AsyncMock(side_effect=RuntimeError("connection refused"))
    with patch("agentic_rag.feedback.judge.ollama.AsyncClient", return_value=mock_ctx):
        result = await classify_failure(query="q", answer="a", sources=[])
    assert result == "unknown"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_judge.py -v
```

Expected: `ModuleNotFoundError: No module named 'agentic_rag.feedback.judge'`

- [ ] **Step 3: Implement the judge**

```python
# src/agentic_rag/feedback/judge.py
"""LLM-as-judge: classifies why a RAG response failed."""

from __future__ import annotations

import json
import logging

import ollama

logger = logging.getLogger(__name__)

_VALID_CATEGORIES = {"retrieval_miss", "synthesis_failure", "missing_content"}

_PROMPT_TEMPLATE = """\
You are evaluating a RAG system response. Classify why it failed.

Query: {query}
Answer: {answer}
Sources retrieved:
{sources_block}

Pick exactly one failure category:
- retrieval_miss: sources retrieved are irrelevant to the query
- synthesis_failure: sources are relevant but the answer is wrong or incomplete
- missing_content: the topic does not exist in the knowledge base

Respond with JSON only, no other text: {{"category": "..."}}"""


async def classify_failure(
    query: str,
    answer: str,
    sources: list[dict],
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
) -> str:
    """Classify a thumbs-down response. Returns one of the three category strings or 'unknown'."""
    sources_block = "\n".join(
        f'  {i + 1}. "{s.get("title", "")}" — "{str(s.get("content", ""))[:350]}..."'
        for i, s in enumerate(sources)
    ) or "  (no sources)"

    prompt = _PROMPT_TEMPLATE.format(
        query=query,
        answer=answer,
        sources_block=sources_block,
    )

    try:
        async with ollama.AsyncClient(host=base_url) as client:
            response = await client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
        text: str = response["message"]["content"]
        start = text.find("{")
        end = text.rfind("}") + 1
        parsed: dict = json.loads(text[start:end])
        category: str = parsed.get("category", "unknown")
        if category not in _VALID_CATEGORIES:
            logger.warning("judge: unexpected category %r; defaulting to unknown", category)
            return "unknown"
        return category
    except (json.JSONDecodeError, KeyError, ValueError):
        logger.warning("judge: failed to parse LLM response")
        return "unknown"
    except Exception as exc:
        logger.warning("judge: unexpected error: %s", exc)
        return "unknown"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_judge.py -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agentic_rag/feedback/judge.py tests/unit/test_judge.py
git commit -m "feat: add LLM-as-judge for failure classification"
```

---

## Task 3: Optimizer

**Files:**
- Create: `src/agentic_rag/feedback/optimizer.py`
- Create: `tests/unit/test_optimizer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_optimizer.py
import json
import pytest
from pathlib import Path
from agentic_rag.feedback.store import FeedbackEntry
from agentic_rag.feedback.optimizer import (
    OptimizationResult,
    apply_optimization,
    get_few_shot_examples,
    get_kb_gaps,
    tune_retrieval_params,
)


def _entry(rating: int, top_score: float, category: str = "", query: str = "q", answer: str = "a") -> FeedbackEntry:
    return FeedbackEntry(query=query, answer=answer, sources=[], top_score=top_score, rating=rating, category=category)


def test_tune_suggests_midpoint() -> None:
    entries = [
        _entry(1, 0.8), _entry(1, 0.7), _entry(1, 0.75),
        _entry(-1, 0.1), _entry(-1, 0.2), _entry(-1, 0.15),
    ]
    result = tune_retrieval_params(entries)
    assert result is not None
    assert 0.1 < result < 0.8


def test_tune_no_signal_when_scores_equal() -> None:
    entries = [_entry(1, 0.5), _entry(-1, 0.5)]
    assert tune_retrieval_params(entries) is None


def test_tune_returns_none_without_both_polarities() -> None:
    assert tune_retrieval_params([_entry(1, 0.8)]) is None
    assert tune_retrieval_params([_entry(-1, 0.1)]) is None
    assert tune_retrieval_params([]) is None


def test_get_few_shot_examples_returns_3_most_recent() -> None:
    entries = [_entry(1, 0.8, query=f"q{i}", answer=f"a{i}") for i in range(5)]
    examples = get_few_shot_examples(entries)
    assert len(examples) == 3
    assert examples[0]["query"] == "q4"
    assert examples[1]["query"] == "q3"
    assert examples[2]["query"] == "q2"


def test_get_few_shot_examples_ignores_thumbs_down() -> None:
    entries = [_entry(-1, 0.1, query="bad")]
    assert get_few_shot_examples(entries) == []


def test_get_kb_gaps_returns_missing_content_queries() -> None:
    entries = [
        _entry(-1, 0.1, category="missing_content", query="What is X?"),
        _entry(-1, 0.1, category="retrieval_miss", query="How to do Y?"),
        _entry(1, 0.8, query="Good Q"),
    ]
    assert get_kb_gaps(entries) == ["What is X?"]


def test_apply_optimization_writes_feedback_config(tmp_path: Path) -> None:
    config_path = tmp_path / "default.yaml"
    config_path.write_text("retriever:\n  min_similarity: 0.35\n")
    fb_config_path = tmp_path / "feedback_config.json"
    entries = [
        _entry(1, 0.8, query="good q", answer="good a"),
        _entry(-1, 0.1, category="missing_content", query="missing q"),
    ]
    result = apply_optimization(entries, config_path=config_path, feedback_config_path=fb_config_path)
    assert isinstance(result, OptimizationResult)
    assert fb_config_path.exists()
    fc = json.loads(fb_config_path.read_text())
    assert "few_shot_examples" in fc
    assert result.kb_gaps == ["missing q"]


def test_apply_optimization_updates_yaml_when_signal(tmp_path: Path) -> None:
    config_path = tmp_path / "default.yaml"
    config_path.write_text("retriever:\n  min_similarity: 0.35\n")
    fb_config_path = tmp_path / "feedback_config.json"
    entries = [
        _entry(1, 0.8), _entry(1, 0.75),
        _entry(-1, 0.1), _entry(-1, 0.15),
    ]
    result = apply_optimization(entries, config_path=config_path, feedback_config_path=fb_config_path)
    if result.new_min_similarity is not None:
        import yaml
        raw = yaml.safe_load(config_path.read_text())
        assert raw["retriever"]["min_similarity"] == result.new_min_similarity
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_optimizer.py -v
```

Expected: `ModuleNotFoundError: No module named 'agentic_rag.feedback.optimizer'`

- [ ] **Step 3: Implement the optimizer**

```python
# src/agentic_rag/feedback/optimizer.py
"""Optimization functions: retrieval param tuning, few-shot selection, KB gap reporting."""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from agentic_rag.feedback.store import FeedbackEntry

logger = logging.getLogger(__name__)

_MIN_SCORE_SEPARATION = 0.005  # ignore signal if medians are this close


@dataclass
class OptimizationResult:
    new_min_similarity: float | None  # None = no change suggested
    few_shot_count: int
    kb_gaps: list[str]


def tune_retrieval_params(entries: list[FeedbackEntry]) -> float | None:
    """Suggest a new min_similarity based on score distributions.

    Returns the midpoint between median thumbs-up and thumbs-down top_score,
    or None if there is insufficient signal.
    """
    up = [e.top_score for e in entries if e.rating == 1 and e.top_score > 0]
    down = [e.top_score for e in entries if e.rating == -1 and e.top_score > 0]
    if not up or not down:
        return None
    median_up = statistics.median(up)
    median_down = statistics.median(down)
    if abs(median_up - median_down) < _MIN_SCORE_SEPARATION:
        return None
    suggested = (median_up + median_down) / 2
    return round(suggested, 4)


def get_few_shot_examples(entries: list[FeedbackEntry]) -> list[dict[str, str]]:
    """Return the 3 most recent thumbs-up (query, answer) pairs, newest first."""
    thumbs_up = [e for e in reversed(entries) if e.rating == 1]
    return [{"query": e.query, "answer": e.answer} for e in thumbs_up[:3]]


def get_kb_gaps(entries: list[FeedbackEntry]) -> list[str]:
    """Return queries where the LLM judge found missing content."""
    return [e.query for e in entries if e.category == "missing_content"]


def apply_optimization(
    entries: list[FeedbackEntry],
    config_path: Path = Path("config/default.yaml"),
    feedback_config_path: Path = Path("feedback_config.json"),
) -> OptimizationResult:
    """Run all three optimizations and persist results to disk."""
    # 1. Retrieval param tuning
    new_min_sim = tune_retrieval_params(entries)
    if new_min_sim is not None:
        try:
            raw: dict = yaml.safe_load(config_path.read_text()) or {}
            raw.setdefault("retriever", {})["min_similarity"] = new_min_sim
            config_path.write_text(yaml.dump(raw, default_flow_style=False))
            logger.info("optimizer: updated min_similarity to %s", new_min_sim)
        except Exception as exc:
            logger.warning("optimizer: failed to update config: %s", exc)
            new_min_sim = None

    # 2. Few-shot examples
    examples = get_few_shot_examples(entries)
    feedback_config_path.write_text(
        json.dumps({"few_shot_examples": examples}, indent=2, ensure_ascii=False)
    )
    logger.info("optimizer: wrote %d few-shot examples", len(examples))

    # 3. KB gaps
    gaps = get_kb_gaps(entries)

    return OptimizationResult(
        new_min_similarity=new_min_sim,
        few_shot_count=len(examples),
        kb_gaps=gaps,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_optimizer.py -v
```

Expected: 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/agentic_rag/feedback/optimizer.py tests/unit/test_optimizer.py
git commit -m "feat: add feedback optimizer (retrieval tuning, few-shot, KB gaps)"
```

---

## Task 4: Extend main.py query() and add few-shot injection

**Files:**
- Modify: `main.py` (two changes: query return shape + synthesize prompt)

The `query()` method currently returns sources without `content` or `score`. The judge needs them. Also, `synthesize()` needs to load few-shot examples from `feedback_config.json`.

- [ ] **Step 1: Modify `query()` to include content, score, and top_score**

Find this block in `main.py` (~line 471):

```python
        rag_results = final_state.get("rag_results") or []
        web_results = final_state.get("web_results") or []
        sources = [
            {"index": i + 1, "title": r["title"], "url": r["source"]}
            for i, r in enumerate(rag_results + web_results)
        ]

        return {
            "answer": final_state.get("final_answer") or "",
            "sources": sources,
            "tool_calls_used": final_state["tool_calls"],
            "latency_ms": round(latency_ms, 2),
        }
```

Replace with:

```python
        rag_results = final_state.get("rag_results") or []
        web_results = final_state.get("web_results") or []
        all_results = rag_results + web_results
        sources = [
            {
                "index": i + 1,
                "title": r["title"],
                "url": r["source"],
                "content": r.get("content", ""),
                "score": r.get("score", 0.0),
            }
            for i, r in enumerate(all_results)
        ]
        top_score = max((r.get("score", 0.0) for r in rag_results), default=0.0)

        return {
            "answer": final_state.get("final_answer") or "",
            "sources": sources,
            "tool_calls_used": final_state["tool_calls"],
            "latency_ms": round(latency_ms, 2),
            "top_score": top_score,
        }
```

- [ ] **Step 2: Add few-shot injection to `synthesize()`**

Find the `prompt = (` assignment in the `synthesize` method (~line 412):

```python
        prompt = (
            "You are a helpful assistant. Answer using ONLY the provided context. "
            "Cite sources inline using [N] notation. Do not fabricate information.\n\n"
            f"Context:\n{context_blocks}\n\nQuestion: {state['query']}"
        )
```

Replace with:

```python
        few_shot_str = ""
        fc_path = Path("feedback_config.json")
        if fc_path.exists():
            try:
                fc = json.loads(fc_path.read_text())
                examples = fc.get("few_shot_examples", [])[:3]
                if examples:
                    few_shot_str = "\n\nExamples of good answers:\n" + "\n\n".join(
                        f"Q: {ex['query']}\nA: {ex['answer']}" for ex in examples
                    )
            except (json.JSONDecodeError, KeyError):
                pass

        prompt = (
            "You are a helpful assistant. Answer using ONLY the provided context. "
            "Cite sources inline using [N] notation. Do not fabricate information."
            f"{few_shot_str}\n\n"
            f"Context:\n{context_blocks}\n\nQuestion: {state['query']}"
        )
```

`Path` is already imported in `main.py`. `json` is already imported.

- [ ] **Step 3: Verify the app still runs**

```bash
uv run python -c "
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

fake_state = {
    'final_answer': 'Test answer.',
    'rag_results': [{'id': '1', 'title': 'T', 'source': 'http://x.com', 'content': 'some content', 'score': 0.05}],
    'web_results': [],
    'tool_calls': 2,
    'error': None,
}

with (
    patch('chromadb.PersistentClient'),
    patch('main.AgenticRAGSystem._load_bm25'),
    patch('main.AgenticRAGSystem._build_graph') as mg,
):
    mg.return_value = MagicMock(ainvoke=AsyncMock(return_value=fake_state))
    from main import AgenticRAGSystem
    sys = AgenticRAGSystem()
    result = asyncio.run(sys.query('test'))

assert 'content' in result['sources'][0], 'missing content'
assert 'score' in result['sources'][0], 'missing score'
assert 'top_score' in result, 'missing top_score'
print('OK:', result['sources'][0]['content'], result['top_score'])
"
```

Expected output: `OK: some content 0.05`

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: extend query() return with content/score/top_score; add few-shot injection to synthesize"
```

---

## Task 5: Streamlit UI

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Replace `app.py` with the new version**

```python
# app.py
import asyncio
import streamlit as st
from main import AgenticRAGSystem
from agentic_rag.feedback.store import FeedbackEntry, get_all, save, update_category
from agentic_rag.feedback.judge import classify_failure
from agentic_rag.feedback.optimizer import apply_optimization


@st.cache_resource
def get_system() -> AgenticRAGSystem:
    return AgenticRAGSystem()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Feedback")

    all_entries = get_all()
    rated_count = len(all_entries)
    st.caption(f"{rated_count} ratings collected")

    optimize_disabled = rated_count < 10
    if st.button("Optimize", disabled=optimize_disabled, help="Requires 10+ ratings"):
        with st.spinner("Optimizing..."):
            result = apply_optimization(all_entries)
        parts = []
        if result.new_min_similarity is not None:
            parts.append(f"min_similarity → {result.new_min_similarity}")
        if result.few_shot_count:
            parts.append(f"{result.few_shot_count} few-shot examples saved")
        st.success(", ".join(parts) if parts else "No changes needed yet")

    gaps = [e.query for e in all_entries if e.category == "missing_content"]
    if gaps:
        st.subheader("KB Gaps")
        st.caption("Consider adding these topics to Notion:")
        for g in gaps:
            st.markdown(f"- {g}")

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("Agentic RAG")

with st.form("search_form"):
    query = st.text_input("Ask a question")
    submitted = st.form_submit_button("Search")

if submitted and query:
    with st.spinner("Thinking..."):
        result = asyncio.run(get_system().query(query))
    st.session_state.result = result
    st.session_state.last_query = query
    st.session_state.rated = False
    st.session_state.show_note = False

if st.session_state.get("result"):
    result = st.session_state.result
    st.markdown(result["answer"])

    if result["sources"]:
        st.divider()
        st.subheader("Sources")
        for s in result["sources"]:
            st.markdown(f"[{s['index']}. {s['title']}]({s['url']})")

    st.caption(
        f"Tool calls: {result['tool_calls_used']} · Latency: {result['latency_ms']:.0f}ms"
    )

    if not st.session_state.get("rated"):
        st.divider()
        col1, col2 = st.columns(2)

        if col1.button("👍 Good answer"):
            entry = FeedbackEntry(
                query=st.session_state.last_query,
                answer=result["answer"],
                sources=[
                    {"title": s["title"], "content": s.get("content", ""), "score": s.get("score", 0.0)}
                    for s in result["sources"]
                ],
                top_score=result.get("top_score", 0.0),
                rating=1,
            )
            save(entry)
            st.session_state.rated = True
            st.rerun()

        if col2.button("👎 Bad answer"):
            st.session_state.show_note = True

        if st.session_state.get("show_note"):
            note = st.text_input("What was wrong? (optional)")
            if st.button("Submit feedback"):
                sources_for_store = [
                    {"title": s["title"], "content": s.get("content", ""), "score": s.get("score", 0.0)}
                    for s in result["sources"]
                ]
                entry = FeedbackEntry(
                    query=st.session_state.last_query,
                    answer=result["answer"],
                    sources=sources_for_store,
                    top_score=result.get("top_score", 0.0),
                    rating=-1,
                    note=note,
                )
                entry_id = save(entry)
                with st.spinner("Analyzing failure..."):
                    category = asyncio.run(
                        classify_failure(
                            query=st.session_state.last_query,
                            answer=result["answer"],
                            sources=sources_for_store,
                        )
                    )
                update_category(entry_id, category)
                st.session_state.rated = True
                st.session_state.show_note = False
                st.rerun()
    else:
        st.caption("Thanks for the feedback!")
```

- [ ] **Step 2: Manual verification**

```bash
uv run streamlit run app.py
```

Run through this checklist:
1. Submit a query → answer appears + 👍/👎 buttons appear below
2. Click 👍 → buttons disappear, "Thanks for the feedback!" appears
3. Submit another query → click 👎 → note field appears → click "Submit feedback" → spinner shows → buttons disappear
4. Check sidebar shows rating count incremented
5. After 10+ ratings, "Optimize" button becomes active → click it → success message shows

- [ ] **Step 3: Run the full test suite to verify nothing broke**

```bash
uv run pytest -x
```

Expected: all existing tests PASS

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add feedback UI to Streamlit app with rating buttons and sidebar"
```

---

## Verification (End-to-End)

After all tasks are complete:

1. `uv run streamlit run app.py` — app starts without errors
2. Submit a query, rate it 👍 — row appears in `feedback.db`
3. Submit another query, rate it 👎 with a note — row appears with `category` filled in
4. Run: `uv run python -c "from agentic_rag.feedback.store import get_all; [print(e) for e in get_all()]"` — entries print correctly
5. After 10+ ratings, click Optimize — `feedback_config.json` created, `config/default.yaml` possibly updated
6. Submit another query — check logs for "few-shot examples" being read (if examples exist)
7. `uv run pytest -x` — all tests pass
