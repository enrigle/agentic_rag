# Background Ingest on Startup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the Streamlit app starts, automatically run `NotionIngester.ingest()` in a background thread so the knowledge base is silently updated while the user can query immediately.

**Architecture:** A module-level `_sync_state` dict tracks ingest status across the background thread and Streamlit's main thread. A daemon thread is spawned once per session (guarded by `st.session_state["sync_started"]`). The sidebar renders the current sync state on every Streamlit rerun.

**Tech Stack:** `threading.Thread`, `NotionIngester` (`src/agentic_rag/ingestion/notion.py`), `load_config` + `OllamaLLM` (same pattern as `scripts/ingest.py`), `pytest`, `unittest.mock.AsyncMock`

---

## File Map

- Modify: `app.py` — add imports, `_sync_state`, `_run_ingest()`, startup trigger, sidebar status UI
- Create: `tests/unit/test_app_ingest.py` — unit tests for `_run_ingest()`

---

### Task 1: Add `_sync_state` and `_run_ingest()` to `app.py`

**Files:**
- Modify: `app.py`
- Test: `tests/unit/test_app_ingest.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_app_ingest.py`:

```python
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import app


def test_run_ingest_success() -> None:
    mock_ingester = MagicMock()
    mock_ingester.ingest = AsyncMock(return_value=42)

    with (
        patch("app.load_config", return_value=MagicMock()),
        patch("app.OllamaLLM", return_value=MagicMock()),
        patch("app.NotionIngester", return_value=mock_ingester),
    ):
        app._sync_state["status"] = "idle"
        app._sync_state["chunks"] = 0
        app._sync_state["error"] = ""
        app._run_ingest()

    assert app._sync_state["status"] == "done"
    assert app._sync_state["chunks"] == 42
    assert app._sync_state["error"] == ""


def test_run_ingest_error() -> None:
    with (
        patch("app.load_config", side_effect=RuntimeError("bad config")),
    ):
        app._sync_state["status"] = "idle"
        app._sync_state["error"] = ""
        app._run_ingest()

    assert app._sync_state["status"] == "error"
    assert "bad config" in app._sync_state["error"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_app_ingest.py -v
```

Expected: `AttributeError: module 'app' has no attribute '_sync_state'`

- [ ] **Step 3: Add imports and `_sync_state` + `_run_ingest` to `app.py`**

Add to the top of `app.py` (after existing imports):

```python
import asyncio
import threading

import streamlit as st
from main import AgenticRAGSystem
from agentic_rag.config import load_config
from agentic_rag.feedback.store import FeedbackEntry, get_all, save, update_category
from agentic_rag.feedback.judge import classify_failure
from agentic_rag.feedback.optimizer import apply_optimization
from agentic_rag.ingestion.notion import NotionIngester
from agentic_rag.llm.ollama import OllamaLLM
```

Note: `asyncio` is already imported — keep the existing import, don't duplicate it.

Then add immediately after the imports (before `get_system`):

```python
_sync_state: dict[str, object] = {"status": "idle", "chunks": 0, "error": ""}


def _run_ingest() -> None:
    """Run NotionIngester in background thread; writes result to _sync_state."""
    try:
        config = load_config()
        llm = OllamaLLM(config.llm)
        ingester = NotionIngester(config, llm)
        total = asyncio.run(ingester.ingest(full=False))
        _sync_state["chunks"] = total
        _sync_state["status"] = "done"
    except Exception as exc:  # noqa: BLE001
        _sync_state["error"] = str(exc)
        _sync_state["status"] = "error"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_app_ingest.py -v
```

Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add app.py tests/unit/test_app_ingest.py
git commit -m "feat: add _sync_state and _run_ingest for background ingest"
```

---

### Task 2: Trigger ingest once per session on startup

**Files:**
- Modify: `app.py` — add session guard that spawns the thread

- [ ] **Step 1: Add startup trigger in `app.py`**

Add this block immediately after `get_system()` definition (before the `# ── Sidebar` comment):

```python
# ── Background ingest (once per session) ───────────────────────────────────────
if not st.session_state.get("sync_started"):
    _sync_state["status"] = "syncing"
    threading.Thread(target=_run_ingest, daemon=True).start()
    st.session_state["sync_started"] = True
```

- [ ] **Step 2: Verify app still starts**

```bash
streamlit run app.py
```

Expected: app loads immediately, no errors in terminal. Sidebar not yet updated (Task 3).

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: spawn background ingest thread on app startup"
```

---

### Task 3: Show sync status in sidebar

**Files:**
- Modify: `app.py` — add Knowledge Base section to sidebar

- [ ] **Step 1: Add Knowledge Base section to sidebar**

Inside the `with st.sidebar:` block in `app.py`, add this **before** `st.header("Feedback")`:

```python
    st.header("Knowledge Base")
    _status = _sync_state["status"]
    if _status == "syncing":
        st.caption("⟳ Syncing knowledge base...")
    elif _status == "done":
        st.caption(f"✓ Synced · {_sync_state['chunks']} chunks indexed")
    elif _status == "error":
        st.caption(f"✗ Sync failed: {_sync_state['error']}")
```

- [ ] **Step 2: Manual end-to-end verification**

```bash
streamlit run app.py
```

Check:
1. Sidebar shows "⟳ Syncing knowledge base..." immediately after launch
2. App is usable — submit a query while sync is running, expect a normal answer
3. After 1-3 minutes, submit another query (triggers a rerun) — sidebar should show "✓ Synced · N chunks indexed"

- [ ] **Step 3: Verify error path**

Temporarily break the Notion token:

```bash
NOTION_API_KEY=invalid streamlit run app.py
```

Expected: sidebar shows "✗ Sync failed: ..." within a few seconds of the first error.

Restore env and re-run normally to confirm it returns to normal.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: show background ingest status in sidebar"
```
