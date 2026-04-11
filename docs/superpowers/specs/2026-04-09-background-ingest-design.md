# Background Ingest on Startup — Design Spec

**Date:** 2026-04-09

## Context

The RAG system queries a Notion knowledge base indexed in ChromaDB + BM25. Currently the index is only updated by running `scripts/ingest.py` manually from the terminal. This means the Streamlit app can serve stale answers if Notion content has changed since the last manual run.

The goal is to make the app self-refreshing: when the user opens it, the knowledge base silently updates in the background so queries always reflect recent Notion content. Ingest takes 1-3 minutes, so it must not block the UI.

## Approach

Launch `NotionIngester.ingest()` in a background `threading.Thread` once per Streamlit session on app startup. The app is immediately usable while ingest runs. A module-level shared dict (`_sync_state`) bridges state between the background thread and Streamlit's main thread (which can't safely receive writes to `st.session_state` from other threads).

## Components

### `_sync_state` (module-level dict)

```python
_sync_state: dict = {"status": "idle", "chunks": 0, "error": ""}
```

- `status`: one of `"idle"` | `"syncing"` | `"done"` | `"error"`
- `chunks`: number of chunks indexed in the last run
- `error`: error message string if status is `"error"`

Written by the background thread; read by Streamlit's main thread on each rerun.

### Background thread

Triggered once per session via `st.session_state["sync_started"]` flag. On first render:

1. Set `_sync_state["status"] = "syncing"`
2. Spawn `threading.Thread(target=_run_ingest, daemon=True)`
3. Set `st.session_state["sync_started"] = True`

`_run_ingest()` calls `asyncio.run(ingester.ingest())`, writes result to `_sync_state`, then sets status to `"done"` or `"error"`.

The `NotionIngester` is constructed independently using `load_config()` and `OllamaLLM(config.llm)` — mirroring exactly what `scripts/ingest.py` does. It does not depend on `AgenticRAGSystem`.

### Sidebar UI

A new "Knowledge Base" section is added above the existing "Feedback" section:

| State | Display |
|-------|---------|
| `syncing` | `⟳ Syncing knowledge base...` |
| `done` | `✓ Synced · {N} chunks indexed` |
| `error` | `✗ Sync failed: {message}` |
| `idle` | *(nothing shown)* |

## Data Flow

```
App starts
    └─ first render: sync_started not set
           └─ set _sync_state["status"] = "syncing"
           └─ spawn daemon thread
           └─ set session_state["sync_started"] = True
    └─ sidebar renders "⟳ Syncing..."

Background thread
    └─ asyncio.run(ingester.ingest(full=False))
    └─ _sync_state["status"] = "done", chunks = N
                   OR
    └─ _sync_state["status"] = "error", error = str(e)

Next user interaction (rerun)
    └─ sidebar reads _sync_state → shows "✓ Synced · N chunks"
```

## Files Modified

- `app.py` — only file changed

## Verification

1. Run `streamlit run app.py`
2. Sidebar should immediately show "⟳ Syncing knowledge base..."
3. While syncing, submit a query — app should respond normally
4. After 1-3 minutes, next interaction should show "✓ Synced · N chunks indexed"
5. Update a Notion page, restart the app, verify the updated content appears in query results
6. To test error path: temporarily set an invalid `NOTION_API_KEY` env var and restart
