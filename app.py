# app.py
import asyncio
import threading
from typing import TypedDict

import streamlit as st
from main import AgenticRAGSystem
from agentic_rag.config import load_config
from agentic_rag.feedback.store import FeedbackEntry, get_all, save, update_category
from agentic_rag.feedback.judge import classify_failure
from agentic_rag.feedback.optimizer import apply_optimization
from agentic_rag.ingestion.notion import NotionIngester
from agentic_rag.llm.ollama import OllamaLLM


class _SyncState(TypedDict):
    status: str
    chunks: int
    error: str


_sync_state: _SyncState = {"status": "idle", "chunks": 0, "error": ""}


def _run_ingest() -> None:
    """Run NotionIngester in background thread; writes result to _sync_state."""
    _sync_state.update({"chunks": 0, "error": ""})
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


@st.cache_resource
def get_system() -> AgenticRAGSystem:
    return AgenticRAGSystem()


# ── Background ingest (once per session) ───────────────────────────────────────
if not st.session_state.get("sync_started"):
    _sync_state["status"] = "syncing"
    threading.Thread(target=_run_ingest, daemon=True).start()
    st.session_state["sync_started"] = True

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Knowledge Base")
    _status = _sync_state["status"]
    if _status == "syncing":
        st.caption("⟳ Syncing knowledge base...")
    elif _status == "done":
        st.caption(f"✓ Synced · {_sync_state['chunks']} chunks indexed")
    elif _status == "error":
        st.caption(f"✗ Sync failed: {_sync_state['error']}")
    else:
        st.caption("—")

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
                    {
                        "title": s["title"],
                        "content": s.get("content", ""),
                        "score": s.get("score", 0.0),
                    }
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
            st.rerun()

        if st.session_state.get("show_note"):
            note = st.text_input("What was wrong? (optional)")
            if st.button("Submit feedback"):
                sources_for_store = [
                    {
                        "title": s["title"],
                        "content": s.get("content", ""),
                        "score": s.get("score", 0.0),
                    }
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
