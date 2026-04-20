# app.py
import asyncio
import threading
import uuid
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


@st.cache_resource
def _get_sync_state() -> _SyncState:
    return {"status": "idle", "chunks": 0, "error": ""}


_sync_state = _get_sync_state()


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

# ── Session defaults ───────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rated" not in st.session_state:
    st.session_state.rated = False
if "show_note" not in st.session_state:
    st.session_state.show_note = False

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Conversation")
    if st.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.rated = False
        st.session_state.show_note = False
        st.rerun()

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

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("result"):
            r = msg["result"]
            if r["sources"]:
                with st.expander("Sources"):
                    for s in r["sources"]:
                        st.markdown(f"[{s['index']}. {s['title']}]({s['url']})")
            st.caption(
                f"Tool calls: {r['tool_calls_used']} · Latency: {r['latency_ms']:.0f}ms"
            )

# Feedback for the last answer (shown below chat)
last_result = (
    st.session_state.messages[-1].get("result")
    if st.session_state.messages
    and st.session_state.messages[-1]["role"] == "assistant"
    else None
)
last_query = (
    st.session_state.messages[-2]["content"]
    if len(st.session_state.messages) >= 2
    else ""
)

if last_result and not st.session_state.rated:
    st.divider()
    col1, col2 = st.columns(2)

    if col1.button("👍 Good answer"):
        entry = FeedbackEntry(
            query=last_query,
            answer=last_result["answer"],
            sources=[
                {
                    "title": s["title"],
                    "content": s.get("content", ""),
                    "score": s.get("score", 0.0),
                }
                for s in last_result["sources"]
            ],
            top_score=last_result.get("top_score", 0.0),
            rating=1,
        )
        save(entry)
        st.session_state.rated = True
        st.rerun()

    if col2.button("👎 Bad answer"):
        st.session_state.show_note = True
        st.rerun()

    if st.session_state.show_note:
        note = st.text_input("What was wrong? (optional)")
        if st.button("Submit feedback"):
            sources_for_store = [
                {
                    "title": s["title"],
                    "content": s.get("content", ""),
                    "score": s.get("score", 0.0),
                }
                for s in last_result["sources"]
            ]
            entry = FeedbackEntry(
                query=last_query,
                answer=last_result["answer"],
                sources=sources_for_store,
                top_score=last_result.get("top_score", 0.0),
                rating=-1,
                note=note,
            )
            entry_id = save(entry)
            with st.spinner("Analyzing failure..."):
                category = asyncio.run(
                    classify_failure(
                        query=last_query,
                        answer=last_result["answer"],
                        sources=sources_for_store,
                    )
                )
            update_category(entry_id, category)
            st.session_state.rated = True
            st.session_state.show_note = False
            st.rerun()
elif last_result and st.session_state.rated:
    st.caption("Thanks for the feedback!")

# Chat input (always at bottom)
query = st.chat_input("Ask a question")

if query:
    st.session_state.messages.append({"role": "user", "content": query, "result": None})
    with st.spinner("Thinking..."):
        result = asyncio.run(
            get_system().query(query, thread_id=st.session_state.thread_id)
        )
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "result": result}
    )
    st.session_state.rated = False
    st.session_state.show_note = False
    st.rerun()
