# app.py
import logging
import os
import asyncio

# transformers resets its own logger level to WARNING when first imported,
# overriding any setLevel() below; it honors this env var at that reset. Must be
# set before transformers is imported (via sentence_transformers downstream).
# docker-compose sets it too; this covers local runs. setdefault → env wins.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agentic_rag.config import RAGConfig, load_config
from agentic_rag.health import ServiceStatus, run_checks
from agentic_rag.feedback.store import get_all, save, update_category
from agentic_rag.feedback.judge import classify_failure
from agentic_rag.feedback.optimizer import apply_optimization
from agentic_rag.pipeline.coordinator import PipelineCoordinator
from agentic_rag.llm.base import ChatModel
from agentic_rag.pipeline.rag_pipeline import create_pipeline, make_synth_llm
from agentic_rag.ui.answers import (
    feedback_entry,
    score_feedback,
    source_markdown,
    to_result_dict,
)
from agentic_rag.ui.conversations import ConversationStore
from agentic_rag.ui.ingest import BackgroundIngest


logger = logging.getLogger(__name__)


# Process-wide singletons: st.cache_resource shares one instance across sessions.
@st.cache_resource
def get_config() -> RAGConfig:
    return load_config()


@st.cache_resource
def get_ingest() -> BackgroundIngest:
    return BackgroundIngest()


@st.cache_resource
def get_conversations() -> ConversationStore:
    return ConversationStore(max_threads=100)


@st.cache_resource
def _service_statuses() -> list[ServiceStatus]:
    cfg = load_config()
    return asyncio.run(
        run_checks(
            ollama_url=cfg.llm.base_url,
            redis_url=cfg.redis.url,
            chroma_path=cfg.chroma_path,
            embed_backend=cfg.embed_backend,
            synth_uses_ollama=not (
                cfg.groq.is_configured() or cfg.azure_openai.is_configured()
            ),
        )
    )


@st.cache_resource(show_spinner=False)
def get_pipeline() -> PipelineCoordinator:
    return create_pipeline()


@st.cache_resource(show_spinner=False)
def get_judge_llm() -> ChatModel:
    return make_synth_llm(load_config())


_cfg = get_config()
_ingest = get_ingest()
_conversations = get_conversations()


# ── Background ingest (once per session) ───────────────────────────────────────
# `streamlit run` executes this file as __main__. A plain import (tests) must not
# start a real Notion ingest that rewrites data/chroma_db.
if __name__ == "__main__" and not st.session_state.get("sync_started"):
    _ingest.start()
    st.session_state["sync_started"] = True

# ── Session defaults ───────────────────────────────────────────────────────────
# thread_id is stored in the URL so it survives WebSocket reconnections
if "thread_id" not in st.session_state:
    _tid = st.query_params.get("thread_id") or str(uuid.uuid4())
    st.session_state.thread_id = _tid
    st.query_params["thread_id"] = _tid
if "messages" not in st.session_state:
    # Restore from process-level store on reconnect; otherwise start fresh
    st.session_state.messages = _conversations.get(st.session_state.thread_id)
if "rated" not in st.session_state:
    st.session_state.rated = False
if "show_note" not in st.session_state:
    st.session_state.show_note = False

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Services")
    for svc in _service_statuses():
        icon = "🟢" if svc.ok else "🔴"
        label = f"{icon} {svc.name}"
        if svc.detail:
            st.caption(f"{label} — {svc.detail}")
        else:
            st.caption(label)

    st.header("Conversation")
    if st.button("New conversation"):
        new_tid = str(uuid.uuid4())
        st.session_state.thread_id = new_tid
        st.session_state.messages = []
        _conversations.put(new_tid, [])
        st.query_params["thread_id"] = new_tid
        st.session_state.rated = False
        st.session_state.show_note = False
        st.rerun()

    st.header("Knowledge Base")
    _sync = _ingest.state
    if _sync.status == "syncing":
        st.caption("⟳ Syncing knowledge base...")
    elif _sync.status == "done":
        st.caption(f"✓ Synced · {_sync.chunks} chunks indexed")
    elif _sync.status == "error":
        st.caption(f"✗ Sync failed ({_sync.error})")
    else:
        st.caption("—")

    st.header("Feedback")

    all_entries = get_all(_cfg.feedback_db_path)
    rated_count = len(all_entries)
    st.caption(f"{rated_count} ratings collected")

    optimize_disabled = rated_count < 10
    if st.button("Optimize", disabled=optimize_disabled, help="Requires 10+ ratings"):
        with st.spinner("Optimizing..."):
            opt_result = apply_optimization(
                all_entries,
                feedback_config_path=_cfg.feedback_config_path,
                few_shot_max=_cfg.retriever.few_shot_max,
            )
        parts = []
        if opt_result.new_min_similarity is not None:
            # The cached pipeline holds the old threshold; rebuild it on next query.
            get_pipeline.clear()
            parts.append(f"min_similarity → {opt_result.new_min_similarity}")
        if opt_result.few_shot_count:
            parts.append(f"{opt_result.few_shot_count} few-shot examples saved")
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
                        st.markdown(source_markdown(s["index"], s["title"], s["url"]))
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
        save(feedback_entry(last_query, last_result, rating=1), _cfg.feedback_db_path)
        score_feedback(last_result, 1)
        st.session_state.rated = True
        st.rerun()

    if col2.button("👎 Bad answer"):
        st.session_state.show_note = True
        st.rerun()

    if st.session_state.show_note:
        note = st.text_input("What was wrong? (optional)")
        if st.button("Submit feedback"):
            entry = feedback_entry(last_query, last_result, rating=-1, note=note)
            entry_id = save(entry, _cfg.feedback_db_path)
            score_feedback(last_result, -1, note)
            with st.spinner("Analyzing failure..."):
                category = asyncio.run(
                    classify_failure(
                        get_judge_llm(),
                        query=last_query,
                        answer=last_result["answer"],
                        sources=entry.sources,
                    )
                )
            update_category(entry_id, category, _cfg.feedback_db_path)
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
        result = to_result_dict(
            asyncio.run(
                get_pipeline().query(query, thread_id=st.session_state.thread_id)
            )
        )
    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "result": result}
    )
    _conversations.put(st.session_state.thread_id, st.session_state.messages)
    st.session_state.rated = False
    st.session_state.show_note = False
    st.rerun()
