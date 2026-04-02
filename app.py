import asyncio
import streamlit as st
from main import AgenticRAGSystem


@st.cache_resource
def get_system() -> AgenticRAGSystem:
    return AgenticRAGSystem()


st.title("Agentic RAG")
with st.form("search_form"):
    query = st.text_input("Ask a question")
    submitted = st.form_submit_button("Search")

if submitted and query:
    with st.spinner("Thinking..."):
        result = asyncio.run(get_system().query(query))

    st.markdown(result["answer"])

    if result["sources"]:
        st.divider()
        st.subheader("Sources")
        for s in result["sources"]:
            st.markdown(f"[{s['index']}. {s['title']}]({s['url']})")

    st.caption(
        f"Tool calls: {result['tool_calls_used']} · Latency: {result['latency_ms']:.0f}ms"
    )
