"""Main orchestration class for the agentic RAG pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Literal

from duckduckgo_search import DDGS
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentic_rag.config import RAGConfig, load_config
from agentic_rag.llm.base import BaseLLM
from agentic_rag.llm.ollama import OllamaLLM
from agentic_rag.models import AgentState, QueryResult, SearchResult
from agentic_rag.retrieval.base import BaseKeywordRetriever, BaseVectorStore
from agentic_rag.retrieval.bm25 import BM25Retriever
from agentic_rag.retrieval.chroma import ChromaVectorStore
from agentic_rag.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Agentic RAG pipeline with LangGraph orchestration."""

    def __init__(
        self,
        llm: BaseLLM,
        vector_store: BaseVectorStore,
        keyword_retriever: BaseKeywordRetriever | None,
        config: RAGConfig,
    ) -> None:
        self._llm = llm
        self._vector_store = vector_store
        self._keyword_retriever = keyword_retriever
        self._config = config
        self._hybrid = HybridRetriever(vector_store, keyword_retriever, config)
        self._graph: CompiledStateGraph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph agent graph."""
        graph: StateGraph = StateGraph(AgentState)

        graph.add_node("analyze", self.analyze_query)
        graph.add_node("rag_search", self.rag_search)
        graph.add_node("web_search", self.web_search)
        graph.add_node("synthesize", self.synthesize)

        graph.add_edge(START, "analyze")
        graph.add_edge("analyze", "rag_search")
        graph.add_conditional_edges(
            "rag_search",
            self.should_web_search,
            path_map={"web_search": "web_search", "synthesize": "synthesize"},
        )
        graph.add_edge("web_search", "synthesize")
        graph.add_edge("synthesize", END)

        return graph.compile(checkpointer=MemorySaver())

    async def analyze_query(self, state: AgentState) -> AgentState:
        """Classify query to determine whether web search is needed."""
        if state["tool_calls"] >= state["max_tool_calls"]:
            logger.warning("Circuit breaker triggered in analyze_query")
            return {
                **state,
                "error": "Circuit breaker: max tool calls reached",
                "needs_web_search": False,
            }

        prompt = (
            "You are a query classifier. Respond with ONLY valid JSON, no prose. "
            'Format: {"needs_web_search": <bool>, "reason": "<str>"}\n\n'
            "Does this query require current web information "
            "or can it be answered from a static knowledge base?\n\n"
            f"Query: {state['query']}"
        )

        try:
            text = await self._llm.chat(prompt)
            start = text.find("{")
            end = text.rfind("}") + 1
            parsed: dict[str, Any] = json.loads(text[start:end])
            needs_web_search = bool(parsed.get("needs_web_search", False))
            logger.info(
                "analyze_query: needs_web_search=%s reason=%s",
                needs_web_search,
                parsed.get("reason"),
            )
            return {
                **state,
                "needs_web_search": needs_web_search,
                "tool_calls": state["tool_calls"] + 1,
            }
        except json.JSONDecodeError:
            logger.warning(
                "analyze_query: failed to parse LLM JSON response; defaulting to no web search"
            )
            return {
                **state,
                "needs_web_search": False,
                "tool_calls": state["tool_calls"] + 1,
            }
        except Exception as exc:
            logger.exception("analyze_query: unexpected error: %s", exc)
            return {
                **state,
                "error": str(exc),
                "tool_calls": state["tool_calls"] + 1,
            }

    async def rag_search(self, state: AgentState) -> AgentState:
        """Hybrid search via HybridRetriever (vector + BM25, merged with RRF)."""
        if state.get("error"):
            logger.warning(
                "rag_search: skipping due to prior error: %s", state["error"]
            )
            return {**state, "rag_results": [], "tool_calls": state["tool_calls"] + 1}

        if state["tool_calls"] >= state["max_tool_calls"]:
            logger.warning("Circuit breaker triggered in rag_search")
            return {
                **state,
                "error": "Circuit breaker: max tool calls reached",
                "rag_results": [],
            }

        try:
            query_vec = await self._llm.embed(state["query"])
            results: list[SearchResult] = await self._hybrid.search(
                query_vec=query_vec,
                query_text=state["query"],
            )

            if not results:
                logger.info("rag_search: no results after RRF — routing to web search")
                return {
                    **state,
                    "rag_results": [],
                    "needs_web_search": True,
                    "tool_calls": state["tool_calls"] + 1,
                }

            rag_results: list[dict[str, Any]] = [
                {
                    "id": r.id,
                    "source": r.source,
                    "title": r.title,
                    "content": r.content,
                    "score": r.score,
                }
                for r in results
            ]
            logger.info("rag_search: hybrid returned %d results", len(rag_results))
            return {
                **state,
                "rag_results": rag_results,
                "tool_calls": state["tool_calls"] + 1,
            }

        except Exception as exc:
            logger.exception("rag_search: error: %s", exc)
            return {
                **state,
                "error": str(exc),
                "rag_results": [],
                "tool_calls": state["tool_calls"] + 1,
            }

    async def web_search(self, state: AgentState) -> AgentState:
        """Search the web via DuckDuckGo."""
        if state.get("error"):
            logger.warning(
                "web_search: skipping due to prior error: %s", state["error"]
            )
            return {**state, "web_results": [], "tool_calls": state["tool_calls"] + 1}

        if state["tool_calls"] >= state["max_tool_calls"]:
            logger.warning("Circuit breaker triggered in web_search")
            return {
                **state,
                "error": "Circuit breaker: max tool calls reached",
                "web_results": [],
            }

        try:
            loop = asyncio.get_running_loop()
            raw: list[dict[str, Any]] = await loop.run_in_executor(
                None,
                lambda: DDGS().text(state["query"], max_results=5),
            )
            web_results: list[dict[str, Any]] = [
                {
                    "id": r.get("href", ""),
                    "source": r.get("href", ""),
                    "title": r.get("title", ""),
                    "content": r.get("body", ""),
                    "score": 1.0,
                }
                for r in (raw or [])
                if r.get("href") and r.get("body")
            ]
            logger.info("web_search: returned %d results", len(web_results))
            return {
                **state,
                "web_results": web_results,
                "tool_calls": state["tool_calls"] + 1,
            }
        except Exception as exc:
            logger.exception("web_search: error: %s", exc)
            return {
                **state,
                "error": str(exc),
                "web_results": [],
                "tool_calls": state["tool_calls"] + 1,
            }

    def should_web_search(
        self, state: AgentState
    ) -> Literal["web_search", "synthesize"]:
        """Conditional edge: decide if web search is needed."""
        if state.get("error"):
            return "synthesize"
        if state["needs_web_search"] and state["tool_calls"] < state["max_tool_calls"]:
            return "web_search"
        return "synthesize"

    async def synthesize(self, state: AgentState) -> AgentState:
        """Generate final answer using LLM with retrieved context."""
        rag_results: list[dict[str, Any]] = state.get("rag_results") or []
        web_results: list[dict[str, Any]] = state.get("web_results") or []
        all_results = rag_results + web_results

        if state["tool_calls"] >= state["max_tool_calls"]:
            logger.warning("Circuit breaker triggered in synthesize; returning best-effort")
            if all_results:
                answer = f"Based on {len(all_results)} retrieved sources (circuit breaker reached)."
            else:
                answer = "Unable to answer: circuit breaker reached with no results."
            return {**state, "final_answer": answer}

        if state.get("error") and not all_results:
            return {
                **state,
                "final_answer": f"Unable to answer: {state['error']}",
                "tool_calls": state["tool_calls"] + 1,
            }

        context_blocks = "\n\n".join(
            f"[{i + 1}] {r['title']}\nSource: {r['source']}\n{r['content']}"
            for i, r in enumerate(all_results)
        )

        prompt = (
            "You are a helpful assistant. Answer using ONLY the provided context. "
            "Cite sources inline using [N] notation. Do not fabricate information.\n\n"
            f"Context:\n{context_blocks}\n\nQuestion: {state['query']}"
        )

        try:
            answer = await self._llm.chat(prompt)
            return {
                **state,
                "final_answer": answer,
                "tool_calls": state["tool_calls"] + 1,
            }
        except Exception as exc:
            logger.exception("synthesize: generation failed: %s", exc)
            return {
                **state,
                "final_answer": f"Generation failed: {exc}",
                "tool_calls": state["tool_calls"] + 1,
            }

    async def query(self, user_query: str, thread_id: str = "default") -> QueryResult:
        """Execute the full agentic pipeline and return a QueryResult."""
        if not user_query:
            raise ValueError("user_query cannot be empty")

        initial_state: AgentState = {
            "query": user_query,
            "chat_history": [],
            "tool_calls": 0,
            "max_tool_calls": self._config.max_tool_calls,
            "rag_results": None,
            "web_results": None,
            "needs_web_search": False,
            "final_answer": None,
            "error": None,
        }

        graph_config = {"configurable": {"thread_id": thread_id}}
        t0 = time.monotonic()

        try:
            final_state: AgentState = await self._graph.ainvoke(
                initial_state, config=graph_config
            )
        except Exception as exc:
            logger.exception("query: graph invocation failed: %s", exc)
            latency_ms = (time.monotonic() - t0) * 1000
            return QueryResult(
                answer=f"Pipeline failed: {exc}",
                sources=[],
                tool_calls_used=0,
                latency_ms=round(latency_ms, 2),
            )

        latency_ms = (time.monotonic() - t0) * 1000

        rag_results = final_state.get("rag_results") or []
        web_results = final_state.get("web_results") or []
        sources: list[SearchResult] = [
            SearchResult(
                id=r.get("id", ""),
                title=r.get("title", ""),
                source=r.get("source", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
            )
            for r in rag_results + web_results
        ]

        return QueryResult(
            answer=final_state.get("final_answer") or "",
            sources=sources,
            tool_calls_used=final_state["tool_calls"],
            latency_ms=round(latency_ms, 2),
        )


def create_pipeline(config: RAGConfig | None = None) -> RAGPipeline:
    """Wire concrete Ollama + Chroma + BM25 implementations from config.

    Args:
        config: RAGConfig instance. If None, calls load_config() to load
                from the default YAML file or use defaults.

    Returns:
        A fully configured RAGPipeline instance.
    """
    if config is None:
        config = load_config()

    llm = OllamaLLM(config.llm)
    vector_store = ChromaVectorStore(config)
    bm25_retriever = BM25Retriever(config)

    return RAGPipeline(llm, vector_store, bm25_retriever, config)
