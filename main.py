import json
import logging
import asyncio
import time
from typing import Any, Literal, TypedDict

import chromadb
import ollama
from duckduckgo_search import DDGS
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State shared across all nodes in the agent graph."""

    query: str
    chat_history: list[dict[str, Any]]
    tool_calls: int
    max_tool_calls: int
    rag_results: list[dict[str, Any]] | None
    web_results: list[dict[str, Any]] | None
    needs_web_search: bool
    final_answer: str | None
    error: str | None


class AgenticRAGSystem:
    """Production agentic RAG with LangGraph orchestration and Ollama."""

    def __init__(
        self,
        model: str = "llama3.2",
        max_tool_calls: int = 5,
        chroma_path: str = "./chroma_db",
    ) -> None:
        self.model = model
        self.client = ollama.AsyncClient()
        self.max_tool_calls = max_tool_calls
        self.embed_model = "nomic-embed-text"
        self.chroma = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma.get_or_create_collection("notion_kb")
        self.graph = self._build_graph()

    async def _invoke_ollama(self, prompt: str) -> str:
        """Call local Ollama daemon asynchronously."""
        response = await self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.message.content  # type: ignore[return-value]

    def _build_graph(self) -> CompiledStateGraph:
        """Build the LangGraph agent graph."""
        graph = StateGraph(AgentState)

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
        """Analyze query to determine which tools are needed."""
        if state["tool_calls"] >= state["max_tool_calls"]:
            logger.warning("Circuit breaker triggered in analyze_query")
            return {
                **state,
                "error": "Circuit breaker: max tool calls reached",
                "needs_web_search": False,
                "tool_calls": state["tool_calls"] + 1,
            }

        prompt = (
            "You are a query classifier. Respond with ONLY valid JSON, no prose. "
            'Format: {"needs_web_search": <bool>, "reason": "<str>"}\n\n'
            "Does this query require current web information "
            "or can it be answered from a static knowledge base?\n\n"
            f"Query: {state['query']}"
        )

        try:
            text = await self._invoke_ollama(prompt)
            # Extract JSON from response (model may wrap it in markdown fences)
            start = text.find("{")
            end = text.rfind("}") + 1
            parsed = json.loads(text[start:end])
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
                "analyze_query: failed to parse Ollama JSON response; defaulting to no web search"
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
        """Search internal knowledge base via ChromaDB + Ollama embeddings."""
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
                "tool_calls": state["tool_calls"] + 1,
            }

        try:
            embed_resp = await self.client.embed(
                model=self.embed_model, input=state["query"]
            )
            query_vec: list[float] = embed_resp["embeddings"][0]

            loop = asyncio.get_running_loop()
            results: dict[str, Any] = await loop.run_in_executor(
                None,
                lambda: self.collection.query(
                    query_embeddings=[query_vec],
                    n_results=5,
                    include=["documents", "distances", "metadatas"],
                ),
            )

            rag_results: list[dict[str, Any]] = [
                {
                    "source": meta.get("source", ""),
                    "title": meta.get("title", ""),
                    "content": doc,
                    "score": round(1 - dist, 4),
                }
                for doc, dist, meta in zip(
                    results["documents"][0],
                    results["distances"][0],
                    results["metadatas"][0],
                )
            ]
            logger.info("rag_search: returned %d results", len(rag_results))
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
                "tool_calls": state["tool_calls"] + 1,
            }

        try:
            loop = asyncio.get_running_loop()
            raw: list[dict[str, Any]] = await loop.run_in_executor(
                None,
                lambda: DDGS().text(state["query"], max_results=5),
            )
            web_results: list[dict[str, Any]] = [
                {
                    "source": r["href"],
                    "title": r["title"],
                    "content": r["body"],
                    "score": 1.0,
                }
                for r in (raw or [])
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
        """Generate final answer using Ollama with retrieved context."""
        rag_results: list[dict[str, Any]] = state.get("rag_results") or []
        web_results: list[dict[str, Any]] = state.get("web_results") or []
        all_results = rag_results + web_results

        if state["tool_calls"] >= state["max_tool_calls"]:
            logger.warning("Circuit breaker triggered in synthesize")

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
            answer = await self._invoke_ollama(prompt)
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

    async def query(
        self, user_query: str, thread_id: str = "default"
    ) -> dict[str, Any]:
        """Execute the full agentic pipeline."""
        if not user_query:
            raise ValueError("user_query cannot be empty")

        initial_state: AgentState = {
            "query": user_query,
            "chat_history": [],
            "tool_calls": 0,
            "max_tool_calls": self.max_tool_calls,
            "rag_results": None,
            "web_results": None,
            "needs_web_search": False,
            "final_answer": None,
            "error": None,
        }

        config = {"configurable": {"thread_id": thread_id}}
        t0 = time.monotonic()

        try:
            final_state: AgentState = await self.graph.ainvoke(
                initial_state, config=config
            )
        except Exception as exc:
            logger.exception("query: graph invocation failed: %s", exc)
            latency_ms = (time.monotonic() - t0) * 1000
            return {
                "answer": f"Pipeline failed: {exc}",
                "sources": [],
                "tool_calls_used": 0,
                "latency_ms": round(latency_ms, 2),
            }

        latency_ms = (time.monotonic() - t0) * 1000

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    system = AgenticRAGSystem()
    result = asyncio.run(system.query("How to deploy ML models in web apps?"))
    print(json.dumps(result, indent=2))
