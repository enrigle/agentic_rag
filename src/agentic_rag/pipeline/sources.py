"""Retrieval source implementations for the pipeline."""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol, runtime_checkable

from tavily import AsyncTavilyClient

from agentic_rag.llm.base import BaseLLM
from agentic_rag.models import PipelineContext, SearchResult
from agentic_rag.observability.langfuse import observation as _lf_obs
from agentic_rag.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


@runtime_checkable
class BaseSource(Protocol):
    name: str
    # Whether the KB-calibrated reranker gate applies to this source. False for
    # sources whose scores aren't on the KB scale and that have no fallback
    # (e.g. web) — gating them would drop answerable results with nowhere to
    # fall through to. Decoupled from list position on purpose.
    gated: bool

    async def search(self, query: str, ctx: PipelineContext) -> list[SearchResult]: ...


class RAGSource:
    name = "rag"
    gated = True

    def __init__(self, llm: BaseLLM, hybrid: HybridRetriever) -> None:
        self._llm = llm
        self._hybrid = hybrid

    async def search(self, query: str, ctx: PipelineContext) -> list[SearchResult]:
        with _lf_obs("rag_search", as_type="retriever", input={"query": query}):
            query_vec = ctx.query_vec or await self._llm.embed(query)
            results = await self._hybrid.search(query_vec=query_vec, query_text=query)
        logger.info("RAGSource: %d results", len(results))
        return results


class WebSource:
    name = "web"
    gated = False

    async def search(self, query: str, ctx: PipelineContext) -> list[SearchResult]:
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            logger.warning("WebSource: TAVILY_API_KEY not set, skipping")
            return []
        with _lf_obs("web_search", as_type="span", input={"query": query}):
            client = AsyncTavilyClient(api_key=api_key)
            response = await client.search(query, max_results=5)
        raw: list[dict[str, Any]] = response.get("results", [])
        # Tavily response is external JSON; convert to SearchResult at this boundary.
        results = [
            SearchResult(
                id=r["url"],
                title=r.get("title", ""),
                source=r["url"],
                content=r["content"],
                score=float(r.get("score", 1.0)),
            )
            for r in raw
            if r.get("url") and r.get("content")
        ]
        logger.info("WebSource: %d results", len(results))
        return results
