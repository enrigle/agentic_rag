"""Retrieval source implementations for the pipeline."""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol, runtime_checkable

from tavily import AsyncTavilyClient  # type: ignore[import-untyped]

from agentic_rag.llm.base import BaseLLM
from agentic_rag.models import PipelineContext
from agentic_rag.observability.langfuse import observation as _lf_obs
from agentic_rag.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


@runtime_checkable
class BaseSource(Protocol):
    name: str

    async def search(
        self, query: str, ctx: PipelineContext
    ) -> list[dict[str, Any]]: ...


class RAGSource:
    name = "rag"

    def __init__(self, llm: BaseLLM, hybrid: HybridRetriever) -> None:
        self._llm = llm
        self._hybrid = hybrid

    async def search(self, query: str, ctx: PipelineContext) -> list[dict[str, Any]]:
        with _lf_obs("rag_search", as_type="retriever", input={"query": query}):
            query_vec = await self._llm.embed(query)
            results = await self._hybrid.search(query_vec=query_vec, query_text=query)
        logger.info("RAGSource: %d results", len(results))
        return [
            {
                "id": r.id,
                "source": r.source,
                "title": r.title,
                "content": r.content,
                "score": r.score,
            }
            for r in results
        ]


class WebSource:
    name = "web"

    async def search(self, query: str, ctx: PipelineContext) -> list[dict[str, Any]]:
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            logger.warning("WebSource: TAVILY_API_KEY not set, skipping")
            return []
        with _lf_obs("web_search", as_type="span", input={"query": query}):
            client = AsyncTavilyClient(api_key=api_key)
            response = await client.search(query, max_results=5)
        raw: list[dict[str, Any]] = response.get("results", [])
        results = [
            {
                "id": r.get("url", ""),
                "source": r.get("url", ""),
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 1.0),
            }
            for r in raw
            if r.get("url") and r.get("content")
        ]
        logger.info("WebSource: %d results", len(results))
        return results
