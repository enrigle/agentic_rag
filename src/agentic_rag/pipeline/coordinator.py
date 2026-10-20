"""Pipeline coordinator: runs sources in priority order, reranks, synthesizes."""

from __future__ import annotations

import logging
import time

from agentic_rag.llm.base import BaseLLM
from agentic_rag.models import PipelineContext, QueryResult, SearchResult
from agentic_rag.observability.langfuse import observation as _lf_obs
from agentic_rag.pipeline.memory import ConversationMemory
from agentic_rag.pipeline.sources import BaseSource
from agentic_rag.pipeline.synthesizer import Synthesizer
from agentic_rag.cache.semantic_cache import SemanticCache
from agentic_rag.retrieval.reranker import _USE_CONFIGURED, CrossEncoderReranker

logger = logging.getLogger(__name__)


class PipelineCoordinator:
    def __init__(
        self,
        sources: list[BaseSource],
        reranker: CrossEncoderReranker,
        synthesizer: Synthesizer,
        memory: ConversationMemory,
        max_tool_calls: int,
        embed_llm: BaseLLM,
        cache: SemanticCache | None = None,
    ) -> None:
        self._sources = sources
        self._reranker = reranker
        self._synthesizer = synthesizer
        self._memory = memory
        self._max_tool_calls = max_tool_calls
        self._embed_llm = embed_llm
        self._cache = cache

    async def query(self, user_query: str, thread_id: str = "default") -> QueryResult:
        if not user_query:
            raise ValueError("user_query cannot be empty")

        # Embed once; reused by the cache lookup, retrieval, and cache store.
        # On failure, leave None — downstream falls back to embedding itself.
        query_vec: list[float] | None = None
        try:
            query_vec = await self._embed_llm.embed(user_query)
        except Exception:  # noqa: BLE001
            logger.debug("embed() failed up front; downstream will retry/skip")

        # The cache key is the query alone, so only stand-alone questions are
        # safe to share: a follow-up's meaning depends on this thread's history.
        chat_history = self._memory.get(thread_id)
        cache = None if chat_history else self._cache

        if cache is not None:
            cached = await cache.get(user_query, query_vec)
            if cached is not None:
                logger.debug("cache hit for query len=%d", len(user_query))
                self._memory.append(thread_id, user_query, cached.answer)
                return cached

        t0 = time.monotonic()
        ctx = PipelineContext(
            query=user_query,
            chat_history=chat_history,
            results=[],
            final_answer=None,
            error=None,
            tool_calls=0,
            max_tool_calls=self._max_tool_calls,
            query_vec=query_vec,
        )

        trace_id: str | None = None
        with _lf_obs(
            "pipeline.query", as_type="span", input={"query": user_query}
        ) as obs:
            if obs is not None:
                trace_id = obs.trace_id
            try:
                for source in self._sources:
                    if ctx.tool_calls >= ctx.max_tool_calls:
                        logger.warning("Circuit breaker: max tool calls reached")
                        break
                    new_results = await source.search(user_query, ctx)
                    ctx.tool_calls += 1
                    # Rerank per source: the cross-encoder relevance gate can
                    # empty the list, letting off-topic queries fall through
                    # to the next source (e.g. web) instead of stopping here.
                    # The gate is KB-calibrated, so honor it only for sources
                    # that opt in (source.gated). Ungated sources (web) reorder
                    # without dropping — their scores aren't on the KB scale and
                    # they have no fallback. Keyed on the source, not position.
                    relevant = self._reranker.rerank(
                        user_query,
                        new_results,
                        min_score=_USE_CONFIGURED if source.gated else None,
                    )
                    if relevant:
                        logger.info(
                            "%s: %d relevant results, stopping",
                            source.name,
                            len(relevant),
                        )
                        ctx.results = relevant
                        break
                    logger.info("%s: no relevant results, falling through", source.name)

                ctx.final_answer = await self._synthesizer.synthesize(user_query, ctx)

            except Exception as exc:
                logger.exception("PipelineCoordinator error: %s", exc)
                ctx.error = str(exc)
                # Detail stays in ctx.error/logs; the user-facing string is fixed so
                # internal paths and provider errors never reach the browser.
                ctx.final_answer = (
                    "Something went wrong answering that. Please try again."
                )

        self._memory.append(thread_id, user_query, ctx.final_answer or "")
        latency_ms = round((time.monotonic() - t0) * 1000, 2)

        sources = [
            SearchResult(
                id=r.get("id", ""),
                title=r.get("title", ""),
                source=r.get("source", ""),
                content=r.get("content", ""),
                score=float(r.get("score", 0.0)),
                similarity=r.get("similarity"),
            )
            for r in ctx.results
        ]
        result = QueryResult(
            answer=ctx.final_answer or "",
            sources=sources,
            tool_calls_used=ctx.tool_calls,
            latency_ms=latency_ms,
            trace_id=trace_id,
        )

        # Never cache the fixed error reply: it would mask recovery for the whole TTL.
        if cache is not None and ctx.error is None:
            await cache.set(user_query, result, query_vec)

        return result
