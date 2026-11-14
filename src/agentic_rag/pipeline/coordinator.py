"""Pipeline coordinator: runs sources in priority order, reranks, synthesizes."""

from __future__ import annotations

import dataclasses
import logging
import re
import time

from agentic_rag.catalog.select import (
    cards_to_sources,
    ensure_incomplete_note,
    cited_card_indices,
    select_cards,
)
from agentic_rag.catalog.store import CardStore, PageCard
from agentic_rag.config import CatalogConfig
from agentic_rag.llm.base import Embedder
from agentic_rag.models import ChatMessage, PipelineContext, QueryResult, SearchResult
from agentic_rag.observability.langfuse import observation as _lf_obs
from agentic_rag.pipeline.memory import ConversationMemory
from agentic_rag.pipeline.router import QueryRouter, RouteDecision
from agentic_rag.pipeline.sources import BaseSource
from agentic_rag.pipeline.synthesizer import Synthesizer
from agentic_rag.cache.semantic_cache import SemanticCache
from agentic_rag.retrieval.chroma import ChromaVectorStore
from agentic_rag.retrieval.reranker import _USE_CONFIGURED, CrossEncoderReranker

logger = logging.getLogger(__name__)

NO_INFO = "No relevant information found for your query."
_CHUNK_SUFFIX = re.compile(r"_chunk_\d+$")


def page_ids_from_sources(sources: list[SearchResult]) -> list[str]:
    """Page ids of card/chunk sources: order kept, unique."""
    ids = [_CHUNK_SUFFIX.sub("", s.id) for s in sources]
    return list(dict.fromkeys(ids))


class PipelineCoordinator:
    def __init__(
        self,
        sources: list[BaseSource],
        reranker: CrossEncoderReranker,
        synthesizer: Synthesizer,
        memory: ConversationMemory,
        max_tool_calls: int,
        embed_llm: Embedder,
        cache: SemanticCache | None = None,
        router: QueryRouter | None = None,
        card_store: CardStore | None = None,
        page_fetcher: ChromaVectorStore | None = None,
        catalog_config: CatalogConfig | None = None,
    ) -> None:
        self._sources = sources
        self._reranker = reranker
        self._synthesizer = synthesizer
        self._memory = memory
        self._max_tool_calls = max_tool_calls
        self._embed_llm = embed_llm
        self._cache = cache
        self._router = router
        self._card_store = card_store
        self._page_fetcher = page_fetcher
        self._catalog_config = catalog_config or CatalogConfig()
        # (page_id, title) of the last answer's sources, per thread.
        self._last_pages: dict[str, list[tuple[str, str]]] = {}

    def seed_history(self, thread_id: str, messages: list[ChatMessage]) -> None:
        """Preload a thread with (user, assistant) message pairs, e.g. for the eval."""
        for user, assistant in zip(messages[::2], messages[1::2]):
            self._memory.append(thread_id, user.content, assistant.content)

    async def query(
        self, user_query: str, thread_id: str = "default", use_web: bool = False
    ) -> QueryResult:
        if not user_query:
            raise ValueError("user_query cannot be empty")

        chat_history = self._memory.get(thread_id)
        previous = self._last_pages.get(thread_id, [])
        if self._router is None:
            decision = RouteDecision(user_query, "chunks", False)
        else:
            decision = await self._router.route(user_query, chat_history, previous)

        # Embed once; reused by the cache lookup, retrieval, and cache store.
        # On failure, leave None — downstream falls back to embedding itself.
        query_vec: list[float] | None = None
        try:
            query_vec = await self._embed_llm.embed(decision.standalone_query)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "embed() failed up front (%s); cache skipped, retrieval will retry",
                exc,
            )

        # The cache key is the query alone, so only stand-alone questions are
        # safe to share: a follow-up's meaning depends on this thread's history.
        # Web results are live, so they never go through the cache either.
        cache = None if chat_history or use_web else self._cache

        if cache is not None:
            cached = await cache.get(user_query, query_vec)
            if cached is not None:
                logger.debug("cache hit for query len=%d", len(user_query))
                self._memory.append(thread_id, user_query, cached.answer)
                self._remember_pages(thread_id, cached.sources)
                return dataclasses.replace(cached, route=decision.route)

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
        sources: list[SearchResult] = []
        with _lf_obs(
            "pipeline.query", as_type="span", input={"query": user_query}
        ) as obs:
            if obs is not None:
                trace_id = obs.trace_id
            try:
                if decision.route == "catalog":
                    ctx.final_answer, sources = await self._run_catalog(
                        user_query, decision, ctx
                    )
                elif decision.route == "both":
                    ctx.final_answer, sources = await self._run_both(
                        user_query, decision, ctx, use_web
                    )
                else:
                    ctx.final_answer, sources = await self._run_chunks(
                        user_query, decision, ctx, previous, use_web
                    )

            except Exception as exc:
                logger.exception("PipelineCoordinator error: %s", exc)
                ctx.error = str(exc)
                # Detail stays in ctx.error/logs; the user-facing string is fixed so
                # internal paths and provider errors never reach the browser.
                ctx.final_answer = (
                    "Something went wrong answering that. Please try again."
                )

        self._memory.append(thread_id, user_query, ctx.final_answer or "")
        if ctx.error is None:
            self._remember_pages(thread_id, sources)
        latency_ms = round((time.monotonic() - t0) * 1000, 2)

        result = QueryResult(
            answer=ctx.final_answer or "",
            sources=sources,
            tool_calls_used=ctx.tool_calls,
            latency_ms=latency_ms,
            trace_id=trace_id,
            route=decision.route,
        )

        # Never cache the fixed error reply: it would mask recovery for the whole TTL.
        if cache is not None and ctx.error is None:
            await cache.set(user_query, result, query_vec)

        return result

    def _remember_pages(self, thread_id: str, sources: list[SearchResult]) -> None:
        titles = {s.id: s.title for s in sources}
        self._last_pages[thread_id] = [
            (pid, titles.get(pid) or titles.get(f"{pid}_chunk_0", ""))
            for pid in page_ids_from_sources(sources)
        ]

    def _select_cards(self, ctx: PipelineContext) -> list[PageCard]:
        if self._card_store is None:
            return []
        return select_cards(
            self._card_store.all(),
            ctx.query_vec,
            self._catalog_config.max_prompt_tokens,
            self._catalog_config.prefilter_top_n,
        )

    async def _search_sources(
        self, query: str, ctx: PipelineContext, sources: list[BaseSource]
    ) -> None:
        """Run sources in order; the first with relevant results fills ctx.results."""
        for source in sources:
            if ctx.tool_calls >= ctx.max_tool_calls:
                logger.warning("Circuit breaker: max tool calls reached")
                break
            new_results = await source.search(query, ctx)
            ctx.tool_calls += 1
            # Rerank per source: the cross-encoder relevance gate can
            # empty the list, letting off-topic queries fall through
            # to the next source (e.g. web) instead of stopping here.
            # The gate is KB-calibrated, so honor it only for sources
            # that opt in (source.gated). Ungated sources (web) reorder
            # without dropping — their scores aren't on the KB scale and
            # they have no fallback. Keyed on the source, not position.
            relevant = self._reranker.rerank(
                query,
                new_results,
                min_score=_USE_CONFIGURED if source.gated else None,
            )
            if relevant:
                logger.info(
                    "%s: %d relevant results, stopping", source.name, len(relevant)
                )
                ctx.results = relevant
                break
            logger.info("%s: no relevant results, falling through", source.name)

    async def _run_chunks(
        self,
        user_query: str,
        decision: RouteDecision,
        ctx: PipelineContext,
        previous: list[tuple[str, str]],
        use_web: bool,
    ) -> tuple[str, list[SearchResult]]:
        if decision.focus_previous and previous and self._page_fetcher is not None:
            ctx.results = await self._page_fetcher.fetch_by_page_ids(
                [p for p, _ in previous], self._catalog_config.focus_top_k
            )
        else:
            # Old construction (no router) keeps every source, as before.
            sources = [
                s
                for s in self._sources
                if use_web or s.name != "web" or self._router is None
            ]
            await self._search_sources(decision.standalone_query, ctx, sources)
        answer = await self._synthesizer.synthesize(user_query, ctx)
        return answer, list(ctx.results)

    async def _run_catalog(
        self, user_query: str, decision: RouteDecision, ctx: PipelineContext
    ) -> tuple[str, list[SearchResult]]:
        cards = self._select_cards(ctx)
        if not cards:
            # No catalog yet: answer from chunks, never web on this route.
            return await self._run_chunks(
                user_query,
                dataclasses.replace(decision, focus_previous=False),
                ctx,
                [],
                False,
            )
        answer = await self._synthesizer.synthesize_catalog(
            user_query, ctx.chat_history, cards, []
        )
        cited = cited_card_indices(answer, cards)
        return ensure_incomplete_note(answer, len(cited)), cards_to_sources(
            cards, cited
        )

    async def _run_both(
        self,
        user_query: str,
        decision: RouteDecision,
        ctx: PipelineContext,
        use_web: bool,
    ) -> tuple[str, list[SearchResult]]:
        cards = self._select_cards(ctx)
        rag = [s for s in self._sources if s.name == "rag"]
        await self._search_sources(decision.standalone_query, ctx, rag)
        chunks = list(ctx.results)
        if not cards and not chunks and use_web:
            web = [s for s in self._sources if s.name == "web"]
            await self._search_sources(decision.standalone_query, ctx, web)
            chunks = list(ctx.results)
        if not cards:
            return await self._synthesizer.synthesize(user_query, ctx), chunks
        answer = await self._synthesizer.synthesize_catalog(
            user_query, ctx.chat_history, cards, chunks
        )
        cited = cited_card_indices(answer, cards)
        return ensure_incomplete_note(answer, len(cited)), (
            cards_to_sources(cards, cited) + chunks
        )
