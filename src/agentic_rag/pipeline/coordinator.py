"""Pipeline coordinator: runs sources in priority order, reranks, synthesizes."""

from __future__ import annotations

import logging
import time

from agentic_rag.models import PipelineContext, QueryResult, SearchResult
from agentic_rag.observability.langfuse import observation as _lf_obs
from agentic_rag.pipeline.memory import ConversationMemory
from agentic_rag.pipeline.sources import BaseSource
from agentic_rag.pipeline.synthesizer import Synthesizer
from agentic_rag.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class PipelineCoordinator:
    def __init__(
        self,
        sources: list[BaseSource],
        reranker: CrossEncoderReranker,
        synthesizer: Synthesizer,
        memory: ConversationMemory,
        threshold: float,
        max_tool_calls: int,
    ) -> None:
        self._sources = sources
        self._reranker = reranker
        self._synthesizer = synthesizer
        self._memory = memory
        self._threshold = threshold
        self._max_tool_calls = max_tool_calls

    async def query(self, user_query: str, thread_id: str = "default") -> QueryResult:
        if not user_query:
            raise ValueError("user_query cannot be empty")

        t0 = time.monotonic()
        ctx = PipelineContext(
            query=user_query,
            chat_history=self._memory.get(thread_id),
            results=[],
            final_answer=None,
            error=None,
            tool_calls=0,
            max_tool_calls=self._max_tool_calls,
        )

        with _lf_obs("pipeline.query", as_type="span", input={"query": user_query}):
            try:
                for source in self._sources:
                    if ctx.tool_calls >= ctx.max_tool_calls:
                        logger.warning("Circuit breaker: max tool calls reached")
                        break
                    new_results = await source.search(user_query, ctx)
                    ctx.results.extend(new_results)
                    ctx.tool_calls += 1
                    if new_results:
                        best = max(r["score"] for r in new_results)
                        if best >= self._threshold:
                            logger.info(
                                "%s: best score %.3f >= threshold %.3f, stopping",
                                source.name,
                                best,
                                self._threshold,
                            )
                            break

                ctx.results = self._reranker.rerank(user_query, ctx.results)
                ctx.final_answer = await self._synthesizer.synthesize(user_query, ctx)

            except Exception as exc:
                logger.exception("PipelineCoordinator error: %s", exc)
                ctx.error = str(exc)
                ctx.final_answer = f"Pipeline error: {exc}"

        self._memory.append(thread_id, user_query, ctx.final_answer or "")
        latency_ms = round((time.monotonic() - t0) * 1000, 2)

        sources = [
            SearchResult(
                id=r.get("id", ""),
                title=r.get("title", ""),
                source=r.get("source", ""),
                content=r.get("content", ""),
                score=float(r.get("score", 0.0)),
            )
            for r in ctx.results
        ]
        return QueryResult(
            answer=ctx.final_answer or "",
            sources=sources,
            tool_calls_used=ctx.tool_calls,
            latency_ms=latency_ms,
        )
