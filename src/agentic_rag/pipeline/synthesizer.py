"""LLM-based answer synthesis from retrieved context."""

from __future__ import annotations

from agentic_rag.llm.base import BaseLLM
from agentic_rag.models import PipelineContext
from agentic_rag.observability.langfuse import observation as _lf_obs


class Synthesizer:
    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    async def synthesize(self, query: str, ctx: PipelineContext) -> str:
        if not ctx.results:
            return "No relevant information found for your query."

        context_blocks = "\n\n".join(
            f"[{i + 1}] {r.get('title', '')}\n"
            f"Source: {r.get('source', '')}\n"
            f"{r.get('content', '')}"
            for i, r in enumerate(ctx.results)
        )

        chat_section = ""
        if ctx.chat_history:
            lines = [
                f"{m['role']}: {m['content']}"
                for m in ctx.chat_history
                if m.get("content")
            ]
            if lines:
                chat_section = "\n\nConversation so far:\n" + "\n".join(lines)

        prompt = (
            "You are a helpful assistant. Using the sources below, answer the question "
            "directly and concisely. Cite sources inline with [N] notation. "
            "If the sources contain relevant information, use it without disclaimers. "
            "Only say you lack information if the sources genuinely contain nothing relevant.\n\n"
            f"Context:\n{context_blocks}"
            f"{chat_section}\n\n"
            f"Question: {query}"
        )

        with _lf_obs("synthesize", as_type="generation", input={"query": query}):
            return await self._llm.chat(prompt)
