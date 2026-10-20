"""LLM-based answer synthesis from retrieved context."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from agentic_rag.llm.base import ChatModel
from agentic_rag.models import PipelineContext
from agentic_rag.observability.langfuse import observation as _lf_obs

logger = logging.getLogger(__name__)


def load_few_shot_examples(path: Path) -> list[tuple[str, str]]:
    """Read (query, answer) pairs written by the optimizer.

    Missing file = no examples yet. A malformed file is logged and ignored so a
    bad write never breaks answering.
    """
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text())
        return [(str(e["query"]), str(e["answer"])) for e in raw["few_shot_examples"]]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Ignoring few-shot examples in %s: %s", path, exc)
        return []


def format_few_shot(examples: list[tuple[str, str]]) -> str:
    """Render examples as a prompt section; empty string when there are none."""
    if not examples:
        return ""
    blocks = "\n\n".join(
        f"Example question: {q}\nExample answer: {a}" for q, a in examples
    )
    return f"Examples of answers users rated helpful:\n{blocks}\n\n"


class Synthesizer:
    def __init__(self, llm: ChatModel, few_shot_path: Path | None = None) -> None:
        self._llm = llm
        # Re-read per call: the optimizer rewrites it while the pipeline stays cached.
        self._few_shot_path = few_shot_path

    async def synthesize(self, query: str, ctx: PipelineContext) -> str:
        if not ctx.results:
            return "No relevant information found for your query."

        context_blocks = "\n\n".join(
            f"[{i + 1}] {r.title}\nSource: {r.source}\n{r.content}"
            for i, r in enumerate(ctx.results)
        )

        chat_section = ""
        if ctx.chat_history:
            lines = [f"{m.role}: {m.content}" for m in ctx.chat_history if m.content]
            if lines:
                chat_section = "\n\nConversation so far:\n" + "\n".join(lines)

        examples = (
            load_few_shot_examples(self._few_shot_path) if self._few_shot_path else []
        )

        prompt = (
            "You are a helpful assistant. Using the sources below, answer the question "
            "directly and concisely. Cite sources inline with [N] notation. "
            "If the sources contain relevant information, use it without disclaimers. "
            "Only say you lack information if the sources genuinely contain nothing relevant.\n\n"
            f"{format_few_shot(examples)}"
            f"Context:\n{context_blocks}"
            f"{chat_section}\n\n"
            f"Question: {query}"
        )

        with _lf_obs("synthesize", as_type="generation", input={"query": query}):
            return await self._llm.chat(prompt)
