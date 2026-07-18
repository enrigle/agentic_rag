"""LLM-as-judge: classifies why a RAG response failed."""

from __future__ import annotations

import json
import logging
from typing import Any

from agentic_rag.config import load_config
from agentic_rag.llm.base import BaseLLM

logger = logging.getLogger(__name__)

_VALID_CATEGORIES = {"retrieval_miss", "synthesis_failure", "missing_content"}

_PROMPT_TEMPLATE = """\
You are evaluating a RAG system response. Classify why it failed.

Query: {query}
Answer: {answer}
Sources retrieved:
{sources_block}

Pick exactly one failure category:
- retrieval_miss: sources retrieved are irrelevant to the query
- synthesis_failure: sources are relevant but the answer is wrong or incomplete
- missing_content: the topic does not exist in the knowledge base

Respond with JSON only, no other text: {{"category": "..."}}"""


def _build_synth_llm() -> BaseLLM:
    from agentic_rag.llm.ollama import OllamaLLM
    from agentic_rag.llm.openai_compat import AzureOpenAILLM, GroqLLM

    cfg = load_config()
    if cfg.groq.is_configured():
        return GroqLLM(cfg.groq)
    if cfg.azure_openai.is_configured():
        return AzureOpenAILLM(cfg.azure_openai)
    return OllamaLLM(cfg.llm)


async def classify_failure(
    query: str,
    answer: str,
    sources: list[dict[str, Any]],
) -> str:
    """Classify a thumbs-down response. Returns one of three categories or 'unknown'."""
    sources_block = (
        "\n".join(
            f'  {i + 1}. "{s.get("title", "")}" — "{str(s.get("content", ""))[:350]}..."'
            for i, s in enumerate(sources)
        )
        or "  (no sources)"
    )

    prompt = _PROMPT_TEMPLATE.format(
        query=query,
        answer=answer,
        sources_block=sources_block,
    )

    try:
        llm = _build_synth_llm()
        text = await llm.chat(prompt)
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning("judge: no JSON object found in response")
            return "unknown"
        parsed: dict[str, Any] = json.loads(text[start:end])
        category: str = parsed.get("category", "unknown")
        if category not in _VALID_CATEGORIES:
            logger.warning(
                "judge: unexpected category %r; defaulting to unknown", category
            )
            return "unknown"
        return category
    except (json.JSONDecodeError, KeyError, ValueError):
        logger.warning("judge: failed to parse LLM response")
        return "unknown"
    except Exception as exc:
        logger.warning("judge: unexpected error: %s", exc)
        return "unknown"
