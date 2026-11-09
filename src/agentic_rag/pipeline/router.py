"""Query router: one LLM call rewrites a follow-up and picks a retrieval route."""

import logging
from dataclasses import dataclass
from typing import Literal

from agentic_rag.llm.base import ChatModel
from agentic_rag.llm.json_reply import extract_json_object
from agentic_rag.models import ChatMessage

logger = logging.getLogger(__name__)

Route = Literal["catalog", "chunks", "both"]
_ROUTES: tuple[Route, ...] = ("catalog", "chunks", "both")

_DEFINITIONS = """\
Routes:
- catalog: list, count, "do I have", "all info about X".
  Examples: "give me all my skills"; "do I have info on the Claude architect certificate?"
- chunks: questions about the content of something specific.
  Examples: "what does my Azure ML page say about endpoints?"; "summarize it" (focus_previous true when previous pages exist)
- both: unclear, or a mix.
  Examples: "what do I know about Claude certificates?"; "tell me about plugins"

focus_previous is true only when the question refers to the pages of the previous answer ("summarize it", "tell me more about the second one")."""


@dataclass(frozen=True)
class RouteDecision:
    standalone_query: str
    route: Route
    focus_previous: bool


def build_router_prompt(
    query: str,
    history: list[ChatMessage],
    previous_pages: list[tuple[str, str]],
    history_turns: int,
) -> str:
    recent = history[-history_turns * 2 :] if history_turns > 0 else []
    history_text = "\n".join(f"{m.role}: {m.content}" for m in recent) or "(none)"
    if previous_pages:
        pages_text = "\n".join(f"- {title}" for _, title in previous_pages)
    else:
        pages_text = (
            "(none: there is no previous answer, so focus_previous must be false)"
        )
    return (
        "Rewrite the new question as a standalone question and choose a route.\n\n"
        f"{_DEFINITIONS}\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"Pages cited in the previous answer:\n{pages_text}\n\n"
        f"New question: {query}\n\n"
        'Reply with JSON only: {"standalone_query": "...", "route": "catalog|chunks|both", '
        '"focus_previous": true|false}'
    )


def _fallback(original_query: str, reason: str) -> RouteDecision:
    logger.warning("Router fallback to 'both': %s", reason)
    return RouteDecision(original_query, "both", False)


def parse_route(text: str, original_query: str) -> RouteDecision:
    try:
        data = extract_json_object(text)
    except ValueError as exc:
        return _fallback(original_query, f"bad JSON ({exc})")
    route = data.get("route")
    if route not in _ROUTES:
        return _fallback(original_query, f"unknown route {route!r}")
    query = data.get("standalone_query")
    if not isinstance(query, str) or not query.strip():
        return _fallback(original_query, "empty query")
    focus = data.get("focus_previous")
    return RouteDecision(
        query.strip(), route, focus if isinstance(focus, bool) else False
    )


class QueryRouter:
    def __init__(self, llm: ChatModel, history_turns: int) -> None:
        self._llm = llm
        self._history_turns = history_turns

    async def route(
        self,
        query: str,
        history: list[ChatMessage],
        previous_pages: list[tuple[str, str]],
    ) -> RouteDecision:
        prompt = build_router_prompt(
            query, history, previous_pages, self._history_turns
        )
        try:
            reply = await self._llm.chat(prompt)
        except (RuntimeError, ValueError) as exc:
            return _fallback(query, f"LLM error ({exc})")
        decision = parse_route(reply, query)
        if not history:
            return RouteDecision(query, decision.route, decision.focus_previous)
        return decision
