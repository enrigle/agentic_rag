"""Pure helpers: pick catalog cards, build the catalog prompt, map citations."""

from __future__ import annotations

import math
import re

from agentic_rag.catalog.store import PageCard
from agentic_rag.models import ChatMessage, SearchResult

INCOMPLETE_NOTE = "Found {n} pages; this list may be incomplete."

_CITATION = re.compile(r"\[(\d+)\]")

_RULES = """\
You answer questions about the user's Notion pages using ONLY the numbered cards below. Do not invent pages.
- Cite cards as [n] only. Passages are context, not citable cards.
- For list or topic questions, say the count first, then group by sub-topic.
- When the question asks for things (e.g. "my skills"), list the matching items under the page that contains them, with the page link.
- When the question asks about a topic, list the pages, one line each, with a link.
- If a page states a count (e.g. "48 skills"), quote it and name the page.
- For "do I have ..." questions, start with "Yes" or "No", then list the matching pages with links."""


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def format_card(n: int, card: PageCard, parent_title: str | None = None) -> str:
    head = f"[{n}] {card.title} | {', '.join(card.tags)} | {card.summary} | {card.url}"
    if parent_title:
        head += f" | parent: {parent_title}"
    if not card.items:
        return head
    more = " (more on page)" if card.items_truncated else ""
    return f"{head}\nitems: {', '.join(card.items)}{more}"


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -math.inf
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return -math.inf
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def select_cards(
    cards: list[PageCard],
    query_vec: list[float] | None,
    max_prompt_tokens: int,
    top_n: int,
) -> list[PageCard]:
    block = "\n".join(format_card(i, c) for i, c in enumerate(cards, 1))
    if estimate_tokens(block) <= max_prompt_tokens:
        return list(cards)
    if query_vec is None:
        return cards[:top_n]
    # sorted() is stable: ties keep stored order
    ranked = sorted(cards, key=lambda c: _cosine(query_vec, c.embedding), reverse=True)
    return ranked[:top_n]


def build_catalog_prompt(
    query: str,
    history: list[ChatMessage],
    cards: list[PageCard],
    passages: list[SearchResult],
) -> str:
    titles = {c.page_id: c.title for c in cards}
    lines = [
        format_card(i, c, titles.get(c.parent_id) if c.parent_id else None)
        for i, c in enumerate(cards, 1)
    ]
    parts = [_RULES]
    if history:
        parts.append(
            "Conversation so far:\n"
            + "\n".join(f"{m.role}: {m.content}" for m in history)
        )
    parts.append("Matching pages:\n" + "\n".join(lines))
    if passages:
        parts.append(
            "Relevant passages (context only, not citable):\n"
            + "\n\n".join(f"{p.title}: {p.content}" for p in passages)
        )
    parts.append(f"Question: {query}\nAnswer:")
    return "\n\n".join(parts)


def extract_citations(answer: str, n_cards: int) -> list[int]:
    seen: list[int] = []
    for m in _CITATION.finditer(answer):
        n = int(m.group(1))
        if 1 <= n <= n_cards and n not in seen:
            seen.append(n)
    return seen


def cards_to_sources(cards: list[PageCard], cited: list[int]) -> list[SearchResult]:
    return [
        SearchResult(
            id=cards[n - 1].page_id,
            title=cards[n - 1].title,
            source=cards[n - 1].url,
            content=cards[n - 1].summary,
            score=1.0,
        )
        for n in cited
    ]


def ensure_incomplete_note(answer: str, n_cited: int) -> str:
    note = INCOMPLETE_NOTE.format(n=n_cited)
    if n_cited < 1 or note in answer:
        return answer
    return f"{answer}\n\n{note}"
