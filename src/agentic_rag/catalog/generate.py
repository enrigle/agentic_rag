"""Turn page text into card fields (summary, tags, items) and merge tag synonyms."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, replace
from typing import Any

from agentic_rag.catalog.store import PageCard
from agentic_rag.llm.base import ChatModel
from agentic_rag.llm.json_reply import extract_json_object

logger = logging.getLogger(__name__)

MAX_ITEMS = 50


@dataclass
class CardFields:
    summary: str
    tags: list[str]
    items: list[str]
    items_truncated: bool


def build_card_prompt(title: str, page_text: str, max_page_chars: int = 24000) -> str:
    return (
        "Describe this Notion page for a catalog. Reply with JSON only, no other text:\n"
        '{"summary": "2-3 sentences", "tags": ["short lowercase topic", ...], "items": [...]}\n'
        "items = named things the page lists (skills, tools, repos, commands), "
        "copied exactly as written. Use [] if the page lists none.\n\n"
        f"Title: {title}\n\nPage text:\n{page_text[:max_page_chars]}"
    )


def _str_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{key!r} must be a list of strings")
    return value


def parse_card_reply(text: str, page_text: str) -> CardFields:
    data = extract_json_object(text)
    summary = data.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("card reply has no summary")
    tags = list(
        dict.fromkeys(t.strip().lower() for t in _str_list(data, "tags") if t.strip())
    )
    haystack = page_text.lower()
    items = [
        i
        for i in dict.fromkeys(_str_list(data, "items"))
        if i and i.lower() in haystack
    ]
    return CardFields(
        summary=summary.strip(),
        tags=tags,
        items=items[:MAX_ITEMS],
        items_truncated=len(items) > MAX_ITEMS,
    )


async def generate_card_fields(
    llm: ChatModel,
    title: str,
    page_text: str,
    retries: int = 3,
    max_page_chars: int = 24000,
) -> CardFields:
    """Retry RuntimeError from the backend with backoff. ValueError is not retried."""
    prompt = build_card_prompt(title, page_text, max_page_chars)
    for attempt in range(retries):
        try:
            reply = await llm.chat(prompt)
        except RuntimeError as exc:
            if attempt == retries - 1:
                raise
            logger.warning("card generation failed (attempt %d): %s", attempt + 1, exc)
            await asyncio.sleep(2**attempt)
        else:
            return parse_card_reply(reply, page_text)
    raise RuntimeError("generate_card_fields needs retries >= 1")


def build_tag_merge_prompt(tags: list[str]) -> str:
    return (
        "Merge synonym or near-duplicate tags. Reply with JSON only: an object mapping "
        "each tag to the canonical lowercase tag to use. Tags to keep map to themselves.\n"
        f"Tags: {json.dumps(tags)}"
    )


def parse_tag_map(text: str, tags: list[str]) -> dict[str, str]:
    raw = extract_json_object(text)
    result: dict[str, str] = {}
    for tag in tags:
        target = raw.get(tag)
        result[tag] = (
            target.strip().lower()
            if isinstance(target, str) and target.strip()
            else tag
        )
    return result


def apply_tag_map(cards: list[PageCard], tag_map: dict[str, str]) -> list[PageCard]:
    return [
        replace(c, tags=list(dict.fromkeys(tag_map.get(t, t) for t in c.tags)))
        for c in cards
    ]
