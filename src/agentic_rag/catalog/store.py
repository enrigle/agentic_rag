"""Per-page catalog cards persisted as an atomically written JSON file."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VERSION = 1


@dataclass
class PageCard:
    page_id: str
    title: str
    url: str
    last_edited_time: str
    summary: str
    tags: list[str]
    items: list[str]
    items_truncated: bool
    parent_id: str | None
    child_ids: list[str]
    embedding: list[float]  # of f"{title}\n{summary}", for the prefilter


def card_to_dict(card: PageCard) -> dict[str, Any]:
    return asdict(card)


def _str_list(raw: dict[str, Any], key: str) -> list[str]:
    value = raw.get(key)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"card field {key!r} must be a list of strings")
    return list(value)


def _str(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise ValueError(f"card field {key!r} must be a string")
    return value


def card_from_dict(raw: dict[str, Any]) -> PageCard:
    """Build a card from a dict. Raises ValueError on a missing or mistyped field."""
    parent_id = raw.get("parent_id")
    if parent_id is not None and not isinstance(parent_id, str):
        raise ValueError("card field 'parent_id' must be a string or null")
    truncated = raw.get("items_truncated")
    if not isinstance(truncated, bool):
        raise ValueError("card field 'items_truncated' must be a bool")
    embedding = raw.get("embedding")
    if not isinstance(embedding, list) or not all(
        isinstance(v, int | float) and not isinstance(v, bool) for v in embedding
    ):
        raise ValueError("card field 'embedding' must be a list of numbers")
    return PageCard(
        page_id=_str(raw, "page_id"),
        title=_str(raw, "title"),
        url=_str(raw, "url"),
        last_edited_time=_str(raw, "last_edited_time"),
        summary=_str(raw, "summary"),
        tags=_str_list(raw, "tags"),
        items=_str_list(raw, "items"),
        items_truncated=truncated,
        parent_id=parent_id,
        child_ids=_str_list(raw, "child_ids"),
        embedding=[float(v) for v in embedding],
    )


class CardStore:
    """In-memory card map backed by a JSON file; reloads when the file changes."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._cards: dict[str, PageCard] = {}
        self._loaded_mtime: float | None = None
        self._load()

    def _load(self) -> None:
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            self._cards = {}
            self._loaded_mtime = None
            return
        self._loaded_mtime = mtime
        self._cards = {}
        try:
            data = json.loads(self._path.read_text())
            raw_cards = data["cards"]
            if not isinstance(raw_cards, list):
                raise TypeError("'cards' must be a list")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logger.warning("Ignoring unreadable catalog file %s: %s", self._path, exc)
            return
        for raw in raw_cards:
            try:
                card = card_from_dict(raw)
            except (ValueError, AttributeError) as exc:
                logger.warning(
                    "Skipping invalid catalog card in %s: %s", self._path, exc
                )
                continue
            self._cards[card.page_id] = card

    def _reload_if_changed(self) -> None:
        """Re-read the file when another process (ingest) rewrote it since our load."""
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            return  # file absent: keep in-memory state
        if mtime != self._loaded_mtime:
            self._load()

    def all(self) -> list[PageCard]:
        self._reload_if_changed()
        return list(self._cards.values())

    def get(self, page_id: str) -> PageCard | None:
        self._reload_if_changed()
        return self._cards.get(page_id)

    def upsert(self, card: PageCard) -> None:
        self._cards[card.page_id] = card

    def delete(self, page_ids: set[str]) -> None:
        for page_id in page_ids:
            self._cards.pop(page_id, None)

    def save(self) -> None:
        """Write atomically: temp file in the same directory, then replace."""
        payload = {
            "version": _VERSION,
            "cards": [card_to_dict(c) for c in self._cards.values()],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(self._path)
        self._loaded_mtime = self._path.stat().st_mtime
