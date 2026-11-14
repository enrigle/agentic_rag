"""Tests for PageCard serialization and CardStore persistence."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from agentic_rag.catalog.store import CardStore, PageCard, card_from_dict, card_to_dict


def make_card(page_id: str = "p1", title: str = "Title") -> PageCard:
    return PageCard(
        page_id=page_id,
        title=title,
        url=f"https://notion.so/{page_id}",
        last_edited_time="2026-01-01T00:00:00Z",
        summary="A summary",
        tags=["a", "b"],
        items=["x", "y"],
        items_truncated=False,
        parent_id=None,
        child_ids=["c1"],
        embedding=[0.1, 0.2],
    )


def test_roundtrip_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    store = CardStore(path)
    cards = [make_card("p1"), make_card("p2", "Other")]
    for card in cards:
        store.upsert(card)
    store.save()
    assert CardStore(path).all() == cards


def test_missing_file_is_empty(tmp_path: Path) -> None:
    assert CardStore(tmp_path / "none.json").all() == []


def test_corrupt_file_is_empty_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "catalog.json"
    path.write_text("{not json")
    with caplog.at_level(logging.WARNING):
        assert CardStore(path).all() == []
    assert "catalog" in caplog.text


def test_invalid_card_is_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "catalog.json"
    bad = card_to_dict(make_card("bad"))
    del bad["title"]
    good = make_card("good")
    store = CardStore(path)
    store.upsert(good)
    store.save()
    raw = json.loads(path.read_text())
    raw["cards"].append(bad)
    path.write_text(json.dumps(raw))
    with caplog.at_level(logging.WARNING):
        assert CardStore(path).all() == [good]
    assert "catalog" in caplog.text


def test_card_from_dict_rejects_mistyped_field() -> None:
    raw = card_to_dict(make_card())
    raw["tags"] = "not-a-list"
    with pytest.raises(ValueError):
        card_from_dict(raw)


def test_save_leaves_no_temp_files(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    store = CardStore(path)
    store.upsert(make_card())
    store.save()
    assert list(tmp_path.iterdir()) == [path]


def test_reload_when_file_changes(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    seed = CardStore(path)
    seed.upsert(make_card("p1"))
    seed.save()
    reader = CardStore(path)
    assert [c.page_id for c in reader.all()] == ["p1"]

    writer = CardStore(path)
    writer.upsert(make_card("p2"))
    writer.save()
    st = path.stat()
    os.utime(path, (st.st_atime, st.st_mtime + 10))
    assert {c.page_id for c in reader.all()} == {"p1", "p2"}


def test_get_and_upsert_replaces(tmp_path: Path) -> None:
    store = CardStore(tmp_path / "catalog.json")
    store.upsert(make_card("p1", "Old"))
    store.upsert(make_card("p1", "New"))
    assert len(store.all()) == 1
    card = store.get("p1")
    assert card is not None and card.title == "New"
    assert store.get("zzz") is None


def test_delete_removes_cards(tmp_path: Path) -> None:
    store = CardStore(tmp_path / "catalog.json")
    for pid in ("p1", "p2", "p3"):
        store.upsert(make_card(pid))
    store.delete({"p1", "p3", "missing"})
    assert [c.page_id for c in store.all()] == ["p2"]
