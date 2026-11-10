"""NotionIngester builds per-page catalog cards during ingest()."""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_rag.catalog.generate import CardFields
from agentic_rag.catalog.store import CardStore, PageCard
from agentic_rag.config import CatalogConfig
from agentic_rag.ingestion.chunker import TextBlock
from agentic_rag.ingestion.notion import INDEX_DOC_FORMAT, NotionIngester

CARD_JSON = json.dumps({"summary": "About cats.", "tags": ["pets"], "items": ["tabby"]})


def _page(
    page_id: str = "p1", edited: str = "t1", parent: Any = None
) -> dict[str, Any]:
    return {
        "id": page_id,
        "url": f"https://notion.so/{page_id}",
        "last_edited_time": edited,
        "parent": parent or {"type": "workspace", "workspace": True},
        "properties": {"title": {"type": "title", "title": [{"plain_text": "Cats"}]}},
    }


def _card(page_id: str = "p1", edited: str = "t1") -> PageCard:
    return PageCard(
        page_id, "Cats", "u", edited, "old", ["pets"], [], False, None, [], [0.1]
    )


def _make(
    tmp_path: Path, chat: AsyncMock | None, store: CardStore | None
) -> NotionIngester:
    with patch("agentic_rag.ingestion.notion.chromadb.PersistentClient"):
        ing = NotionIngester(
            MagicMock(), AsyncMock(embed=AsyncMock(return_value=[0.5])), chat, store
        )
    ing._config.ingestion.vision_model = ""
    ing._config.ingestion.chunk_size = 1000
    ing._config.ingestion.chunk_overlap = 100
    ing._config.catalog = CatalogConfig()
    ing._collection = MagicMock()
    ing._rebuild_bm25 = MagicMock()  # type: ignore[method-assign]
    return ing


def coll(ing: NotionIngester) -> Any:
    return ing._collection


def _chunks_current(ing: NotionIngester, edited: str = "t1") -> None:
    """Collection already holds up-to-date chunks for the page."""
    coll(ing).get.return_value = {
        "ids": ["p1_chunk_0"],
        "metadatas": [
            {
                "page_id": "p1",
                "last_edited_time": edited,
                "doc_format": INDEX_DOC_FORMAT,
            }
        ],
    }


def _chat(reply: str = CARD_JSON) -> AsyncMock:
    return AsyncMock(chat=AsyncMock(return_value=reply))


def _fake_fetch(blocks: list[TextBlock], child_ids: list[str]) -> Any:
    return functools.partial(_fetch_stub, blocks, child_ids)


async def _fetch_stub(
    fixed_blocks: list[TextBlock], fixed_ids: list[str], *args: Any, **kwargs: Any
) -> list[TextBlock]:
    if kwargs.get("child_ids") is not None:
        kwargs["child_ids"].extend(fixed_ids)
    return fixed_blocks


async def _run(
    ing: NotionIngester,
    pages: list[dict[str, Any]],
    blocks: list[TextBlock],
    child_ids: list[str] | None = None,
    sleep: AsyncMock | None = None,
) -> AsyncMock:
    fetch = AsyncMock(side_effect=_fake_fetch(blocks, child_ids or []))
    with (
        patch.dict("os.environ", {"NOTION_TOKEN": "x"}),
        patch("agentic_rag.ingestion.notion.AsyncClient"),
        patch(
            "agentic_rag.ingestion.notion.async_collect_paginated_api",
            AsyncMock(return_value=pages),
        ),
        patch.object(NotionIngester, "_fetch_text_from_blocks", fetch),
        patch("agentic_rag.catalog.generate.asyncio.sleep", AsyncMock()),
        patch("agentic_rag.ingestion.notion.asyncio.sleep", sleep or AsyncMock()),
    ):
        await ing.ingest()
    return fetch


async def test_card_built_for_page_with_current_chunks_but_no_card(
    tmp_path: Path,
) -> None:
    store = CardStore(tmp_path / "catalog.json")
    chat = _chat(
        '{"summary": "About cats.", "tags": [], "items": ["tabby"]}'
    )  # no tags: no merge call
    ing = _make(tmp_path, chat, store)
    _chunks_current(ing)
    fetch = await _run(ing, [_page()], [TextBlock("paragraph", "tabby cats")])
    assert chat.chat.await_count == 1
    assert fetch.await_count == 1
    coll(ing).upsert.assert_not_called()
    saved = CardStore(tmp_path / "catalog.json").get("p1")
    assert saved is not None and saved.summary == "About cats."
    assert saved.items == ["tabby"] and saved.embedding == [0.5]


async def test_up_to_date_page_and_card_is_skipped(tmp_path: Path) -> None:
    store = CardStore(tmp_path / "catalog.json")
    store.upsert(_card())
    chat = _chat()
    ing = _make(tmp_path, chat, store)
    _chunks_current(ing)
    fetch = await _run(ing, [_page()], [TextBlock("paragraph", "x")])
    assert fetch.await_count == 0 and chat.chat.await_count == 0


async def test_card_failure_keeps_old_card_and_counts(tmp_path: Path) -> None:
    store = CardStore(tmp_path / "catalog.json")
    store.upsert(_card(edited="t0"))
    chat = AsyncMock(chat=AsyncMock(side_effect=RuntimeError("down")))
    ing = _make(tmp_path, chat, store)
    _chunks_current(ing, "t1")
    await _run(ing, [_page(edited="t1")], [TextBlock("paragraph", "x")])
    assert ing.card_failures == 1
    kept = store.get("p1")
    assert kept is not None and kept.summary == "old" and kept.last_edited_time == "t0"


async def test_stale_page_card_is_deleted(tmp_path: Path) -> None:
    store = CardStore(tmp_path / "catalog.json")
    store.upsert(_card("gone"))
    ing = _make(tmp_path, _chat(), store)
    coll(ing).get.return_value = {
        "ids": [],
        "metadatas": [{"page_id": "gone"}],
    }
    await _run(ing, [], [])
    assert store.get("gone") is None
    assert CardStore(tmp_path / "catalog.json").all() == []


async def test_page_without_text_gets_no_card(tmp_path: Path) -> None:
    store = CardStore(tmp_path / "catalog.json")
    chat = _chat()
    ing = _make(tmp_path, chat, store)
    _chunks_current(ing)
    await _run(ing, [_page()], [])
    assert chat.chat.await_count == 0 and store.all() == []


async def test_child_page_ids_and_parent_recorded(tmp_path: Path) -> None:
    store = CardStore(tmp_path / "catalog.json")
    ing = _make(tmp_path, _chat(), store)
    _chunks_current(ing)
    page = _page(parent={"type": "page_id", "page_id": "root"})
    await _run(ing, [page], [TextBlock("paragraph", "x")], child_ids=["c1", "c2"])
    card = store.get("p1")
    assert card is not None
    assert card.parent_id == "root" and card.child_ids == ["c1", "c2"]


async def test_tags_merged_when_raw_tags_changed(tmp_path: Path) -> None:
    store = CardStore(tmp_path / "catalog.json")
    chat = AsyncMock(chat=AsyncMock(side_effect=[CARD_JSON, '{"pets": "animals"}']))
    ing = _make(tmp_path, chat, store)
    _chunks_current(ing)
    await _run(ing, [_page()], [TextBlock("paragraph", "x")])
    card = store.get("p1")
    assert card is not None and card.tags == ["animals"]
    assert chat.chat.await_count == 2


async def test_merge_failure_keeps_unmerged_tags(tmp_path: Path) -> None:
    store = CardStore(tmp_path / "catalog.json")
    chat = AsyncMock(chat=AsyncMock(side_effect=[CARD_JSON, "not json"]))
    ing = _make(tmp_path, chat, store)
    _chunks_current(ing)
    await _run(ing, [_page()], [TextBlock("paragraph", "x")])
    card = store.get("p1")
    assert card is not None and card.tags == ["pets"]


async def test_catalog_skipped_when_no_chat_llm(tmp_path: Path) -> None:
    ing = _make(tmp_path, None, None)
    _chunks_current(ing)
    fetch = await _run(ing, [_page()], [TextBlock("paragraph", "x")])
    assert fetch.await_count == 0
    assert not (tmp_path / "catalog.json").exists()


def test_two_arg_constructor_still_works() -> None:
    with patch("agentic_rag.ingestion.notion.chromadb.PersistentClient"):
        ing = NotionIngester(MagicMock(), MagicMock())
    assert ing.card_failures == 0


def _pages(n: int) -> list[dict[str, Any]]:
    return [_page(f"p{i}") for i in range(n)]


async def test_delay_between_card_calls_not_before_first(tmp_path: Path) -> None:
    store = CardStore(tmp_path / "catalog.json")
    ing = _make(tmp_path, _chat('{"summary": "s", "tags": [], "items": []}'), store)
    coll(ing).get.return_value = {"ids": [], "metadatas": []}
    sleep = AsyncMock()
    await _run(ing, _pages(3), [TextBlock("paragraph", "x")], sleep=sleep)
    assert [c.args for c in sleep.await_args_list] == [(30.0,), (30.0,)]


async def test_stops_cards_after_consecutive_failures(tmp_path: Path) -> None:
    store = CardStore(tmp_path / "catalog.json")
    chat = AsyncMock(chat=AsyncMock(side_effect=RuntimeError("429")))
    ing = _make(tmp_path, chat, store)
    coll(ing).get.return_value = {"ids": [], "metadatas": []}
    await _run(ing, _pages(5), [TextBlock("paragraph", "x")])
    assert ing.card_failures == 3
    assert chat.chat.await_count == 9  # 3 pages x 3 internal retries; pages 4-5 skipped
    assert coll(ing).upsert.call_count == 5  # chunks still upserted
    assert store.all() == []


async def test_success_resets_consecutive_failures(tmp_path: Path) -> None:
    store = CardStore(tmp_path / "catalog.json")
    ok = CardFields("s", [], [], False)
    err = RuntimeError("429")
    chat = AsyncMock(chat=AsyncMock())
    ing = _make(tmp_path, chat, store)
    coll(ing).get.return_value = {"ids": [], "metadatas": []}
    gen = AsyncMock(side_effect=[err, err, ok, err, err, ok])
    with patch("agentic_rag.ingestion.notion.generate_card_fields", gen):
        await _run(ing, _pages(6), [TextBlock("paragraph", "x")])
    assert gen.await_count == 6
    assert ing.card_failures == 4
    assert len(store.all()) == 2
