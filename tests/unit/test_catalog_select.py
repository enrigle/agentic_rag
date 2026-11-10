from agentic_rag.catalog.select import (
    INCOMPLETE_NOTE,
    build_catalog_prompt,
    cards_to_sources,
    ensure_incomplete_note,
    extract_citations,
    format_card,
    select_cards,
)
from agentic_rag.catalog.store import PageCard
from agentic_rag.models import ChatMessage, SearchResult


def _card(
    pid: str,
    emb: list[float] | None = None,
    items: list[str] | None = None,
    truncated: bool = False,
    parent_id: str | None = None,
) -> PageCard:
    return PageCard(
        page_id=pid,
        title=f"T{pid}",
        url=f"https://n/{pid}",
        last_edited_time="2026-01-01",
        summary=f"sum {pid}",
        tags=["a", "b"],
        items=items or [],
        items_truncated=truncated,
        parent_id=parent_id,
        child_ids=[],
        embedding=emb or [],
    )


def test_format_card_marks_truncated_items() -> None:
    text = format_card(1, _card("x", items=["i1", "i2"], truncated=True))
    assert text.startswith("[1] Tx | a, b | sum x | https://n/x")
    assert text.endswith("items: i1, i2 (more on page)")
    assert "items:" not in format_card(2, _card("y"))


def test_format_card_adds_parent_hint() -> None:
    assert "parent: Tp" in format_card(1, _card("c"), parent_title="Tp")


def test_select_cards_returns_all_when_under_budget() -> None:
    cards = [_card("1"), _card("2")]
    assert select_cards(cards, [1.0], 10_000, 1) == cards


def test_select_cards_prefilters_by_similarity_when_over_budget() -> None:
    cards = [
        _card("far", [0.0, 1.0]),
        _card("none"),
        _card("near", [1.0, 0.1]),
        _card("mid", [1.0, 1.0]),
    ]
    got = select_cards(cards, [1.0, 0.0], 1, 2)
    assert [c.page_id for c in got] == ["near", "mid"]


def test_select_cards_without_query_vec_takes_first_n() -> None:
    cards = [_card("1"), _card("2"), _card("3")]
    assert select_cards(cards, None, 1, 2) == cards[:2]


def test_extract_citations_ignores_out_of_range() -> None:
    assert extract_citations("See [0], [2], [99], [2] and [1].", n_cards=3) == [2, 1]
    assert extract_citations("no cites", n_cards=3) == []


def test_extract_citations_groups_and_ranges() -> None:
    assert extract_citations("Foo [1, 2], Bar [3-4]", 4) == [1, 2, 3, 4]
    assert extract_citations("A [1,2]", 4) == [1, 2]
    assert extract_citations("A [3\u20134]", 4) == [3, 4]
    assert extract_citations("A [4] [1][2] [2, 3]", 4) == [4, 1, 2, 3]
    assert extract_citations("A [1, 9] [3-6]", 4) == [1, 3, 4]


def test_cards_to_sources_maps_fields() -> None:
    cards = [_card("1"), _card("2")]
    assert cards_to_sources(cards, [2]) == [
        SearchResult(
            id="2", title="T2", source="https://n/2", content="sum 2", score=1.0
        )
    ]


def test_ensure_incomplete_note_appends_once() -> None:
    out = ensure_incomplete_note("Pages: [1]", 1)
    assert out.endswith("Found 1 pages; this list may be incomplete.")
    assert INCOMPLETE_NOTE.format(n=1) in out
    assert ensure_incomplete_note(out, 1) == out
    assert ensure_incomplete_note("No pages.", 0) == "No pages."


def test_build_catalog_prompt_labels_passages_only_when_present() -> None:
    cards = [_card("1"), _card("2", parent_id="1")]
    hist = [ChatMessage("user", "hello")]
    p = SearchResult("c", "t", "s", "passage body", 1.0)
    without = build_catalog_prompt("q?", [], cards, [])
    assert "Matching pages" in without and "Relevant passages" not in without
    assert "[1] T1" in without and "parent: T1" in without
    assert "user: hello" not in without and "q?" in without
    with_p = build_catalog_prompt("q?", hist, cards, [p])
    assert "Relevant passages" in with_p and "passage body" in with_p
    assert "user: hello" in with_p


def test_extract_citations_clamps_huge_range() -> None:
    assert extract_citations("see [1-10000000]", 3) == [1, 2, 3]
