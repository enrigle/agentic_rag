from agentic_rag.catalog.select import (
    INCOMPLETE_NOTE,
    build_catalog_prompt,
    cards_to_sources,
    cited_card_indices,
    ensure_incomplete_note,
    estimate_tokens,
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


def _est(cards: list[PageCard]) -> int:
    return estimate_tokens("\n".join(format_card(i, c) for i, c in enumerate(cards, 1)))


def _ranked_cards() -> list[PageCard]:
    return [
        _card("far", [0.0, 1.0]),
        _card("none"),
        _card("near", [1.0, 0.1]),
        _card("mid", [1.0, 1.0]),
    ]


def test_estimate_tokens_is_len_over_three() -> None:
    assert estimate_tokens("abcdef") == 2


def test_select_cards_fills_budget_in_cosine_order_when_over_budget() -> None:
    cards = _ranked_cards()
    budget = _est([cards[2], cards[3]])  # fits exactly two, not three
    assert _est(cards[:3]) > budget
    got = select_cards(cards, [1.0, 0.0], budget, 10)
    assert [c.page_id for c in got] == ["near", "mid"]


def test_select_cards_over_budget_never_exceeds_budget() -> None:
    cards = [_card(str(i), [1.0, i / 10]) for i in range(20)]
    budget = _est(cards) // 2
    got = select_cards(cards, [1.0, 0.0], budget, 100)
    assert 1 < len(got) < len(cards)
    assert _est(got) <= budget


def test_select_cards_single_oversized_card_still_returned() -> None:
    cards = [_card("big", [1.0, 0.0], items=["x" * 500]), _card("b", [0.0, 1.0])]
    got = select_cards(cards, [1.0, 0.0], 5, 10)
    assert [c.page_id for c in got] == ["big"]


def test_select_cards_top_n_caps_count_when_budget_allows_more() -> None:
    cards = _ranked_cards()
    got = select_cards(cards, [1.0, 0.0], _est(cards) - 1, 2)
    assert [c.page_id for c in got] == ["near", "mid"]


def test_select_cards_without_query_vec_takes_first_that_fit() -> None:
    cards = [_card("1"), _card("2"), _card("3")]
    budget = _est(cards[:2])
    assert select_cards(cards, None, budget, 10) == cards[:2]
    assert select_cards(cards, None, budget, 1) == cards[:1]


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


def test_extract_citations_full_width_brackets() -> None:
    assert extract_citations("A \u30101\u3011", 3) == [1]
    assert extract_citations("A \u30101, 2\u3011 \u30102\u20133\u3011", 3) == [1, 2, 3]


def _titled(pid: str, title: str) -> PageCard:
    c = _card(pid)
    c.title = title
    return c


def test_cited_card_indices_numeric_wins_over_title() -> None:
    cards = [_titled("a", "Alpha"), _titled("b", "Beta")]
    assert cited_card_indices("Beta page, see [1]", cards) == [1]


def test_cited_card_indices_falls_back_to_url() -> None:
    cards = [_card("a"), _card("b")]
    assert cited_card_indices("see [x](https://n/b)", cards) == [2]


def test_cited_card_indices_falls_back_to_title_case_insensitive() -> None:
    cards = [_titled("a", "Vectorless RAG"), _titled("b", "Other")]
    assert cited_card_indices("about vectorless rag here", cards) == [1]


def test_cited_card_indices_ignores_short_title() -> None:
    cards = [_titled("a", "RAG")]
    assert cited_card_indices("about rag here", cards) == []


def _with_items(pid: str, items: list[str]) -> PageCard:
    c = _titled(pid, f"Page {pid}")
    c.items = items
    return c


def test_cited_card_indices_falls_back_to_best_item_match() -> None:
    # The model listed the items but forgot "[1]": still cite the page they came from.
    cards = [
        _with_items("a", ["/plan", "/compact", "/rewind", "/help"]),
        _with_items("b", ["/compact", "/other"]),
    ]
    answer = "Your commands: /plan, /compact, /rewind and /help."
    assert cited_card_indices(answer, cards) == [1]


def test_cited_card_indices_needs_three_item_matches() -> None:
    cards = [_with_items("a", ["/plan", "/compact", "/rewind"])]
    assert cited_card_indices("Use /plan or /compact.", cards) == []


def test_cited_card_indices_ignores_short_items() -> None:
    cards = [_with_items("a", ["ai", "ml", "go", "/plan"])]
    assert cited_card_indices("ai ml go /plan", cards) == []
