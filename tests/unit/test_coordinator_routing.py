from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from agentic_rag.catalog.store import CardStore, PageCard
from agentic_rag.config import CatalogConfig
from agentic_rag.models import QueryResult, SearchResult
from agentic_rag.pipeline.coordinator import PipelineCoordinator, page_ids_from_sources
from agentic_rag.pipeline.memory import ConversationMemory
from agentic_rag.pipeline.router import RouteDecision
from agentic_rag.pipeline.synthesizer import Synthesizer
from agentic_rag.retrieval.reranker import CrossEncoderReranker

NO_INFO = "No relevant information found for your query."


class FakeRouter:
    def __init__(self, *decisions: RouteDecision) -> None:
        self._decisions = list(decisions)
        self.calls: list[tuple[str, list[tuple[str, str]]]] = []

    async def route(self, query, history, previous_pages):  # type: ignore[no-untyped-def]
        self.calls.append((query, previous_pages))
        return (
            self._decisions.pop(0) if len(self._decisions) > 1 else self._decisions[0]
        )


def _card(pid: str, title: str) -> PageCard:
    return PageCard(
        pid,
        title,
        f"https://n/{pid}",
        "t",
        f"sum {title}",
        [],
        [],
        False,
        None,
        [],
        [1.0, 0.0],
    )


def _sr(id_: str, title: str = "T") -> SearchResult:
    return SearchResult(id_, title, "u", "content", 1.0)


def _source(name: str, results: list[SearchResult]) -> MagicMock:
    s = MagicMock()
    s.name = name
    s.gated = name != "web"
    s.search = AsyncMock(return_value=results)
    return s


def _build(
    tmp_path: Path,
    decisions: list[RouteDecision] | None,
    sources: list[MagicMock] | None = None,
    cards: list[PageCard] | None = None,
    answer: str = "answer [1]",
    cache: MagicMock | None = None,
) -> tuple[PipelineCoordinator, MagicMock, MagicMock, MagicMock]:
    store = CardStore(tmp_path / "catalog.json")
    for c in cards or []:
        store.upsert(c)
    store.save()
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=answer)
    reranker = MagicMock(spec=CrossEncoderReranker)
    reranker.rerank = MagicMock(side_effect=lambda q, c, min_score=None: c)
    embed = MagicMock()
    embed.embed = AsyncMock(return_value=[1.0, 0.0])
    fetcher = MagicMock()
    fetcher.fetch_by_page_ids = AsyncMock(return_value=[_sr("p1_chunk_0")])
    coord = PipelineCoordinator(
        sources=cast(Any, sources or []),
        reranker=reranker,
        synthesizer=Synthesizer(llm),
        memory=ConversationMemory(),
        max_tool_calls=5,
        embed_llm=embed,
        cache=cache,
        router=cast(Any, FakeRouter(*decisions)) if decisions else None,
        card_store=store,
        page_fetcher=fetcher,
        catalog_config=CatalogConfig(),
    )
    return coord, llm, fetcher, embed


def _d(route: str, q: str = "standalone", focus: bool = False) -> RouteDecision:
    return RouteDecision(q, route, focus)  # type: ignore[arg-type]


async def test_catalog_route_returns_cited_cards_and_note(tmp_path: Path) -> None:
    coord, _, _, _ = _build(
        tmp_path, [_d("catalog")], cards=[_card("a", "A"), _card("b", "B")]
    )
    r = await coord.query("list")
    assert r.route == "catalog"
    assert [s.id for s in r.sources] == ["a"]
    assert r.answer.endswith("Found 1 pages; this list may be incomplete.")


async def test_catalog_route_without_catalog_file(tmp_path: Path) -> None:
    rag = _source("rag", [_sr("p1_chunk_0")])
    coord, _, _, _ = _build(tmp_path, [_d("catalog", "the standalone")], [rag])
    r = await coord.query("list")
    assert r.answer == "answer [1]"
    assert rag.search.call_args.args[0] == "the standalone"
    assert r.route == "catalog"


async def test_catalog_fallback_never_uses_web(tmp_path: Path) -> None:
    rag = _source("rag", [])
    web = _source("web", [_sr("w")])
    coord, _, _, _ = _build(tmp_path, [_d("catalog")], [rag, web])
    r = await coord.query("list", use_web=True)
    web.search.assert_not_called()
    assert r.answer == NO_INFO
    assert r.route == "catalog"


async def test_cache_hit_keeps_route_and_sets_last_pages(tmp_path: Path) -> None:
    cached = QueryResult(
        answer="cached",
        sources=[_sr("p1_chunk_0", "Page One")],
        tool_calls_used=0,
        latency_ms=1.0,
        route="chunks",
    )
    cache = MagicMock()
    cache.get = AsyncMock(return_value=cached)
    cache.set = AsyncMock()
    coord, _, fetcher, _ = _build(
        tmp_path,
        [_d("catalog"), _d("chunks", focus=True)],
        cache=cache,
    )
    r = await coord.query("list")
    assert r.route == "catalog"
    assert r.answer == "cached"
    cache.get = AsyncMock(return_value=None)
    await coord.query("summarize it")
    fetcher.fetch_by_page_ids.assert_awaited_once_with(["p1"], 20)


async def test_catalog_route_never_calls_web_even_if_use_web(tmp_path: Path) -> None:
    web = _source("web", [_sr("w")])
    coord, _, _, _ = _build(tmp_path, [_d("catalog")], [web], cards=[_card("a", "A")])
    await coord.query("list", use_web=True)
    web.search.assert_not_called()


async def test_chunks_route_searches_with_standalone_query(tmp_path: Path) -> None:
    rag = _source("rag", [_sr("p1_chunk_0")])
    coord, _, _, embed = _build(tmp_path, [_d("chunks", "the standalone")], [rag])
    r = await coord.query("it?")
    assert rag.search.call_args.args[0] == "the standalone"
    embed.embed.assert_awaited_with("the standalone")
    assert r.route == "chunks"


async def test_web_source_skipped_when_use_web_false(tmp_path: Path) -> None:
    rag = _source("rag", [])
    web = _source("web", [_sr("w")])
    coord, _, _, _ = _build(tmp_path, [_d("chunks")], [rag, web])
    r = await coord.query("q")
    web.search.assert_not_called()
    assert r.answer == NO_INFO


async def test_web_fallback_runs_when_use_web_true_and_kb_empty(tmp_path: Path) -> None:
    rag = _source("rag", [])
    web = _source("web", [_sr("w")])
    coord, _, _, _ = _build(tmp_path, [_d("chunks")], [rag, web])
    r = await coord.query("q", use_web=True)
    web.search.assert_awaited_once()
    assert [s.id for s in r.sources] == ["w"]


async def test_cache_bypassed_when_use_web_true(tmp_path: Path) -> None:
    cache = MagicMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    rag = _source("rag", [_sr("p1_chunk_0")])
    coord, _, _, _ = _build(tmp_path, [_d("chunks")], [rag], cache=cache)
    await coord.query("q", use_web=True)
    cache.get.assert_not_called()
    cache.set.assert_not_called()


async def test_focus_previous_fetches_previous_pages(tmp_path: Path) -> None:
    rag = _source("rag", [_sr("p1_chunk_0", "Page One")])
    coord, _, fetcher, _ = _build(
        tmp_path, [_d("chunks"), _d("chunks", focus=True)], [rag]
    )
    await coord.query("first")
    r = await coord.query("summarize it")
    fetcher.fetch_by_page_ids.assert_awaited_once_with(["p1"], 20)
    assert rag.search.await_count == 1
    assert r.sources[0].id == "p1_chunk_0"


async def test_focus_previous_without_previous_pages_uses_chunks(
    tmp_path: Path,
) -> None:
    rag = _source("rag", [_sr("p1_chunk_0")])
    coord, _, fetcher, _ = _build(tmp_path, [_d("chunks", focus=True)], [rag])
    r = await coord.query("summarize it")
    fetcher.fetch_by_page_ids.assert_not_called()
    assert r.answer == "answer [1]"


async def test_both_route_merges_cards_and_chunks(tmp_path: Path) -> None:
    rag = _source("rag", [_sr("p9_chunk_0", "Chunky")])
    coord, llm, _, _ = _build(tmp_path, [_d("both")], [rag], cards=[_card("a", "A")])
    r = await coord.query("what do I know")
    assert [s.id for s in r.sources] == ["a", "p9_chunk_0"]
    assert "Chunky" in llm.chat.call_args.args[0]
    assert r.route == "both"


async def test_no_router_keeps_old_behaviour(tmp_path: Path) -> None:
    rag = _source("rag", [])
    web = _source("web", [_sr("w")])
    coord, _, _, _ = _build(tmp_path, None, [rag, web])
    r = await coord.query("q")
    web.search.assert_awaited_once()
    assert r.route == "chunks"


def test_page_ids_from_sources_strips_chunk_suffix() -> None:
    srcs = [
        SearchResult("p1_chunk_0", "A", "u", "c", 1.0),
        SearchResult("p1_chunk_3", "A", "u", "c", 1.0),
        SearchResult("p2", "B", "u", "c", 1.0),
    ]
    assert page_ids_from_sources(srcs) == ["p1", "p2"]
