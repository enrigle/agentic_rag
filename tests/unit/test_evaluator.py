"""Unit tests for Evaluator.run: per-query isolation and Langfuse scoring."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_rag.evaluation import evaluator as ev
from agentic_rag.models import ChatMessage, QueryResult, SearchResult


@pytest.mark.asyncio
async def test_run_isolates_queries_and_scores_traces(tmp_path: Path) -> None:
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps([{"id": "q1", "query": "first"}, {"id": "q2", "query": "second"}])
    )
    pipeline = MagicMock()
    pipeline.query = AsyncMock(
        return_value=QueryResult(
            answer="a", sources=[], tool_calls_used=1, latency_ms=1.0, trace_id="t-1"
        )
    )
    answers = iter(["y", "", "n", "bad"])  # rating, note, rating, note

    with (
        patch.object(ev, "QUERIES_FILE", queries),
        patch.object(ev, "RESULTS_FILE", tmp_path / "results.jsonl"),
        patch("builtins.input", lambda _: next(answers)),
        patch.object(ev, "score_trace") as score,
    ):
        evaluator = ev.Evaluator()
        evaluator._pipeline = pipeline
        await evaluator.run()

    # Each eval query gets its own conversation: no history leaks between items.
    threads = [c.kwargs["thread_id"] for c in pipeline.query.await_args_list]
    assert len(set(threads)) == 2
    assert [c.kwargs["value"] for c in score.call_args_list] == [1, 0]
    assert score.call_args_list[1].kwargs["comment"] == "bad"


def _result(
    answer: str = "Yes, ok", titles: list[str] | None = None, route: str = "catalog"
) -> QueryResult:
    sources = [SearchResult("i", t, "u", "c", 1.0) for t in titles or []]
    return QueryResult(answer, sources, 0, 1.0, route=route)


def _item(**kw: object) -> ev.EvalItem:
    raw: dict[str, object] = {"id": "x", "query": "q", **kw}
    return ev.eval_item_from_dict(raw)


def test_check_item_passes_when_all_match() -> None:
    item = _item(
        expected_route="catalog",
        expected_pages=["A"],
        forbidden_pages=["B"],
        expected_answer_start="Yes",
        expected_keywords=["ok"],
    )
    assert ev.check_item(item, _result(titles=["A"])) == []


def test_check_item_wrong_route() -> None:
    assert ev.check_item(_item(expected_route="chunks"), _result())


def test_check_item_missing_expected_page() -> None:
    assert ev.check_item(_item(expected_pages=["A"]), _result(titles=["B"]))


def test_check_item_forbidden_page_present() -> None:
    assert ev.check_item(_item(forbidden_pages=["B"]), _result(titles=["B"]))


def test_check_item_answer_start_case_insensitive() -> None:
    item = _item(expected_answer_start="Yes")
    assert ev.check_item(item, _result(answer="  yes, you have")) == []
    assert ev.check_item(item, _result(answer="No, you do not"))


def test_check_item_missing_keyword() -> None:
    assert ev.check_item(_item(expected_keywords=["zzz"]), _result())


def test_eval_item_from_dict_defaults() -> None:
    item = ev.eval_item_from_dict(
        {"id": "q1", "query": "?", "expected_keywords": ["k"]}
    )
    assert item.history == [] and item.expected_route is None
    assert item.expected_pages == [] and item.expected_keywords == ["k"]


def test_eval_item_from_dict_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        ev.eval_item_from_dict({"id": "", "query": "q"})
    with pytest.raises(ValueError):
        ev.eval_item_from_dict({"id": "a"})
    with pytest.raises(ValueError):
        ev.eval_item_from_dict(
            {"id": "a", "query": "q", "history": [{"role": "bot", "content": "x"}]}
        )
    with pytest.raises(ValueError):
        ev.eval_item_from_dict(
            {"id": "a", "query": "q", "history": [{"role": "user", "content": 1}]}
        )


@pytest.mark.asyncio
async def test_run_auto_returns_1_on_any_failure(tmp_path: Path) -> None:
    queries = tmp_path / "queries.json"
    queries.write_text(
        json.dumps(
            [
                {"id": "ok", "query": "a", "expected_route": "catalog"},
                {
                    "id": "bad",
                    "query": "b",
                    "expected_route": "chunks",
                    "history": [
                        {"role": "user", "content": "u"},
                        {"role": "assistant", "content": "a"},
                    ],
                },
                {"id": "", "query": "broken"},
            ]
        )
    )
    pipeline = MagicMock()
    pipeline.query = AsyncMock(return_value=_result())
    with patch.object(ev, "QUERIES_FILE", queries):
        evaluator = ev.Evaluator()
        evaluator._pipeline = pipeline
        assert await evaluator.run_auto() == 1
    assert pipeline.query.await_count == 2
    assert all(c.kwargs["use_web"] is False for c in pipeline.query.await_args_list)
    pipeline.seed_history.assert_called_once()
    assert pipeline.seed_history.call_args.args[1] == [
        ChatMessage("user", "u"),
        ChatMessage("assistant", "a"),
    ]


@pytest.mark.asyncio
async def test_run_auto_returns_0_when_all_pass(tmp_path: Path) -> None:
    queries = tmp_path / "queries.json"
    queries.write_text(json.dumps([{"id": "ok", "query": "a"}]))
    pipeline = MagicMock()
    pipeline.query = AsyncMock(return_value=_result())
    with patch.object(ev, "QUERIES_FILE", queries):
        evaluator = ev.Evaluator()
        evaluator._pipeline = pipeline
        assert await evaluator.run_auto() == 0


@pytest.mark.asyncio
async def test_run_auto_returns_1_when_no_items(tmp_path: Path) -> None:
    queries = tmp_path / "queries.json"
    queries.write_text("[]")
    with patch.object(ev, "QUERIES_FILE", queries):
        assert await ev.Evaluator().run_auto() == 1
