"""Unit tests for Evaluator.run: per-query isolation and Langfuse scoring."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_rag.evaluation import evaluator as ev
from agentic_rag.models import QueryResult


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
