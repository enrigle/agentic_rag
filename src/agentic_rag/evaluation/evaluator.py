"""Evaluator for the agentic RAG system."""

from __future__ import annotations

import json
import logging
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_rag.config import RAGConfig
from agentic_rag.models import ChatMessage, QueryResult
from agentic_rag.observability.langfuse import score_trace
from agentic_rag.pipeline.coordinator import PipelineCoordinator
from agentic_rag.pipeline.rag_pipeline import create_pipeline

logger = logging.getLogger(__name__)

# evaluator.py -> evaluation/ -> agentic_rag/ -> src/ -> repo_root/
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
EVALS_DIR = _REPO_ROOT / "evals"
QUERIES_FILE = EVALS_DIR / "queries.json"
RESULTS_FILE = EVALS_DIR / "results.jsonl"


def _load_queries() -> list[dict[str, Any]]:
    if not QUERIES_FILE.exists():
        logger.error("Queries file not found: %s", QUERIES_FILE)
        sys.exit(1)
    with QUERIES_FILE.open() as f:
        queries: list[dict[str, Any]] = json.load(f)
    if not queries:
        print("No queries found in queries.json.")
        sys.exit(0)
    return queries


@dataclass
class EvalItem:
    id: str
    query: str
    history: list[ChatMessage]
    expected_route: str | None
    expected_pages: list[str]
    forbidden_pages: list[str]
    expected_answer_start: str | None
    expected_keywords: list[str]


def _parse_history(raw: Any) -> list[ChatMessage]:
    if not isinstance(raw, list):
        raise ValueError("history must be a list")
    messages: list[ChatMessage] = []
    for m in raw:
        if not isinstance(m, dict):
            raise ValueError("history entries must be objects")
        role, content = m.get("role"), m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            raise ValueError(f"bad history entry: {m!r}")
        messages.append(ChatMessage(role, content))
    return messages


def eval_item_from_dict(raw: dict[str, Any]) -> EvalItem:
    """Validate one queries.json entry at the boundary."""
    item_id, query = raw.get("id"), raw.get("query")
    if not isinstance(item_id, str) or not item_id:
        raise ValueError("eval item needs a non-empty string 'id'")
    if not isinstance(query, str) or not query:
        raise ValueError(f"eval item {item_id!r} needs a non-empty string 'query'")
    return EvalItem(
        id=item_id,
        query=query,
        history=_parse_history(raw.get("history", [])),
        expected_route=raw.get("expected_route"),
        expected_pages=list(raw.get("expected_pages", [])),
        forbidden_pages=list(raw.get("forbidden_pages", [])),
        expected_answer_start=raw.get("expected_answer_start"),
        expected_keywords=list(raw.get("expected_keywords", [])),
    )


def check_item(item: EvalItem, result: QueryResult) -> list[str]:
    """Return failure messages for one result; an empty list means pass."""
    failures: list[str] = []
    titles = [s.title for s in result.sources]
    answer = result.answer.strip().lower()
    if item.expected_route is not None and result.route != item.expected_route:
        failures.append(f"route {result.route!r} != {item.expected_route!r}")
    failures += [f"missing page {t!r}" for t in item.expected_pages if t not in titles]
    failures += [
        f"forbidden page {t!r} cited" for t in item.forbidden_pages if t in titles
    ]
    start = item.expected_answer_start
    if start is not None and not answer.startswith(start.lower()):
        failures.append(f"answer does not start with {start!r}")
    failures += [
        f"missing keyword {k!r}"
        for k in item.expected_keywords
        if k.lower() not in answer
    ]
    return failures


def _append_result(record: dict[str, Any]) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _load_results() -> list[dict[str, Any]]:
    if not RESULTS_FILE.exists():
        return []
    results: list[dict[str, Any]] = []
    with RESULTS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


class Evaluator:
    """Interactive evaluator for the agentic RAG pipeline."""

    def __init__(self, config: RAGConfig | None = None) -> None:
        self._config = config
        self._pipeline: PipelineCoordinator | None = None

    def _get_pipeline(self) -> PipelineCoordinator:
        if self._pipeline is None:
            self._pipeline = create_pipeline(self._config)
        return self._pipeline

    async def run(self) -> None:
        """Run interactive evaluation over queries.json, appending rated results."""
        queries = _load_queries()
        print(
            f"Evaluating {len(queries)} queries. Rate each answer: [y]es / [n]o / [s]kip\n"
        )

        for q in queries:
            qid: str = q.get("id", "")
            query: str = q.get("query", "")
            expected: list[str] = q.get("expected_keywords", [])

            if not query:
                continue

            print(f"{'─' * 60}")
            print(f"[{qid}] {query}")
            if expected:
                print(f"Expected keywords: {', '.join(expected)}")
            print("Running query...")

            # Fresh thread per item: shared history would leak earlier answers in.
            result = await self._get_pipeline().query(
                query, thread_id=f"eval-{qid}-{uuid.uuid4().hex[:8]}"
            )
            answer: str = result.answer
            sources = result.sources

            print(f"\nAnswer:\n{answer}")
            if sources:
                print("\nSources:")
                for i, s in enumerate(sources, start=1):
                    print(f"  [{i}] {s.title} — {s.source}")

            while True:
                rating_input = input("\nRate [y/n/s]: ").strip().lower()
                if rating_input in ("y", "n", "s"):
                    break
                print("Please enter y, n, or s.")

            if rating_input == "s":
                print("Skipped.\n")
                continue

            note = input("Note (optional, press Enter to skip): ").strip()

            record: dict[str, Any] = {
                "id": qid,
                "query": query,
                "answer": answer,
                "sources": [{"title": s.title, "url": s.source} for s in sources],
                "rating": rating_input,
                "note": note,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            _append_result(record)
            if result.trace_id:
                score_trace(
                    trace_id=result.trace_id,
                    name="human_rating",
                    value=1 if rating_input == "y" else 0,
                    comment=note or None,
                )
            print(f"Saved {'✓' if rating_input == 'y' else '✗'}\n")

        print("Done.")

    async def run_auto(self) -> int:
        """Run every item against its checks. Return 0 if all pass, else 1."""
        failed = 0
        raw_items = _load_queries()
        for index, raw in enumerate(raw_items):
            try:
                item = eval_item_from_dict(raw)
            except ValueError as exc:
                print(f"FAIL #{index} — invalid item: {exc}")
                failed += 1
                continue
            pipeline = self._get_pipeline()
            thread_id = f"eval-{item.id}-{uuid.uuid4().hex[:8]}"
            if item.history:
                pipeline.seed_history(thread_id, item.history)
            result = await pipeline.query(
                item.query, thread_id=thread_id, use_web=False
            )
            failures = check_item(item, result)
            if failures:
                failed += 1
                print(f"FAIL {item.id} — {'; '.join(failures)}")
            else:
                print(f"PASS {item.id}")
        print(f"{len(raw_items) - failed}/{len(raw_items)} passed")
        return 1 if failed else 0

    def report(self) -> None:
        """Print a summary table from saved results.jsonl."""
        results = _load_results()
        if not results:
            print("No results found. Run without --report first.")
            return

        by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in results:
            by_id[r["id"]].append(r)

        print(f"{'─' * 60}")
        print(f"{'Query ID':<12} {'Pass rate':>10}  {'Last 3 ratings'}")
        print(f"{'─' * 60}")

        total_yes = 0
        total_rated = 0

        for qid, records in sorted(by_id.items()):
            rated = [r for r in records if r["rating"] in ("y", "n")]
            yes = sum(1 for r in rated if r["rating"] == "y")
            pass_rate = f"{yes}/{len(rated)}" if rated else "—"
            last3 = " ".join(r["rating"] for r in rated[-3:]) if rated else "—"
            print(f"{qid:<12} {pass_rate:>10}  {last3}")
            total_yes += yes
            total_rated += len(rated)

        print(f"{'─' * 60}")
        overall = f"{total_yes}/{total_rated}" if total_rated else "—"
        print(f"{'Overall':<12} {overall:>10}")
