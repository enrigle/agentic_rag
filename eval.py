"""Eval script for the agentic RAG system.

Usage:
    uv run python eval.py           # interactive rating
    uv run python eval.py --report  # summary from saved results
"""

import argparse
import asyncio
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from main import AgenticRAGSystem
from agentic_rag.observability.langfuse import score_trace

logger = logging.getLogger(__name__)

EVALS_DIR = Path(__file__).parent / "evals"
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


def _append_result(record: dict[str, Any]) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _load_results() -> list[dict[str, Any]]:
    if not RESULTS_FILE.exists():
        return []
    results = []
    with RESULTS_FILE.open() as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


async def run_eval() -> None:
    queries = _load_queries()
    system = AgenticRAGSystem()
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

        result = await system.query(
            query,
            trace_tags=["eval"],
            trace_metadata={"eval_query_id": qid},
        )
        answer: str = result.get("answer", "")
        sources: list[dict[str, Any]] = result.get("sources", [])
        trace_id: str = result.get("trace_id", "")

        print(f"\nAnswer:\n{answer}")
        if sources:
            print("\nSources:")
            for s in sources:
                print(f"  [{s['index']}] {s['title']} — {s['url']}")

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
            "sources": sources,
            "rating": rating_input,
            "note": note,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        _append_result(record)

        if rating_input in ("y", "n") and trace_id:
            score_trace(
                trace_id=trace_id,
                name="human_rating",
                value=1 if rating_input == "y" else 0,
                comment=note or None,
            )
        print(f"Saved {'✓' if rating_input == 'y' else '✗'}\n")

    print("Done.")


def run_report() -> None:
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Eval the agentic RAG system")
    parser.add_argument(
        "--report", action="store_true", help="Print summary from saved results"
    )
    args = parser.parse_args()

    if args.report:
        run_report()
    else:
        asyncio.run(run_eval())
