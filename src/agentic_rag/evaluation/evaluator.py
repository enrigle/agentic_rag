"""Evaluator for the agentic RAG system."""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_rag.config import RAGConfig
from agentic_rag.pipeline.rag_pipeline import RAGPipeline, create_pipeline

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
        self._pipeline: RAGPipeline = create_pipeline(config)

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

            result = await self._pipeline.query(query)
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
            print(f"Saved {'✓' if rating_input == 'y' else '✗'}\n")

        print("Done.")

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
