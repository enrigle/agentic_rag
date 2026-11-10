"""Interactive evaluation CLI.

Usage:
    uv run python scripts/eval.py           # rate answers to evals/queries.json
    uv run python scripts/eval.py --report  # summary from evals/results.jsonl
    uv run python scripts/eval.py --auto    # automated checks; exit 0 only if all pass

Thin wrapper around Evaluator so ratings, result storage, and Langfuse scoring
share one implementation with the library pipeline.
"""

import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv

load_dotenv()  # loads .env from cwd (or any parent dir)

from agentic_rag.evaluation.evaluator import Evaluator


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Eval the agentic RAG system")
    parser.add_argument(
        "--report", action="store_true", help="Print summary from saved results"
    )
    parser.add_argument(
        "--auto", action="store_true", help="Run automated checks; exit 1 on any fail"
    )
    args = parser.parse_args()

    evaluator = Evaluator()
    if args.auto:
        sys.exit(asyncio.run(evaluator.run_auto()))
    elif args.report:
        evaluator.report()
    else:
        asyncio.run(evaluator.run())
