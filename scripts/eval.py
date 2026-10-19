"""Interactive evaluation CLI.

Usage:
    uv run python scripts/eval.py           # rate answers to evals/queries.json
    uv run python scripts/eval.py --report  # summary from evals/results.jsonl

Thin wrapper around Evaluator so ratings, result storage, and Langfuse scoring
share one implementation with the library pipeline.
"""

import argparse
import asyncio
import logging

from dotenv import load_dotenv

load_dotenv()  # loads .env from cwd (or any parent dir)

from agentic_rag.evaluation.evaluator import Evaluator


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Eval the agentic RAG system")
    parser.add_argument(
        "--report", action="store_true", help="Print summary from saved results"
    )
    args = parser.parse_args()

    evaluator = Evaluator()
    if args.report:
        evaluator.report()
    else:
        asyncio.run(evaluator.run())
