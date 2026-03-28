#!/usr/bin/env python
"""Thin CLI wrapper for Evaluator."""

import argparse
import asyncio
import logging

from agentic_rag.evaluation.evaluator import Evaluator


def main() -> None:
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    main()
