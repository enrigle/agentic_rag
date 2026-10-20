"""Run one query through the RAG pipeline without the UI.

Usage:
    uv run python scripts/main.py "How do we deploy ML models?"
    uv run python scripts/main.py "And on Kubernetes?" --thread-id t1

Thin wrapper around create_pipeline() so the CLI and the Streamlit app share
one implementation (retrieval, reranking, web fallback, synthesis, cache).
"""

import argparse
import asyncio
import dataclasses
import json
import logging

from dotenv import load_dotenv

load_dotenv()  # loads .env from cwd (or any parent dir)

from agentic_rag.pipeline.rag_pipeline import create_pipeline


async def run(query: str, thread_id: str) -> None:
    result = await create_pipeline().query(query, thread_id=thread_id)
    print(json.dumps(dataclasses.asdict(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Query the agentic RAG pipeline")
    parser.add_argument("query", help="Question to answer")
    parser.add_argument(
        "--thread-id", default="cli", help="Conversation id for follow-up memory"
    )
    args = parser.parse_args()
    if not args.query.strip():
        parser.error("query cannot be empty")
    asyncio.run(run(args.query, args.thread_id))
