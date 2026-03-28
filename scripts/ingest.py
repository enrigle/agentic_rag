#!/usr/bin/env python
"""Thin CLI wrapper for NotionIngester."""

import argparse
import asyncio
import logging

from agentic_rag.config import load_config
from agentic_rag.ingestion.notion import NotionIngester
from agentic_rag.llm.ollama import OllamaLLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Notion → ChromaDB ingestion")
    parser.add_argument("--full", action="store_true", help="Force full re-index")
    parser.add_argument(
        "--status", action="store_true", help="Print collection stats and exit"
    )
    args = parser.parse_args()

    config = load_config()
    llm = OllamaLLM(config.llm)
    ingester = NotionIngester(config, llm)

    if args.status:
        stats = ingester.status()
        for key, value in stats.items():
            print(f"{key}: {value}")
        return

    total = asyncio.run(ingester.ingest(full=args.full))
    print(f"Done. Indexed {total} chunks.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
