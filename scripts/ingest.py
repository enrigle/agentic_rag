"""Notion → ChromaDB ingestion CLI.

Usage:
    export NOTION_TOKEN=secret_xxx
    uv run python scripts/ingest.py              # incremental (skip unchanged, prune deleted)
    uv run python scripts/ingest.py --full       # force full re-index
    uv run python scripts/ingest.py --status     # print collection stats and exit

Thin wrapper around NotionIngester so the CLI and the app's background ingest
share one implementation (chunking, embedding, image OCR/captioning, BM25).
"""

import argparse
import asyncio
import logging

from dotenv import load_dotenv

load_dotenv()  # loads .env from cwd (or any parent dir)

from agentic_rag.config import load_config
from agentic_rag.ingestion.notion import NotionIngester
from agentic_rag.pipeline.rag_pipeline import make_embed_llm

logger = logging.getLogger(__name__)


async def ingest(args: argparse.Namespace) -> None:
    config = load_config()
    ingester = NotionIngester(config, make_embed_llm(config))

    if args.status:
        stats = ingester.status()
        print(f"Total chunks  : {stats['total_chunks']}")
        print(f"Distinct pages: {stats['distinct_pages']}")
        if stats["oldest_edit"]:
            print(f"Oldest edit   : {stats['oldest_edit']}")
        if stats["newest_edit"]:
            print(f"Newest edit   : {stats['newest_edit']}")
        return

    total = await ingester.ingest(full=args.full)
    print(f"Done. Indexed {total} chunks into ChromaDB at {config.chroma_path!r}.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Notion → ChromaDB ingestion")
    parser.add_argument("--full", action="store_true", help="Force full re-index")
    parser.add_argument(
        "--status", action="store_true", help="Print collection stats and exit"
    )
    args = parser.parse_args()
    asyncio.run(ingest(args))
