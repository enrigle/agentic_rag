"""Background Notion ingest shared by every browser session of the app."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from agentic_rag.config import load_config
from agentic_rag.ingestion.base import BaseIngester
from agentic_rag.ingestion.notion import NotionIngester
from agentic_rag.pipeline.rag_pipeline import make_embed_llm

logger = logging.getLogger(__name__)


@dataclass
class IngestStatus:
    status: Literal["idle", "syncing", "done", "error"] = "idle"
    chunks: int = 0
    error: str = ""


def default_ingester() -> BaseIngester:
    """NotionIngester using the configured embedding backend."""
    config = load_config()
    return NotionIngester(config, make_embed_llm(config))


class BackgroundIngest:
    """Runs at most one incremental ingest at a time, in a daemon thread.

    One instance is shared process-wide (app.py caches it), so the lock and
    status check stop N browser sessions from fanning out into N ingests.
    """

    def __init__(
        self, make_ingester: Callable[[], BaseIngester] = default_ingester
    ) -> None:
        self._make_ingester = make_ingester
        self._lock = threading.Lock()
        self.state = IngestStatus()

    def start(self) -> bool:
        """Start an ingest unless one is running. Returns True if this call started it."""
        with self._lock:
            if self.state.status == "syncing":
                return False
            self.state.status = "syncing"
        threading.Thread(target=self.run, daemon=True).start()
        return True

    def run(self) -> None:
        """Ingest synchronously, recording the outcome in self.state."""
        self.state.chunks = 0
        self.state.error = ""
        try:
            total = asyncio.run(self._make_ingester().ingest(full=False))
            self.state.chunks = total
            self.state.status = "done"
        except Exception:  # noqa: BLE001
            # Detail goes to the server log only; the UI gets a fixed string.
            logger.exception("ingest failed")
            self.state.error = "see server logs"
            self.state.status = "error"
