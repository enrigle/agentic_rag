"""NotionIngester.status() summarises the index as typed IngestStats (SPEC §15)."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

from agentic_rag.ingestion.base import BaseIngester, IngestStats
from agentic_rag.ingestion.notion import NotionIngester


def test_status_is_part_of_the_ingester_interface() -> None:
    assert "status" in BaseIngester.__abstractmethods__
    assert inspect.signature(BaseIngester.status).return_annotation in (
        IngestStats,
        "IngestStats",
    )


def test_status_summarises_chunks_pages_and_edit_range() -> None:
    with patch("agentic_rag.ingestion.notion.chromadb.PersistentClient"):
        ingester = NotionIngester(MagicMock(), MagicMock())
    ingester._collection = MagicMock()
    ingester._collection.get.return_value = {
        "metadatas": [
            {"page_id": "p1", "last_edited_time": "2026-01-02"},
            {"page_id": "p1", "last_edited_time": "2026-01-02"},
            {"page_id": "p2", "last_edited_time": "2025-12-30"},
            {"page_id": "p3"},
        ]
    }
    assert ingester.status() == IngestStats(
        total_chunks=4,
        distinct_pages=3,
        oldest_edit="2025-12-30",
        newest_edit="2026-01-02",
    )


def test_status_of_empty_index() -> None:
    with patch("agentic_rag.ingestion.notion.chromadb.PersistentClient"):
        ingester = NotionIngester(MagicMock(), MagicMock())
    ingester._collection = MagicMock()
    ingester._collection.get.return_value = {"metadatas": []}
    assert ingester.status() == IngestStats(0, 0, None, None)
