"""Interface for ingesters that load a source into the RAG knowledge base."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class IngestStats:
    """What is currently indexed. Edit times are ISO-8601 strings from the source."""

    total_chunks: int
    distinct_pages: int
    oldest_edit: str | None
    newest_edit: str | None


class BaseIngester(ABC):
    """Base interface for data ingestion into the RAG knowledge base."""

    @abstractmethod
    async def ingest(self, full: bool = False) -> int:
        """Ingest documents into the knowledge base.

        Args:
            full: If True, force a full re-index regardless of change detection.

        Returns:
            Total number of chunks ingested.
        """
        ...

    @abstractmethod
    def status(self) -> IngestStats:
        """Summarise what is currently indexed."""
        ...
