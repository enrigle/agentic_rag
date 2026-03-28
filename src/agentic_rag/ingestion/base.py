"""Abstract base class for all ingester implementations."""

from abc import ABC, abstractmethod


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
