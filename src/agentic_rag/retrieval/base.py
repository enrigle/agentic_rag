"""Abstract base classes for retrieval components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentic_rag.models import SearchResult


class BaseVectorStore(ABC):
    @abstractmethod
    async def search(self, query_vec: list[float], top_k: int) -> list[SearchResult]:
        """Search vector store and return list[SearchResult]."""
        ...

    @abstractmethod
    async def fetch_by_ids(self, ids: list[str]) -> list[SearchResult]:
        """Fetch documents by ID without requiring a query vector."""
        ...

    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Upsert documents into the vector store."""
        ...


class BaseKeywordRetriever(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int) -> list[str]:
        """Search keyword index and return doc IDs."""
        ...

    @abstractmethod
    def rebuild(self, ids: list[str], documents: list[str]) -> None:
        """Rebuild the keyword index from the given documents."""
        ...
