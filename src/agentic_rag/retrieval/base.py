"""Abstract base classes for retrieval components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseVectorStore(ABC):
    @abstractmethod
    async def search(self, query_vec: list[float], top_k: int) -> list[Any]:
        """Search vector store and return list[SearchResult]."""
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
