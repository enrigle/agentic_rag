"""BM25 keyword retriever backed by bm25s."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import bm25s  # type: ignore[import-untyped]

from agentic_rag.config import RAGConfig
from agentic_rag.retrieval.base import BaseKeywordRetriever

logger = logging.getLogger(__name__)


class BM25Retriever(BaseKeywordRetriever):
    """BM25 keyword retriever that persists its index to disk."""

    def __init__(self, config: RAGConfig) -> None:
        self._bm25_path = Path(config.bm25_path)
        self._retriever: bm25s.BM25 | None = None
        self._ids: list[str] = []
        self._load()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load BM25 index and ID map from disk; logs a warning if absent."""
        id_map_path = self._bm25_path / "id_map.json"
        if not self._bm25_path.exists() or not id_map_path.exists():
            logger.warning(
                "BM25 index not found at %s — falling back to vector-only search. "
                "Run ingest to build it.",
                self._bm25_path,
            )
            return
        try:
            self._retriever = bm25s.BM25.load(str(self._bm25_path), load_corpus=False)
            self._ids = json.loads(id_map_path.read_text())
            logger.info("BM25 index loaded: %d documents", len(self._ids))
        except Exception as exc:
            logger.warning(
                "Failed to load BM25 index from %s: %s — vector-only search active",
                self._bm25_path,
                exc,
            )
            self._retriever = None
            self._ids = []

    # ------------------------------------------------------------------
    # BaseKeywordRetriever interface
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int) -> list[str]:
        """Tokenize query and return up to top_k doc IDs by BM25 score.

        Args:
            query: Raw query string.
            top_k: Maximum number of doc IDs to return.

        Returns:
            Ordered list of doc IDs, or [] if the index is not loaded.
        """
        if self._retriever is None or not self._ids:
            return []

        k = min(top_k, len(self._ids))
        try:
            results, _ = self._retriever.retrieve(
                bm25s.tokenize([query], show_progress=False),
                corpus=self._ids,
                k=k,
                show_progress=False,
            )
            doc_ids: list[str] = list(results[0])
            logger.debug("BM25Retriever.search: returned %d candidates", len(doc_ids))
            return doc_ids
        except Exception as exc:
            logger.warning("BM25Retriever.search failed: %s", exc)
            return []

    def rebuild(self, ids: list[str], documents: list[str]) -> None:
        """Tokenize documents, build a new BM25 index, and save to disk.

        Args:
            ids: Document IDs parallel to *documents*.
            documents: Raw document strings to index.
        """
        if not documents:
            logger.warning(
                "BM25Retriever.rebuild: called with empty documents — skipping"
            )
            return

        tokenized = bm25s.tokenize(documents, show_progress=False)
        retriever = bm25s.BM25()
        retriever.index(tokenized, show_progress=False)

        self._bm25_path.mkdir(parents=True, exist_ok=True)
        retriever.save(str(self._bm25_path))
        (self._bm25_path / "id_map.json").write_text(json.dumps(ids))

        self._retriever = retriever
        self._ids = list(ids)
        logger.info(
            "BM25Retriever.rebuild: indexed and saved %d documents", len(documents)
        )
