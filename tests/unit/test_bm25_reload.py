"""BM25Retriever picks up on-disk index changes without being recreated.

The Streamlit app keeps one cached BM25Retriever alive while a background
ingest thread rebuilds the index on disk; search() must serve the new index.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agentic_rag.retrieval.bm25 import BM25Retriever


@pytest.fixture
def cfg(tmp_path: Path) -> Any:
    return SimpleNamespace(bm25_path=str(tmp_path / "bm25_index"))


def test_search_picks_up_index_built_after_init(cfg: Any) -> None:
    retriever = BM25Retriever(cfg)
    assert retriever.search("apple", top_k=5) == []

    # Ingest builds the first index via its own instance
    BM25Retriever(cfg).rebuild(["d1"], ["apple pie recipe"])

    assert retriever.search("apple", top_k=5) == ["d1"]


def test_search_picks_up_rebuilt_index(cfg: Any) -> None:
    BM25Retriever(cfg).rebuild(["d1"], ["apple pie recipe"])
    retriever = BM25Retriever(cfg)
    assert retriever.search("apple", top_k=5) == ["d1"]

    BM25Retriever(cfg).rebuild(["d2"], ["banana bread"])
    # Force a distinct mtime in case both writes land in the same clock tick
    id_map = Path(cfg.bm25_path) / "id_map.json"
    os.utime(id_map, (id_map.stat().st_atime, id_map.stat().st_mtime + 1))

    assert retriever.search("banana", top_k=5) == ["d2"]
    assert retriever.search("apple", top_k=5) != ["d1"]


def test_missing_index_stays_empty(cfg: Any) -> None:
    retriever = BM25Retriever(cfg)
    assert retriever.search("anything", top_k=5) == []
