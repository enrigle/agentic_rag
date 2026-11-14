"""Unit tests for ChromaVectorStore.fetch_by_page_ids."""

from __future__ import annotations

import uuid

import chromadb

from agentic_rag.config import RAGConfig
from agentic_rag.retrieval.chroma import ChromaVectorStore


def _store(chunks: list[tuple[str, str]]) -> ChromaVectorStore:
    """Build a store; chunks are (page_id, chunk_id suffix index)."""
    collection = chromadb.EphemeralClient().create_collection(f"t_{uuid.uuid4().hex}")
    collection.add(
        ids=[f"{pid}_chunk_{i}" for pid, i in chunks],
        documents=[f"{pid} text {i}" for pid, i in chunks],
        embeddings=[[1.0, 0.0, 0.0]] * len(chunks),  # type: ignore[arg-type]
        metadatas=[
            {"title": f"T {pid}", "source": f"s/{pid}", "page_id": pid}
            for pid, _ in chunks
        ],
    )
    return ChromaVectorStore(RAGConfig(), collection=collection)


_CHUNKS = [("p1", "2"), ("p1", "10"), ("p2", "0"), ("p1", "0"), ("p3", "0")]


async def test_fetch_by_page_ids_orders_by_page_then_chunk() -> None:
    res = await _store(_CHUNKS).fetch_by_page_ids(["p2", "p1"], limit=10)
    assert [r.id for r in res] == [
        "p2_chunk_0",
        "p1_chunk_0",
        "p1_chunk_2",
        "p1_chunk_10",
    ]
    assert all(r.score == 1.0 and r.similarity is None for r in res)
    assert res[0].title == "T p2" and res[0].content == "p2 text 0"


async def test_fetch_by_page_ids_respects_limit() -> None:
    res = await _store(_CHUNKS).fetch_by_page_ids(["p1"], limit=2)
    assert [r.id for r in res] == ["p1_chunk_0", "p1_chunk_2"]


async def test_fetch_by_page_ids_empty_list_returns_empty() -> None:
    assert await _store(_CHUNKS).fetch_by_page_ids([], limit=5) == []


async def test_fetch_by_page_ids_gives_each_page_a_quota() -> None:
    chunks = [("p1", str(i)) for i in range(30)] + [("p2", "0"), ("p2", "1")]
    res = await _store(chunks).fetch_by_page_ids(["p1", "p2"], limit=20)
    ids = [r.id for r in res]
    assert len(ids) == 20
    assert "p2_chunk_0" in ids and "p2_chunk_1" in ids
    assert ids[0] == "p1_chunk_0"


async def test_fetch_by_page_ids_fills_leftover_slots() -> None:
    chunks = [("p1", str(i)) for i in range(5)] + [("p2", "0")]
    res = await _store(chunks).fetch_by_page_ids(["p1", "p2"], limit=4)
    assert [r.id for r in res] == [
        "p1_chunk_0",
        "p1_chunk_1",
        "p1_chunk_2",
        "p2_chunk_0",
    ]
