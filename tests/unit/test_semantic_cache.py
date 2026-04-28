"""Unit tests for SemanticCache."""

from __future__ import annotations

import dataclasses
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from agentic_rag.cache.semantic_cache import SemanticCache
from agentic_rag.config import RedisConfig
from agentic_rag.models import QueryResult, SearchResult


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_result(answer: str = "cached answer") -> QueryResult:
    return QueryResult(
        answer=answer,
        sources=[
            SearchResult(
                id="1",
                title="Doc",
                source="kb",
                content="some content",
                score=0.9,
            )
        ],
        tool_calls_used=1,
        latency_ms=42.0,
    )


def _serialise(result: QueryResult, vec: list[float]) -> dict[bytes, bytes]:
    arr = np.array(vec, dtype=np.float32)
    return {
        b"embedding": arr.tobytes(),
        b"result": json.dumps(dataclasses.asdict(result)).encode(),
        b"query": b"what is X",
    }


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_redis() -> MagicMock:
    r = MagicMock()
    r.keys = AsyncMock(return_value=[])
    r.hgetall = AsyncMock(return_value={})
    r.hset = AsyncMock()
    r.expire = AsyncMock()
    return r


@pytest.fixture
def cache(mock_redis: MagicMock) -> SemanticCache:
    with patch(
        "agentic_rag.cache.semantic_cache.aioredis.Redis.from_url",
        return_value=mock_redis,
    ):
        cfg = RedisConfig(
            url="redis://localhost:6379",
            ttl_seconds=60,
            similarity_threshold=0.9,
        )
        embed_llm = MagicMock()
        embed_llm.embed = AsyncMock(return_value=[1.0, 0.0, 0.0])
        return SemanticCache(cfg, embed_llm)


# ── tests ─────────────────────────────────────────────────────────────────────


async def test_get_returns_none_on_empty_cache(
    cache: SemanticCache, mock_redis: MagicMock
) -> None:
    mock_redis.keys.return_value = []

    result = await cache.get("what is X?")

    assert result is None


async def test_get_returns_result_on_similarity_hit(
    cache: SemanticCache, mock_redis: MagicMock
) -> None:
    stored = _make_result("cached answer")
    vec = [1.0, 0.0, 0.0]
    mock_redis.keys.return_value = [b"cache:abc123"]
    mock_redis.hgetall.return_value = _serialise(stored, vec)
    # embed returns the same vector → cosine similarity = 1.0
    cache._embed_llm.embed = AsyncMock(return_value=vec)

    result = await cache.get("what is X?")

    assert result is not None
    assert result.answer == "cached answer"
    assert len(result.sources) == 1
    assert result.sources[0].id == "1"


async def test_get_returns_none_on_similarity_miss(
    cache: SemanticCache, mock_redis: MagicMock
) -> None:
    stored = _make_result("cached answer")
    stored_vec = [1.0, 0.0, 0.0]
    orthogonal_vec = [0.0, 1.0, 0.0]  # dot product with stored_vec = 0
    mock_redis.keys.return_value = [b"cache:abc123"]
    mock_redis.hgetall.return_value = _serialise(stored, stored_vec)
    cache._embed_llm.embed = AsyncMock(return_value=orthogonal_vec)

    result = await cache.get("unrelated query")

    assert result is None


async def test_set_stores_with_ttl(cache: SemanticCache, mock_redis: MagicMock) -> None:
    result = _make_result()

    await cache.set("what is X?", result)

    mock_redis.hset.assert_awaited_once()
    call_kwargs = mock_redis.hset.call_args
    mapping = call_kwargs.kwargs.get("mapping") or call_kwargs.args[1]
    assert isinstance(mapping["embedding"], bytes)
    assert "cached answer" in mapping["result"]

    mock_redis.expire.assert_awaited_once()
    expire_args = mock_redis.expire.call_args.args
    assert expire_args[1] == 60  # ttl_seconds from fixture


async def test_cache_roundtrip(mock_redis: MagicMock) -> None:
    """set() followed by get() returns the same answer via an in-memory store."""
    store: dict[bytes, dict[bytes, bytes]] = {}

    async def fake_hset(key: str, mapping: dict[str, bytes]) -> None:
        store[key.encode()] = {k.encode(): v for k, v in mapping.items()}

    async def fake_keys(pattern: str) -> list[bytes]:
        return list(store.keys())

    async def fake_hgetall(key: str | bytes) -> dict[bytes, bytes]:
        lookup = key.encode() if isinstance(key, str) else key
        return store.get(lookup, {})

    mock_redis.hset = AsyncMock(side_effect=fake_hset)
    mock_redis.keys = AsyncMock(side_effect=fake_keys)
    mock_redis.hgetall = AsyncMock(side_effect=fake_hgetall)

    vec = [1.0, 0.0, 0.0]

    with patch(
        "agentic_rag.cache.semantic_cache.aioredis.Redis.from_url",
        return_value=mock_redis,
    ):
        cfg = RedisConfig(
            url="redis://localhost:6379",
            ttl_seconds=60,
            similarity_threshold=0.9,
        )
        embed_llm = MagicMock()
        embed_llm.embed = AsyncMock(return_value=vec)
        sc = SemanticCache(cfg, embed_llm)

    original = _make_result("roundtrip answer")
    await sc.set("test query", original)
    retrieved = await sc.get("test query")

    assert retrieved is not None
    assert retrieved.answer == "roundtrip answer"
    assert retrieved.tool_calls_used == original.tool_calls_used
    assert retrieved.latency_ms == original.latency_ms
