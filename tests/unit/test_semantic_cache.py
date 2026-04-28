"""Unit tests for SemanticCache."""

from __future__ import annotations

import dataclasses
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import redis.asyncio as aioredis

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


def _make_scan_side_effect(keys: list[bytes]) -> AsyncMock:
    """Return an AsyncMock that emits all keys in a single SCAN page."""
    return AsyncMock(return_value=(0, keys))


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_redis() -> MagicMock:
    r = MagicMock()
    r.scan = _make_scan_side_effect([])
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


# ── in-memory store helpers for roundtrip test ────────────────────────────────


class _InMemoryRedis:
    """Minimal in-memory Redis stand-in for roundtrip tests."""

    def __init__(self) -> None:
        self._store: dict[bytes, dict[bytes, bytes]] = {}

    async def hset(self, key: str, mapping: dict[str, bytes]) -> None:
        self._store[key.encode()] = {k.encode(): v for k, v in mapping.items()}

    async def scan(
        self, cursor: int, match: str = "*", count: int = 100
    ) -> tuple[int, list[bytes]]:
        return (0, list(self._store.keys()))

    async def hgetall(self, key: str | bytes) -> dict[bytes, bytes]:
        lookup = key.encode() if isinstance(key, str) else key
        return self._store.get(lookup, {})

    async def expire(self, key: str, ttl: int) -> None:
        pass


# ── tests ─────────────────────────────────────────────────────────────────────


async def test_get_returns_none_on_empty_cache(
    cache: SemanticCache, mock_redis: MagicMock
) -> None:
    mock_redis.scan = _make_scan_side_effect([])

    result = await cache.get("what is X?")

    assert result is None


async def test_get_returns_result_on_similarity_hit(
    cache: SemanticCache, mock_redis: MagicMock
) -> None:
    stored = _make_result("cached answer")
    vec = [1.0, 0.0, 0.0]
    mock_redis.scan = _make_scan_side_effect([b"cache:abc123"])
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
    mock_redis.scan = _make_scan_side_effect([b"cache:abc123"])
    mock_redis.hgetall.return_value = _serialise(stored, stored_vec)
    cache._embed_llm.embed = AsyncMock(return_value=orthogonal_vec)

    result = await cache.get("unrelated query")

    assert result is None


async def test_set_stores_with_ttl(cache: SemanticCache, mock_redis: MagicMock) -> None:
    result = _make_result()

    await cache.set("what is X?", result)

    mock_redis.hset.assert_awaited_once()
    call_kwargs = mock_redis.hset.call_args
    mapping = call_kwargs.kwargs["mapping"]
    assert isinstance(mapping["embedding"], bytes)
    assert "cached answer" in mapping["result"]

    mock_redis.expire.assert_awaited_once()
    expire_args = mock_redis.expire.call_args.args
    assert expire_args[1] == 60  # ttl_seconds from fixture


async def test_cache_roundtrip(mock_redis: MagicMock) -> None:
    """set() followed by get() returns the same answer via an in-memory store."""
    in_mem = _InMemoryRedis()
    mock_redis.hset = AsyncMock(side_effect=in_mem.hset)
    mock_redis.scan = AsyncMock(side_effect=in_mem.scan)
    mock_redis.hgetall = AsyncMock(side_effect=in_mem.hgetall)
    mock_redis.expire = AsyncMock(side_effect=in_mem.expire)

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


async def test_get_returns_none_on_redis_error(
    cache: SemanticCache, mock_redis: MagicMock
) -> None:
    """SCAN raising RedisError causes get() to fail open (return None, not raise)."""
    mock_redis.scan = AsyncMock(side_effect=aioredis.RedisError("connection lost"))

    result = await cache.get("any query")

    assert result is None


async def test_get_returns_none_on_corrupt_data(
    cache: SemanticCache, mock_redis: MagicMock
) -> None:
    """Corrupt hgetall data causes get() to skip the key and return None."""
    mock_redis.scan = _make_scan_side_effect([b"cache:corrupt"])
    mock_redis.hgetall.return_value = {
        b"embedding": b"garbage",
        b"result": b"not json",
        b"query": b"x",
    }

    result = await cache.get("any query")

    assert result is None


async def test_get_returns_best_match_among_multiple_keys(
    cache: SemanticCache, mock_redis: MagicMock
) -> None:
    """get() returns the result from the highest-similarity key above threshold."""
    low_result = _make_result("low similarity answer")
    high_result = _make_result("high similarity answer")

    # query vector
    query_vec = [1.0, 0.0, 0.0]
    cache._embed_llm.embed = AsyncMock(return_value=query_vec)

    # stored_low: similarity ~0.92  (slightly off-axis)
    low_vec = np.array([0.92, 0.39, 0.0], dtype=np.float32)
    low_vec = (low_vec / np.linalg.norm(low_vec)).tolist()

    # stored_high: similarity ~0.97 (closer to query)
    high_vec = np.array([0.97, 0.24, 0.0], dtype=np.float32)
    high_vec = (high_vec / np.linalg.norm(high_vec)).tolist()

    mock_redis.scan = _make_scan_side_effect([b"cache:low", b"cache:high"])

    async def hgetall_dispatch(key: str | bytes) -> dict[bytes, bytes]:
        k = key.decode() if isinstance(key, bytes) else key
        if k == "cache:low":
            return _serialise(low_result, low_vec)
        return _serialise(high_result, high_vec)

    mock_redis.hgetall = AsyncMock(side_effect=hgetall_dispatch)

    result = await cache.get("test query")

    assert result is not None
    assert result.answer == "high similarity answer"


async def test_set_fails_open_on_redis_error(
    cache: SemanticCache, mock_redis: MagicMock
) -> None:
    """set() swallows RedisError and does not raise."""
    mock_redis.hset = AsyncMock(side_effect=aioredis.RedisError("write failed"))

    # Must not raise
    await cache.set("any query", _make_result())
