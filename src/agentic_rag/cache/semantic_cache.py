"""Redis-backed semantic cache for RAG query results."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from typing import Any, Optional, cast

import numpy as np
import redis.asyncio as aioredis

from agentic_rag.config import RedisConfig
from agentic_rag.llm.base import BaseLLM
from agentic_rag.models import QueryResult, SearchResult

logger = logging.getLogger(__name__)


class SemanticCache:
    def __init__(self, config: RedisConfig, embed_llm: BaseLLM) -> None:
        self._config = config
        self._embed_llm = embed_llm
        self._redis: aioredis.Redis = aioredis.Redis.from_url(
            config.url, decode_responses=False
        )
        self._available = True

    async def get(
        self, query: str, query_vec: Optional[list[float]] = None
    ) -> Optional[QueryResult]:
        """Return cached QueryResult for the best-matching similar query, else None.

        ``query_vec`` lets the caller pass a precomputed embedding to avoid a
        redundant embed call; when None the query is embedded here.
        """
        if not self._available:
            return None

        query_arr = await self._query_vector(query, query_vec)
        if query_arr is None:
            return None

        keys = await self._scan_keys()
        if not keys:
            return None

        all_data = await self._fetch_hashes(keys)
        if all_data is None:
            return None

        return self._best_match(query_arr, keys, all_data)

    async def _query_vector(
        self, query: str, query_vec: Optional[list[float]]
    ) -> Optional[np.ndarray]:
        """Embed the query (if needed) and return it as a float32 array, or None."""
        if query_vec is None:
            try:
                query_vec = await self._embed_llm.embed(query)
            except Exception:
                logger.debug("embed() failed during cache lookup; skipping cache")
                return None

        query_arr = np.array(query_vec, dtype=np.float32)
        if query_arr.size == 0:
            logger.warning("embed() returned empty vector; skipping cache lookup")
            return None
        return query_arr

    async def _scan_keys(self) -> Optional[list[bytes]]:
        """Return all cache:* keys, or None if Redis is unavailable."""
        try:
            cursor = 0
            keys: list[bytes] = []
            while True:
                cursor, batch = cast(
                    tuple[int, list[bytes]],
                    await self._redis.scan(cursor, match="cache:*", count=100),
                )
                keys.extend(batch)
                if cursor == 0:
                    break
            return keys
        except aioredis.RedisError:
            self._available = False
            logger.warning("Redis unavailable at %s; cache disabled", self._config.url)
            return None

    async def _fetch_hashes(
        self, keys: list[bytes]
    ) -> Optional[list[dict[bytes, bytes]]]:
        """Fetch every candidate hash in a single round-trip, or None on error."""
        try:
            pipe = self._redis.pipeline()
            for key in keys:
                pipe.hgetall(key.decode() if isinstance(key, bytes) else key)
            return cast(list[dict[bytes, bytes]], await pipe.execute())
        except aioredis.RedisError:
            logger.warning("Redis pipeline HGETALL failed; skipping", exc_info=True)
            return None

    def _best_match(
        self,
        query_arr: np.ndarray,
        keys: list[bytes],
        all_data: list[dict[bytes, bytes]],
    ) -> Optional[QueryResult]:
        """Return the QueryResult of the most similar candidate above threshold."""
        best_similarity = -1.0
        best_result: Optional[QueryResult] = None

        for key, data in zip(keys, all_data):
            if b"embedding" not in data:
                continue

            try:
                stored_arr = np.frombuffer(data[b"embedding"], dtype=np.float32)

                if stored_arr.size == 0 or stored_arr.shape != query_arr.shape:
                    logger.warning(
                        "Skipping cache key %s: bad stored embedding shape", key
                    )
                    continue

                similarity = float(
                    np.dot(query_arr, stored_arr)
                    / (np.linalg.norm(query_arr) * np.linalg.norm(stored_arr) + 1e-9)
                )

                if (
                    similarity >= self._config.similarity_threshold
                    and similarity > best_similarity
                ):
                    raw: dict[str, Any] = json.loads(data[b"result"])
                    sources = [SearchResult(**s) for s in raw["sources"]]
                    best_similarity = similarity
                    best_result = QueryResult(
                        answer=str(raw["answer"]),
                        sources=sources,
                        tool_calls_used=int(raw["tool_calls_used"]),
                        latency_ms=float(raw["latency_ms"]),
                    )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                logger.warning(
                    "Corrupt data for cache key %s; skipping", key, exc_info=True
                )
                continue

        return best_result

    async def set(
        self, query: str, result: QueryResult, query_vec: Optional[list[float]] = None
    ) -> None:
        """Store query result in Redis with TTL.

        ``query_vec`` lets the caller pass a precomputed embedding to avoid a
        redundant embed call; when None the query is embedded here.
        """
        if not self._available:
            return

        try:
            if query_vec is None:
                query_vec = await self._embed_llm.embed(query)
            arr = np.array(query_vec, dtype=np.float32)
            key_hash = hashlib.sha256(arr.tobytes()).hexdigest()
            key = f"cache:{key_hash}"
            result_dict = dataclasses.asdict(result)
            await cast(
                Any,
                self._redis.hset(
                    key,
                    mapping={
                        "embedding": arr.tobytes(),
                        "result": json.dumps(result_dict),
                        "query": query,
                    },
                ),
            )
            await cast(Any, self._redis.expire(key, self._config.ttl_seconds))
        except aioredis.RedisError:
            self._available = False
            logger.warning("Redis unavailable at %s; cache disabled", self._config.url)
        except Exception:
            logger.debug("Failed to store result in cache; skipping")
