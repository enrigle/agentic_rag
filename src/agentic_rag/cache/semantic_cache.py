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

    async def get(self, query: str) -> Optional[QueryResult]:
        """Return cached QueryResult for the best-matching similar query, else None."""
        try:
            vec = await self._embed_llm.embed(query)
        except Exception:
            logger.warning(
                "embed() failed during cache lookup; skipping cache", exc_info=True
            )
            return None

        query_arr = np.array(vec, dtype=np.float32)

        if query_arr.size == 0:
            logger.warning("embed() returned empty vector; skipping cache lookup")
            return None

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
        except aioredis.RedisError:
            logger.warning("Redis SCAN failed; skipping cache lookup", exc_info=True)
            return None

        if not keys:
            return None

        best_similarity = -1.0
        best_result: Optional[QueryResult] = None

        for key in keys:
            str_key = key.decode() if isinstance(key, bytes) else key
            try:
                data: dict[bytes, bytes] = cast(
                    dict[bytes, bytes],
                    await cast(Any, self._redis.hgetall(str_key)),
                )
            except aioredis.RedisError:
                logger.warning(
                    "Redis HGETALL failed for key %s; skipping", str_key, exc_info=True
                )
                return None

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

    async def set(self, query: str, result: QueryResult) -> None:
        """Store query result in Redis with TTL."""
        try:
            vec = await self._embed_llm.embed(query)
            arr = np.array(vec, dtype=np.float32)
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
        except Exception:
            logger.warning(
                "Failed to store result in cache; failing open", exc_info=True
            )
