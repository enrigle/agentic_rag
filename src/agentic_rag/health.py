"""Startup health checks for external dependencies."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from agentic_rag.config import EmbedBackend


@dataclass
class ServiceStatus:
    name: str
    ok: bool
    detail: str = ""


async def _check_ollama(base_url: str) -> ServiceStatus:
    try:
        import ollama

        await ollama.AsyncClient(host=base_url).list()
        return ServiceStatus("Ollama", True)
    except Exception:
        return ServiceStatus("Ollama", False, "Not running — start with: ollama serve")


async def _check_redis(url: str) -> ServiceStatus:
    try:
        import redis.asyncio as aioredis

        r = aioredis.Redis.from_url(url)
        # redis-py types ping() as Awaitable[bool] | bool; asyncio client awaits.
        await cast(Awaitable[bool], r.ping())
        await r.aclose()
        return ServiceStatus("Redis", True)
    except Exception:
        return ServiceStatus("Redis", False, "Not running — semantic cache disabled")


def _check_groq() -> ServiceStatus:
    if os.environ.get("GROQ_API_KEY"):
        return ServiceStatus("Groq", True)
    return ServiceStatus("Groq", False, "GROQ_API_KEY not set — using configured LLM")


def _check_chromadb(chroma_path: Path) -> ServiceStatus:
    if chroma_path.exists():
        return ServiceStatus("ChromaDB", True)
    return ServiceStatus("ChromaDB", False, "No index found — sync Notion first")


async def run_checks(
    ollama_url: str,
    redis_url: str,
    chroma_path: Path,
    embed_backend: EmbedBackend = "ollama",
    synth_uses_ollama: bool = True,
) -> list[ServiceStatus]:
    """Probe every service the configured pipeline depends on.

    Ollama is needed when it embeds *or* synthesizes (no Groq/Azure configured),
    so in-process embeddings alone don't make it optional.
    """
    st_embed = embed_backend == "sentence_transformers"
    needs_ollama = not st_embed or synth_uses_ollama

    if needs_ollama:
        ollama_st, redis_st = await asyncio.gather(
            _check_ollama(ollama_url),
            _check_redis(redis_url),
        )
        services = [ollama_st]
    else:
        redis_st = await _check_redis(redis_url)
        services = []

    if st_embed:
        services.insert(
            0, ServiceStatus("Embeddings", True, "sentence-transformers (in-process)")
        )
    return [*services, redis_st, _check_groq(), _check_chromadb(chroma_path)]
