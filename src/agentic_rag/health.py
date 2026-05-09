"""Startup health checks for external dependencies."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path


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
        await r.ping()
        await r.aclose()
        return ServiceStatus("Redis", True)
    except Exception:
        return ServiceStatus("Redis", False, "Not running — semantic cache disabled")


def _check_groq() -> ServiceStatus:
    if os.environ.get("GROQ_API_KEY"):
        return ServiceStatus("Groq", True)
    return ServiceStatus("Groq", False, "GROQ_API_KEY not set — falling back to Ollama")


def _check_chromadb(chroma_path: str) -> ServiceStatus:
    if Path(chroma_path).exists():
        return ServiceStatus("ChromaDB", True)
    return ServiceStatus("ChromaDB", False, "No index found — sync Notion first")


async def run_checks(
    ollama_url: str,
    redis_url: str,
    chroma_path: str,
) -> list[ServiceStatus]:
    ollama_st, redis_st = await asyncio.gather(
        _check_ollama(ollama_url),
        _check_redis(redis_url),
    )
    return [ollama_st, redis_st, _check_groq(), _check_chromadb(chroma_path)]
