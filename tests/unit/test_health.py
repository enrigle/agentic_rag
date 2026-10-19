"""Unit tests for run_checks: which services get probed."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from agentic_rag import health
from agentic_rag.health import ServiceStatus


async def _names(embed_backend: str, synth_uses_ollama: bool) -> list[str]:
    ollama = AsyncMock(return_value=ServiceStatus("Ollama", False, "down"))
    redis = AsyncMock(return_value=ServiceStatus("Redis", True))
    with (
        patch.object(health, "_check_ollama", ollama),
        patch.object(health, "_check_redis", redis),
    ):
        statuses = await health.run_checks(
            ollama_url="http://x",
            redis_url="redis://x",
            chroma_path="/nonexistent",
            embed_backend=embed_backend,
            synth_uses_ollama=synth_uses_ollama,
        )
    return [s.name for s in statuses]


@pytest.mark.asyncio
async def test_st_embeddings_with_ollama_synthesis_still_probes_ollama() -> None:
    names = await _names("sentence_transformers", synth_uses_ollama=True)
    assert "Ollama" in names
    assert "Embeddings" in names


@pytest.mark.asyncio
async def test_st_embeddings_with_cloud_synthesis_skips_ollama() -> None:
    assert "Ollama" not in await _names(
        "sentence_transformers", synth_uses_ollama=False
    )


@pytest.mark.asyncio
async def test_ollama_embeddings_probe_ollama() -> None:
    assert "Ollama" in await _names("ollama", synth_uses_ollama=False)
