"""Unit tests for AzureOpenAILLM."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from agentic_rag.config import AzureOpenAIConfig
from agentic_rag.llm.azure_openai import AzureOpenAILLM

_VALID_CONFIG = AzureOpenAIConfig(
    endpoint="https://my-resource.openai.azure.com",
    api_key="test-key",
    deployment="gpt-4o-mini",
    api_version="2024-02-01",
)


def _make_llm(config: AzureOpenAIConfig = _VALID_CONFIG) -> AzureOpenAILLM:
    """Construct AzureOpenAILLM with a patched client constructor."""
    with patch("agentic_rag.llm.azure_openai.openai.AsyncAzureOpenAI"):
        return AzureOpenAILLM(config)


def _make_completion(content: str) -> MagicMock:
    """Build a minimal chat completion mock with a single choice."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    completion = MagicMock()
    completion.choices = [choice]
    return completion


# ── __init__ ──────────────────────────────────────────────────────────────────


def test_init_raises_if_endpoint_empty() -> None:
    config = AzureOpenAIConfig(endpoint="", api_key="key")
    with pytest.raises(ValueError, match="endpoint"):
        AzureOpenAILLM(config)


def test_init_raises_if_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    config = AzureOpenAIConfig(
        endpoint="https://my-resource.openai.azure.com",
        api_key=None,
    )
    with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
        AzureOpenAILLM(config)


def test_init_accepts_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
    config = AzureOpenAIConfig(
        endpoint="https://my-resource.openai.azure.com",
        api_key=None,
    )
    with patch("agentic_rag.llm.azure_openai.openai.AsyncAzureOpenAI"):
        llm = AzureOpenAILLM(config)
    assert llm is not None


# ── chat ──────────────────────────────────────────────────────────────────────


async def test_chat_returns_content() -> None:
    llm = _make_llm()
    llm._client = MagicMock()  # type: ignore[assignment]
    llm._client.chat.completions.create = AsyncMock(
        return_value=_make_completion("hello")
    )

    result = await llm.chat("What is 2+2?")

    assert result == "hello"


async def test_chat_raises_runtime_error_on_api_failure() -> None:
    llm = _make_llm()
    llm._client = MagicMock()  # type: ignore[assignment]
    llm._client.chat.completions.create = AsyncMock(
        side_effect=openai.APIError("fail", request=MagicMock(), body=None)
    )

    with pytest.raises(RuntimeError, match="Azure OpenAI chat failed"):
        await llm.chat("test prompt")


async def test_chat_raises_value_error_on_empty_content() -> None:
    llm = _make_llm()
    llm._client = MagicMock()  # type: ignore[assignment]
    llm._client.chat.completions.create = AsyncMock(return_value=_make_completion(""))

    with pytest.raises(ValueError, match="empty content"):
        await llm.chat("test prompt")


async def test_chat_raises_value_error_on_empty_prompt() -> None:
    llm = _make_llm()

    with pytest.raises(ValueError, match="non-empty"):
        await llm.chat("")


# ── embed ─────────────────────────────────────────────────────────────────────


async def test_embed_raises_not_implemented() -> None:
    llm = _make_llm()

    with pytest.raises(NotImplementedError, match="OllamaLLM"):
        await llm.embed("some text")
