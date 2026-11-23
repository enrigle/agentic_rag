"""Unit tests for AzureOpenAILLM."""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from agentic_rag.config import AzureOpenAIConfig, GroqConfig
from agentic_rag.llm.openai_compat import AzureOpenAILLM, GroqLLM

_VALID_CONFIG = AzureOpenAIConfig(
    endpoint="https://my-resource.openai.azure.com",
    api_key="test-key",
    deployment="gpt-4o-mini",
    api_version="2024-02-01",
)


@pytest.fixture
def llm() -> Generator[AzureOpenAILLM, None, None]:
    """Construct AzureOpenAILLM with a patched client constructor held open."""
    with patch("agentic_rag.llm.openai_compat.openai.AsyncAzureOpenAI") as mock_cls:
        instance = AzureOpenAILLM(_VALID_CONFIG)
        instance._client = mock_cls.return_value
        yield instance


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
        endpoint="https://x.openai.azure.com",
        api_key=None,
    )
    with patch("agentic_rag.llm.openai_compat.openai.AsyncAzureOpenAI") as mock_cls:
        AzureOpenAILLM(config)
    mock_cls.assert_called_once_with(
        azure_endpoint=config.endpoint,
        api_key="env-key",
        api_version=config.api_version,
        timeout=30.0,
    )


# ── chat ──────────────────────────────────────────────────────────────────────


async def test_chat_returns_content(llm: AzureOpenAILLM) -> None:
    llm._client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        return_value=_make_completion("hello")
    )

    result = await llm.chat("What is 2+2?")

    assert result == "hello"


async def test_chat_raises_runtime_error_on_api_failure(llm: AzureOpenAILLM) -> None:
    llm._client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        side_effect=openai.APIError("fail", request=MagicMock(), body=None)
    )

    with pytest.raises(RuntimeError, match="Azure OpenAI chat failed"):
        await llm.chat("test prompt")


async def test_chat_raises_value_error_on_empty_content(llm: AzureOpenAILLM) -> None:
    llm._client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        return_value=_make_completion("")
    )

    with pytest.raises(ValueError, match="empty content"):
        await llm.chat("test prompt")


async def test_chat_raises_value_error_on_empty_prompt(llm: AzureOpenAILLM) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        await llm.chat("")


async def test_chat_raises_on_empty_choices(llm: AzureOpenAILLM) -> None:
    llm._client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        return_value=MagicMock(choices=[])
    )
    with pytest.raises(ValueError, match="no choices"):
        await llm.chat("test")


# ── embed ─────────────────────────────────────────────────────────────────────


def test_cloud_llm_has_no_embed(llm: AzureOpenAILLM) -> None:
    # Embeddings stay local: the cloud backend does not offer embed() at all.
    assert not hasattr(llm, "embed")


async def test_chat_json_output_sets_response_format(llm: AzureOpenAILLM) -> None:
    create = AsyncMock(return_value=_make_completion("{}"))
    llm._client.chat.completions.create = create  # type: ignore[method-assign]

    await llm.chat("p", json_output=True)
    await llm.chat("p")

    assert create.await_args_list[0].kwargs["response_format"] == {
        "type": "json_object"
    }
    assert "response_format" not in create.await_args_list[1].kwargs


async def test_azure_json_output_has_no_reasoning_effort(llm: AzureOpenAILLM) -> None:
    create = AsyncMock(return_value=_make_completion("{}"))
    llm._client.chat.completions.create = create  # type: ignore[method-assign]

    await llm.chat("p", json_output=True)

    assert "reasoning_effort" not in create.await_args.kwargs


async def test_groq_uses_low_reasoning_effort_on_every_call() -> None:
    # gpt-oss on Groq can spend its whole 2048-token reply on reasoning and
    # return empty content, in JSON mode and in plain answers alike.
    with patch("agentic_rag.llm.openai_compat.openai.AsyncOpenAI") as mock_cls:
        groq = GroqLLM(GroqConfig(api_key="k", model="openai/gpt-oss-20b"))
    create = AsyncMock(return_value=_make_completion("{}"))
    mock_cls.return_value.chat.completions.create = create

    await groq.chat("p", json_output=True)
    await groq.chat("p")

    assert [c.kwargs["reasoning_effort"] for c in create.await_args_list] == [
        "low",
        "low",
    ]
