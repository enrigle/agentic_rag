"""LLM backends for OpenAI-compatible APIs (Azure OpenAI and Groq)."""

from __future__ import annotations

import logging
import os

import openai

from agentic_rag.config import AzureOpenAIConfig, GroqConfig
from agentic_rag.llm.base import BaseLLM

logger = logging.getLogger(__name__)

_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class OpenAICompatLLM(BaseLLM):
    """Shared chat logic for any OpenAI-compatible provider."""

    def __init__(
        self,
        client: openai.AsyncOpenAI,
        model: str,
        provider: str,
    ) -> None:
        self._client = client
        self._model = model
        self._provider = provider
        logger.debug("%s initialised: model=%s", provider, model)

    async def chat(self, prompt: str) -> str:
        if not prompt:
            raise ValueError("prompt must be a non-empty string")
        logger.debug(
            "chat() → %s model=%s prompt_len=%d",
            self._provider,
            self._model,
            len(prompt),
        )
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
        except openai.APIError as exc:
            raise RuntimeError(f"{self._provider} chat failed: {exc}") from exc
        if not response.choices:
            raise ValueError(
                f"{self._provider} returned no choices for model={self._model!r}"
            )
        content = response.choices[0].message.content
        if not content:
            raise ValueError(
                f"{self._provider} returned empty content for model={self._model!r}"
            )
        logger.debug("chat() ← content_len=%d", len(content))
        return content

    async def embed(self, _text: str) -> list[float]:
        raise NotImplementedError("embeddings stay local — use OllamaLLM for embed()")


class AzureOpenAILLM(OpenAICompatLLM):
    """Synthesis-only LLM backed by Azure OpenAI."""

    def __init__(self, config: AzureOpenAIConfig) -> None:
        if not config.endpoint:
            raise ValueError("AzureOpenAIConfig.endpoint must not be empty")
        resolved_key = config.api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "AzureOpenAIConfig.api_key or AZURE_OPENAI_API_KEY env var must be set"
            )
        super().__init__(
            client=openai.AsyncAzureOpenAI(
                azure_endpoint=config.endpoint,
                api_key=resolved_key,
                api_version=config.api_version,
            ),
            model=config.deployment,
            provider="Azure OpenAI",
        )


class GroqLLM(OpenAICompatLLM):
    """Synthesis-only LLM backed by Groq (via OpenAI-compatible API)."""

    def __init__(self, config: GroqConfig) -> None:
        resolved_key = config.api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError("GroqConfig.api_key or GROQ_API_KEY env var must be set")
        super().__init__(
            client=openai.AsyncOpenAI(base_url=_GROQ_BASE_URL, api_key=resolved_key),
            model=config.model,
            provider="Groq",
        )
