"""Azure OpenAI-backed LLM implementation (synthesis only)."""

from __future__ import annotations

import logging
import os

import openai

from agentic_rag.config import AzureOpenAIConfig
from agentic_rag.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class AzureOpenAILLM(BaseLLM):
    """Concrete LLM backend using the Azure OpenAI async client.

    Only ``chat()`` is supported; ``embed()`` is intentionally not implemented
    because embeddings remain local via OllamaLLM.
    """

    def __init__(self, config: AzureOpenAIConfig) -> None:
        if not config.endpoint:
            raise ValueError("AzureOpenAIConfig.endpoint must not be empty")

        resolved_api_key: str | None = config.api_key or os.environ.get(
            "AZURE_OPENAI_API_KEY"
        )
        if not resolved_api_key:
            raise ValueError(
                "AzureOpenAIConfig.api_key or AZURE_OPENAI_API_KEY env var must be set"
            )

        self._config = config
        self._client = openai.AsyncAzureOpenAI(
            azure_endpoint=config.endpoint,
            api_key=resolved_api_key,
            api_version=config.api_version,
        )
        logger.debug(
            "AzureOpenAILLM initialised: deployment=%s api_version=%s endpoint=%s",
            config.deployment,
            config.api_version,
            config.endpoint,
        )

    async def chat(self, prompt: str) -> str:
        """Send a user prompt to the Azure OpenAI chat model and return the reply text."""
        if not prompt:
            raise ValueError("prompt must be a non-empty string")
        logger.debug(
            "chat() → deployment=%s prompt_len=%d",
            self._config.deployment,
            len(prompt),
        )
        try:
            response = await self._client.chat.completions.create(
                model=self._config.deployment,
                messages=[{"role": "user", "content": prompt}],
            )
        except openai.APIError as exc:
            raise RuntimeError(f"Azure OpenAI chat failed: {exc}") from exc

        if not response.choices:
            raise ValueError(
                f"Azure OpenAI returned no choices for deployment={self._config.deployment!r}"
            )
        content = response.choices[0].message.content
        if not content:
            raise ValueError(
                f"Azure OpenAI returned empty content for deployment={self._config.deployment!r}"
            )
        logger.debug("chat() ← content_len=%d", len(content))
        return content

    async def embed(self, text: str) -> list[float]:
        """Not implemented — embeddings stay local via OllamaLLM."""
        raise NotImplementedError("embeddings stay local — use OllamaLLM for embed()")
