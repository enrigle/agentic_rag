"""Ollama-backed LLM implementation."""

import logging

import ollama

from agentic_rag.config import LLMConfig
from agentic_rag.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """Concrete LLM backend using the Ollama async client."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._client = ollama.AsyncClient(host=config.base_url)
        logger.debug(
            "OllamaLLM initialised: model=%s embed_model=%s base_url=%s",
            config.model,
            config.embed_model,
            config.base_url,
        )

    async def chat(self, prompt: str) -> str:
        """Send a user prompt to the chat model and return the reply text."""
        logger.debug("chat() → model=%s prompt_len=%d", self._config.model, len(prompt))
        response = await self._client.chat(
            model=self._config.model,
            messages=[{"role": "user", "content": prompt}],
        )
        content: str = response.message.content  # type: ignore[assignment]
        logger.debug("chat() ← content_len=%d", len(content))
        return content

    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text* using the embed model."""
        logger.debug(
            "embed() → model=%s text_len=%d", self._config.embed_model, len(text)
        )
        response = await self._client.embed(
            model=self._config.embed_model,
            input=text,
        )
        vector: list[float] = response["embeddings"][0]
        logger.debug("embed() ← dim=%d", len(vector))
        return vector
