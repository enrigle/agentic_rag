"""Ollama-backed LLM implementation."""

import logging

import ollama

from agentic_rag.config import LLMConfig
from agentic_rag.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """Concrete LLM backend using the Ollama async client."""

    def __init__(self, config: LLMConfig) -> None:
        if not config.base_url:
            raise ValueError("LLMConfig.base_url must not be empty")
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
        if not prompt:
            raise ValueError("prompt must be a non-empty string")
        logger.debug("chat() → model=%s prompt_len=%d", self._config.model, len(prompt))
        try:
            response = await self._client.chat(
                model=self._config.model,
                messages=[{"role": "user", "content": prompt}],
            )
        except ollama.ResponseError as exc:
            raise RuntimeError(
                f"Ollama chat failed (model={self._config.model!r}): {exc}"
            ) from exc
        content = response.message.content
        if not content:
            raise ValueError(
                f"Ollama returned empty content for model={self._config.model!r}"
            )
        logger.debug("chat() ← content_len=%d", len(content))
        return content

    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text* using the embed model."""
        if not text:
            raise ValueError("text must be a non-empty string")
        logger.debug(
            "embed() → model=%s text_len=%d", self._config.embed_model, len(text)
        )
        try:
            response = await self._client.embed(
                model=self._config.embed_model, input=text
            )
        except ollama.ResponseError as exc:
            raise RuntimeError(
                f"Ollama embed failed (model={self._config.embed_model!r}): {exc}"
            ) from exc
        embeddings: list[list[float]] = response.embeddings or []  # type: ignore[assignment]
        if not embeddings or not embeddings[0]:
            raise ValueError(
                f"Ollama returned no embeddings for model={self._config.embed_model!r}"
            )
        vector: list[float] = embeddings[0]
        logger.debug("embed() ← dim=%d", len(vector))
        return vector
