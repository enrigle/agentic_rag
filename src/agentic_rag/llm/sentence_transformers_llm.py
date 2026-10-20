"""sentence-transformers embedding backend (CPU-safe, no Ollama dependency)."""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SentenceTransformersLLM:
    """Embedder backed by sentence-transformers (CPU inference). Has no chat()."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model = SentenceTransformer(model_name)
        logger.debug("SentenceTransformersLLM initialised: model=%s", model_name)

    async def embed(self, text: str) -> list[float]:
        if not text:
            raise ValueError("text must be non-empty")
        vector: list[float] = self._model.encode(
            text, normalize_embeddings=True
        ).tolist()
        logger.debug("embed() ← dim=%d", len(vector))
        return vector
