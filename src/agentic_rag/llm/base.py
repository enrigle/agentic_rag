"""Abstract base class for all LLM backends."""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Abstract base for LLM backends.

    All methods are coroutines and must be awaited.
    Implementations should raise RuntimeError on backend/network failure
    and ValueError on invalid or empty responses.
    """

    @abstractmethod
    async def chat(self, prompt: str) -> str:
        """Send prompt to the chat model and return the reply as a string.

        Raises:
            ValueError: If the model returns an empty or null response.
            RuntimeError: On backend or network failure.
        """
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Return the embedding vector for text.

        Raises:
            ValueError: If the model returns no embeddings.
            RuntimeError: On backend or network failure.
        """
        ...
