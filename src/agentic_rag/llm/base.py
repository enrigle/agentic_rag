"""Abstract base class for all LLM backends."""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    async def chat(self, prompt: str) -> str: ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...
