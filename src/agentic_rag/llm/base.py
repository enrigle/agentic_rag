"""LLM interfaces: one Protocol per job, so a backend implements only what it can do."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Turns text into a vector (Ollama, sentence-transformers).

    Implementations raise ValueError on empty input/output and RuntimeError on
    backend or network failure.
    """

    async def embed(self, text: str) -> list[float]: ...


@runtime_checkable
class ChatModel(Protocol):
    """Answers a prompt with text (Ollama, Groq, Azure OpenAI).

    Implementations raise ValueError on an empty reply and RuntimeError on
    backend or network failure.
    """

    async def chat(self, prompt: str) -> str: ...
