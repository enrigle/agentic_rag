"""Each backend implements exactly the jobs it can do (no NotImplementedError stubs)."""

from __future__ import annotations

from agentic_rag.llm.base import ChatModel, Embedder
from agentic_rag.llm.ollama import OllamaLLM
from agentic_rag.llm.openai_compat import AzureOpenAILLM, GroqLLM
from agentic_rag.llm.sentence_transformers_llm import SentenceTransformersLLM


def test_ollama_embeds_and_chats() -> None:
    assert issubclass(OllamaLLM, Embedder)
    assert issubclass(OllamaLLM, ChatModel)


def test_sentence_transformers_only_embeds() -> None:
    assert issubclass(SentenceTransformersLLM, Embedder)
    assert not issubclass(SentenceTransformersLLM, ChatModel)


def test_cloud_llms_only_chat() -> None:
    for cls in (GroqLLM, AzureOpenAILLM):
        assert issubclass(cls, ChatModel)
        assert not issubclass(cls, Embedder)  # embeddings stay local


async def test_ollama_chat_passes_format_json_only_when_requested() -> None:
    from unittest.mock import AsyncMock, MagicMock

    from agentic_rag.config import LLMConfig

    llm = OllamaLLM(LLMConfig())
    chat = AsyncMock(return_value=MagicMock(message=MagicMock(content="{}")))
    llm._client.chat = chat  # type: ignore[method-assign]

    await llm.chat("p", json_output=True)
    await llm.chat("p")

    assert chat.await_args_list[0].kwargs["format"] == "json"
    assert "format" not in chat.await_args_list[1].kwargs
