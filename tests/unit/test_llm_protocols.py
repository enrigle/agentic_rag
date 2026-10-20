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
