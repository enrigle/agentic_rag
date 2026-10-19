"""make_synth_llm must return a chat-capable model for every backend combo."""

from __future__ import annotations

import pytest

from agentic_rag.config import RAGConfig
from agentic_rag.llm.ollama import OllamaLLM
from agentic_rag.pipeline.rag_pipeline import make_synth_llm


def test_st_embeddings_without_cloud_synthesize_with_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: synthesis used the embed model, and the sentence-transformers
    # one raises NotImplementedError on chat().
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    cfg = RAGConfig(embed_backend="sentence_transformers")
    cfg.groq.api_key = ""
    cfg.azure_openai.endpoint = ""
    assert isinstance(make_synth_llm(cfg), OllamaLLM)
