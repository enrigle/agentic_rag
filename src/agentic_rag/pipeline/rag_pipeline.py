"""Factory for wiring a PipelineCoordinator from config."""

from __future__ import annotations

import logging

from agentic_rag.cache.semantic_cache import SemanticCache
from agentic_rag.config import RAGConfig, load_config
from agentic_rag.llm.base import BaseLLM
from agentic_rag.llm.ollama import OllamaLLM
from agentic_rag.llm.openai_compat import AzureOpenAILLM, GroqLLM
from agentic_rag.llm.sentence_transformers_llm import SentenceTransformersLLM
from agentic_rag.pipeline.coordinator import PipelineCoordinator
from agentic_rag.pipeline.memory import ConversationMemory
from agentic_rag.pipeline.sources import RAGSource, WebSource
from agentic_rag.pipeline.synthesizer import Synthesizer
from agentic_rag.retrieval.bm25 import BM25Retriever
from agentic_rag.retrieval.chroma import ChromaVectorStore
from agentic_rag.retrieval.hybrid import HybridRetriever
from agentic_rag.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


def create_pipeline(config: RAGConfig | None = None) -> PipelineCoordinator:
    """Wire PipelineCoordinator from config.

    - Embeddings always use OllamaLLM (local, private).
    - Synthesis uses GroqLLM when GROQ_API_KEY is set, then AzureOpenAILLM when
      azure_openai.endpoint is set, otherwise falls back to OllamaLLM.
    - SemanticCache is always attached; it fails open if Redis is unreachable.
    """
    if config is None:
        config = load_config()

    if config.embed_backend == "sentence_transformers":
        llm: BaseLLM = SentenceTransformersLLM(config.llm.embed_model)
        logger.info("Embeddings: SentenceTransformersLLM model=%s", config.llm.embed_model)
    else:
        llm = OllamaLLM(config.llm)
        logger.info("Embeddings: OllamaLLM (model=%s)", config.llm.embed_model)
    hybrid = HybridRetriever(ChromaVectorStore(config), BM25Retriever(config), config)

    synth_llm: BaseLLM = llm
    if config.groq.is_configured():
        synth_llm = GroqLLM(config.groq)
        logger.info("Synthesis: GroqLLM (model=%s)", config.groq.model)
    elif config.azure_openai.is_configured():
        synth_llm = AzureOpenAILLM(config.azure_openai)
        logger.info("Synthesis: AzureOpenAILLM deployment=%s", config.azure_openai.deployment)

    cache = SemanticCache(config.redis, llm)

    return PipelineCoordinator(
        sources=[RAGSource(llm, hybrid), WebSource()],
        reranker=CrossEncoderReranker(
            model=config.retriever.reranker_model,
            top_k=config.retriever.reranker_top_k,
        ),
        synthesizer=Synthesizer(synth_llm),
        memory=ConversationMemory(),
        max_tool_calls=config.max_tool_calls,
        cache=cache,
    )
