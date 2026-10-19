"""Factory for wiring a PipelineCoordinator from config."""

from __future__ import annotations

import logging

from agentic_rag.cache.semantic_cache import SemanticCache
from agentic_rag.config import RAGConfig, load_config
from agentic_rag.feedback.optimizer import FEEDBACK_CONFIG_PATH
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


def make_embed_llm(config: RAGConfig) -> BaseLLM:
    """Embedding LLM selected by config.embed_backend.

    Shared by the pipeline and the in-app ingest so both sides always write
    and query vectors in the same embedding space.
    """
    if config.embed_backend == "sentence_transformers":
        llm: BaseLLM = SentenceTransformersLLM(config.llm.embed_model)
        logger.info(
            "Embeddings: SentenceTransformersLLM model=%s", config.llm.embed_model
        )
    else:
        llm = OllamaLLM(config.llm)
        logger.info("Embeddings: OllamaLLM (model=%s)", config.llm.embed_model)
    return llm


def make_synth_llm(config: RAGConfig) -> BaseLLM:
    """Answer-writing LLM: Groq if configured, then Azure OpenAI, else local Ollama."""
    if config.groq.is_configured():
        logger.info("Synthesis: GroqLLM (model=%s)", config.groq.model)
        return GroqLLM(config.groq, timeout=config.llm.request_timeout)
    if config.azure_openai.is_configured():
        logger.info(
            "Synthesis: AzureOpenAILLM deployment=%s", config.azure_openai.deployment
        )
        return AzureOpenAILLM(config.azure_openai, timeout=config.llm.request_timeout)
    return OllamaLLM(config.llm)


def create_pipeline(config: RAGConfig | None = None) -> PipelineCoordinator:
    """Wire PipelineCoordinator from config.

    - Embeddings use make_embed_llm (embed_backend: Ollama or sentence-transformers).
    - Synthesis uses GroqLLM when GROQ_API_KEY is set, then AzureOpenAILLM when
      azure_openai.endpoint is set, otherwise falls back to OllamaLLM.
    - SemanticCache is always attached; it fails open if Redis is unreachable.
    """
    if config is None:
        config = load_config()

    llm = make_embed_llm(config)
    hybrid = HybridRetriever(ChromaVectorStore(config), BM25Retriever(config), config)

    synth_llm = make_synth_llm(config)

    cache = SemanticCache(config.redis, llm)

    return PipelineCoordinator(
        sources=[RAGSource(llm, hybrid), WebSource()],
        reranker=CrossEncoderReranker(
            model=config.retriever.reranker_model,
            top_k=config.retriever.reranker_top_k,
            min_score=config.retriever.reranker_min_score,
        ),
        synthesizer=Synthesizer(synth_llm, few_shot_path=FEEDBACK_CONFIG_PATH),
        memory=ConversationMemory(),
        max_tool_calls=config.max_tool_calls,
        cache=cache,
        embed_llm=llm,
    )
