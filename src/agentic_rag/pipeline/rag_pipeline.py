"""Factory for wiring a PipelineCoordinator from config."""

from __future__ import annotations

from agentic_rag.config import RAGConfig, load_config
from agentic_rag.llm.ollama import OllamaLLM
from agentic_rag.pipeline.coordinator import PipelineCoordinator
from agentic_rag.pipeline.memory import ConversationMemory
from agentic_rag.pipeline.sources import RAGSource, WebSource
from agentic_rag.pipeline.synthesizer import Synthesizer
from agentic_rag.retrieval.bm25 import BM25Retriever
from agentic_rag.retrieval.chroma import ChromaVectorStore
from agentic_rag.retrieval.hybrid import HybridRetriever
from agentic_rag.retrieval.reranker import CrossEncoderReranker


def create_pipeline(config: RAGConfig | None = None) -> PipelineCoordinator:
    """Wire concrete Ollama + Chroma + BM25 implementations from config."""
    if config is None:
        config = load_config()

    llm = OllamaLLM(config.llm)
    hybrid = HybridRetriever(ChromaVectorStore(config), BM25Retriever(config), config)

    return PipelineCoordinator(
        sources=[RAGSource(llm, hybrid), WebSource()],
        reranker=CrossEncoderReranker(
            model=config.retriever.reranker_model,
            top_k=config.retriever.reranker_top_k,
        ),
        synthesizer=Synthesizer(llm),
        memory=ConversationMemory(),
        threshold=config.retriever.web_search_fallback_score,
        max_tool_calls=config.max_tool_calls,
    )
