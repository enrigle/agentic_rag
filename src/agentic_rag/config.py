"""Configuration dataclasses and YAML loader for agentic_rag."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import yaml  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# Package root is two levels up from this file: src/agentic_rag/config.py
_PACKAGE_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_CONFIG_PATH = _PACKAGE_ROOT / "config" / "default.yaml"


@dataclass
class LLMConfig:
    model: str = "llama3.2"
    embed_model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"


@dataclass
class RetrieverConfig:
    min_similarity: float = 0.35
    top_n: int = 20
    rrf_k: int = 60
    bm25_top_k: int = 10
    web_search_fallback_score: float = 0.4
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"
    reranker_top_k: int = 5


@dataclass
class IngestionConfig:
    chunk_size: int = 800
    chunk_overlap: int = 100
    vision_model: str = "llava"


@dataclass
class RAGConfig:
    chroma_path: str = "./chroma_db"
    bm25_path: str = "./bm25_index"
    collection_name: str = "notion_kb"
    max_tool_calls: int = 5
    llm: LLMConfig = field(default_factory=LLMConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)


_DC = TypeVar("_DC")


def _parse_sub(cls: type[_DC], data: dict[str, Any]) -> _DC:
    """Construct a dataclass from a dict, ignoring unknown keys."""
    known = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
    return cls(**{k: v for k, v in data.items() if k in known})


def load_config(path: Path | None = None) -> RAGConfig:
    """Load RAGConfig from YAML, falling back to defaults for missing keys.

    Args:
        path: Path to a YAML config file. If None, uses
              ``config/default.yaml`` relative to the package root.

    Returns:
        A fully populated RAGConfig instance.
    """
    resolved = path if path is not None else _DEFAULT_CONFIG_PATH

    if not resolved.exists():
        logger.warning("Config file not found at %s; using all defaults.", resolved)
        return RAGConfig()

    try:
        raw: dict[str, Any] = yaml.safe_load(resolved.read_text()) or {}
    except yaml.YAMLError as exc:
        logger.warning(
            "Failed to parse config file %s: %s; using defaults.", resolved, exc
        )
        return RAGConfig()

    llm_cfg = _parse_sub(LLMConfig, raw.get("llm") or {})
    retriever_cfg = _parse_sub(RetrieverConfig, raw.get("retriever") or {})
    ingestion_cfg = _parse_sub(IngestionConfig, raw.get("ingestion") or {})

    top_level_keys = {"chroma_path", "bm25_path", "collection_name", "max_tool_calls"}
    top_level = {k: v for k, v in raw.items() if k in top_level_keys}

    try:
        return RAGConfig(
            **top_level,
            llm=llm_cfg,
            retriever=retriever_cfg,
            ingestion=ingestion_cfg,
        )
    except TypeError as exc:
        logger.warning(
            "Invalid top-level config values in %s: %s; using defaults.", resolved, exc
        )
        return RAGConfig(
            llm=llm_cfg,
            retriever=retriever_cfg,
            ingestion=ingestion_cfg,
        )
