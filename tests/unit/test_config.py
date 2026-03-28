"""Unit tests for agentic_rag.config.load_config."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rag.config import RAGConfig, load_config


def test_load_config_no_path_returns_defaults() -> None:
    """load_config() with no args returns RAGConfig with default values."""
    # The default config file may or may not exist; either way we get a RAGConfig
    cfg = load_config()
    assert isinstance(cfg, RAGConfig)


def test_load_config_nonexistent_path_returns_defaults(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    cfg = load_config(path=missing)
    assert isinstance(cfg, RAGConfig)
    # Should match defaults
    defaults = RAGConfig()
    assert cfg.chroma_path == defaults.chroma_path
    assert cfg.collection_name == defaults.collection_name


def test_load_config_valid_yaml_overrides_values(tmp_path: Path) -> None:
    yaml_content = """\
chroma_path: ./custom_chroma
bm25_path: ./custom_bm25
collection_name: my_collection
max_tool_calls: 10
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content)

    cfg = load_config(path=config_file)
    assert cfg.chroma_path == "./custom_chroma"
    assert cfg.bm25_path == "./custom_bm25"
    assert cfg.collection_name == "my_collection"
    assert cfg.max_tool_calls == 10


def test_load_config_unknown_keys_do_not_crash(tmp_path: Path) -> None:
    yaml_content = """\
chroma_path: ./chroma
unknown_top_level_key: some_value
another_unknown: 42
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content)

    cfg = load_config(path=config_file)
    assert isinstance(cfg, RAGConfig)
    assert cfg.chroma_path == "./chroma"


def test_load_config_nested_llm_model_parsed(tmp_path: Path) -> None:
    yaml_content = """\
llm:
  model: mistral
  embed_model: custom-embed
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content)

    cfg = load_config(path=config_file)
    assert cfg.llm.model == "mistral"
    assert cfg.llm.embed_model == "custom-embed"


def test_load_config_nested_llm_unknown_keys_ignored(tmp_path: Path) -> None:
    yaml_content = """\
llm:
  model: phi3
  unknown_llm_key: ignored
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content)

    cfg = load_config(path=config_file)
    assert cfg.llm.model == "phi3"


def test_load_config_retriever_partial_override(tmp_path: Path) -> None:
    yaml_content = """\
retriever:
  top_n: 7
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content)

    cfg = load_config(path=config_file)
    assert cfg.retriever.top_n == 7
    # Other retriever fields should keep defaults
    from agentic_rag.config import RetrieverConfig
    defaults = RetrieverConfig()
    assert cfg.retriever.min_similarity == defaults.min_similarity


def test_load_config_empty_yaml_returns_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("")

    cfg = load_config(path=config_file)
    assert isinstance(cfg, RAGConfig)
    defaults = RAGConfig()
    assert cfg.chroma_path == defaults.chroma_path
