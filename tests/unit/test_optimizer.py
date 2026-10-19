import pytest
import json
from pathlib import Path
from agentic_rag.feedback.store import FeedbackEntry
from agentic_rag.feedback.optimizer import (
    OptimizationResult,
    apply_optimization,
    get_few_shot_examples,
    get_kb_gaps,
    set_min_similarity,
    tune_retrieval_params,
)


def _entry(
    rating: int,
    similarity: float | None,
    category: str = "",
    query: str = "q",
    answer: str = "a",
) -> FeedbackEntry:
    # top_score is a cross-encoder logit in production; tuning must ignore it.
    return FeedbackEntry(
        query=query,
        answer=answer,
        sources=[],
        top_score=-5.0,
        rating=rating,
        category=category,
        top_similarity=similarity,
    )


def test_tune_ignores_entries_without_similarity() -> None:
    # Web answers and legacy rows have no cosine similarity; they carry no signal.
    entries = [
        _entry(1, 0.8),
        _entry(1, None),
        _entry(-1, 0.2),
        _entry(-1, None),
    ]
    assert tune_retrieval_params(entries) == 0.5


def test_tune_returns_none_when_only_legacy_rows() -> None:
    entries = [_entry(1, None), _entry(-1, None)]
    assert tune_retrieval_params(entries) is None


def test_tune_suggests_midpoint() -> None:
    entries = [
        _entry(1, 0.8),
        _entry(1, 0.7),
        _entry(1, 0.75),
        _entry(-1, 0.1),
        _entry(-1, 0.2),
        _entry(-1, 0.15),
    ]
    result = tune_retrieval_params(entries)
    assert result is not None
    assert 0.1 < result < 0.8


def test_tune_no_signal_when_scores_equal() -> None:
    entries = [_entry(1, 0.5), _entry(-1, 0.5)]
    assert tune_retrieval_params(entries) is None


def test_tune_returns_none_without_both_polarities() -> None:
    assert tune_retrieval_params([_entry(1, 0.8)]) is None
    assert tune_retrieval_params([_entry(-1, 0.1)]) is None
    assert tune_retrieval_params([]) is None


def test_get_few_shot_examples_returns_3_most_recent() -> None:
    entries = [_entry(1, 0.8, query=f"q{i}", answer=f"a{i}") for i in range(5)]
    examples = get_few_shot_examples(entries)
    assert len(examples) == 3
    assert examples[0]["query"] == "q4"
    assert examples[1]["query"] == "q3"
    assert examples[2]["query"] == "q2"


def test_get_few_shot_examples_ignores_thumbs_down() -> None:
    entries = [_entry(-1, 0.1, query="bad")]
    assert get_few_shot_examples(entries) == []


def test_get_kb_gaps_returns_missing_content_queries() -> None:
    entries = [
        _entry(-1, 0.1, category="missing_content", query="What is X?"),
        _entry(-1, 0.1, category="retrieval_miss", query="How to do Y?"),
        _entry(1, 0.8, query="Good Q"),
    ]
    assert get_kb_gaps(entries) == ["What is X?"]


def test_apply_optimization_writes_feedback_config(tmp_path: Path) -> None:
    config_path = tmp_path / "default.yaml"
    config_path.write_text("retriever:\n  min_similarity: 0.35\n")
    fb_config_path = tmp_path / "feedback_config.json"
    entries = [
        _entry(1, 0.8, query="good q", answer="good a"),
        _entry(-1, 0.1, category="missing_content", query="missing q"),
    ]
    result = apply_optimization(
        entries, config_path=config_path, feedback_config_path=fb_config_path
    )
    assert isinstance(result, OptimizationResult)
    assert fb_config_path.exists()
    fc = json.loads(fb_config_path.read_text())
    assert "few_shot_examples" in fc
    assert result.kb_gaps == ["missing q"]


def test_apply_optimization_updates_yaml_when_signal(tmp_path: Path) -> None:
    config_path = tmp_path / "default.yaml"
    config_path.write_text("retriever:\n  min_similarity: 0.35\n")
    fb_config_path = tmp_path / "feedback_config.json"
    entries = [
        _entry(1, 0.8),
        _entry(1, 0.75),
        _entry(-1, 0.1),
        _entry(-1, 0.15),
    ]
    result = apply_optimization(
        entries, config_path=config_path, feedback_config_path=fb_config_path
    )
    if result.new_min_similarity is not None:
        import yaml

        raw = yaml.safe_load(config_path.read_text())
        assert raw["retriever"]["min_similarity"] == result.new_min_similarity


def test_apply_optimization_yaml_write_failure_clears_min_sim(tmp_path: Path) -> None:
    # config_path parent doesn't exist — read_text() raises FileNotFoundError
    config_path = tmp_path / "nonexistent_dir" / "default.yaml"
    fb_config_path = tmp_path / "feedback_config.json"
    entries = [_entry(1, 0.8), _entry(-1, 0.1)]
    result = apply_optimization(
        entries, config_path=config_path, feedback_config_path=fb_config_path
    )
    assert result.new_min_similarity is None


_YAML = """\
retriever:
  # cosine cutoff
  min_similarity: 0.50  # keep me
  top_n: 20
redis:
  url: "redis://x"
"""


def test_set_min_similarity_keeps_comments_and_other_keys() -> None:
    out = set_min_similarity(_YAML, 0.42)
    assert "# cosine cutoff" in out
    assert "# keep me" in out
    assert "min_similarity: 0.42" in out
    import yaml

    raw = yaml.safe_load(out)
    assert raw == {
        "retriever": {"min_similarity": 0.42, "top_n": 20},
        "redis": {"url": "redis://x"},
    }


def test_set_min_similarity_adds_missing_key_and_section() -> None:
    import yaml

    no_key = set_min_similarity("retriever:\n  top_n: 20\n", 0.4)
    assert yaml.safe_load(no_key)["retriever"] == {"min_similarity": 0.4, "top_n": 20}
    no_section = set_min_similarity("redis:\n  url: x\n", 0.4)
    assert yaml.safe_load(no_section)["retriever"] == {"min_similarity": 0.4}


def test_set_min_similarity_ignores_same_key_in_other_section() -> None:
    import yaml

    text = "redis:\n  min_similarity: 0.9\nretriever:\n  min_similarity: 0.5\n"
    raw = yaml.safe_load(set_min_similarity(text, 0.3))
    assert raw["redis"]["min_similarity"] == 0.9
    assert raw["retriever"]["min_similarity"] == 0.3


def test_apply_optimization_writes_active_config_file(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    # Must edit the file load_config() reads (RAG_CONFIG_PATH), not a fixed path.
    active = tmp_path / "docker.yaml"
    active.write_text("retriever:\n  min_similarity: 0.35\n")
    monkeypatch.setenv("RAG_CONFIG_PATH", str(active))
    entries = [_entry(1, 0.8), _entry(1, 0.75), _entry(-1, 0.1), _entry(-1, 0.15)]
    result = apply_optimization(
        entries, feedback_config_path=tmp_path / "feedback_config.json"
    )
    assert result.new_min_similarity is not None
    assert f"min_similarity: {result.new_min_similarity}" in active.read_text()
