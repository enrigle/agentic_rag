import json
from pathlib import Path
from agentic_rag.feedback.store import FeedbackEntry
from agentic_rag.feedback.optimizer import (
    OptimizationResult,
    apply_optimization,
    get_few_shot_examples,
    get_kb_gaps,
    tune_retrieval_params,
)


def _entry(
    rating: int,
    top_score: float,
    category: str = "",
    query: str = "q",
    answer: str = "a",
) -> FeedbackEntry:
    return FeedbackEntry(
        query=query,
        answer=answer,
        sources=[],
        top_score=top_score,
        rating=rating,
        category=category,
    )


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
