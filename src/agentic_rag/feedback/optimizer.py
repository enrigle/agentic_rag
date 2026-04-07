"""Optimization functions: retrieval param tuning, few-shot selection, KB gap reporting."""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from agentic_rag.feedback.store import FeedbackEntry

logger = logging.getLogger(__name__)

_MIN_SCORE_SEPARATION = 0.005  # ignore signal if medians are this close


@dataclass
class OptimizationResult:
    new_min_similarity: float | None  # None = no change suggested
    few_shot_count: int
    kb_gaps: list[str]


def tune_retrieval_params(entries: list[FeedbackEntry]) -> float | None:
    """Suggest a new min_similarity based on score distributions.

    Returns the midpoint between median thumbs-up and thumbs-down top_score,
    or None if there is insufficient signal.
    """
    up = [e.top_score for e in entries if e.rating == 1 and e.top_score > 0]
    down = [e.top_score for e in entries if e.rating == -1 and e.top_score > 0]
    if not up or not down:
        return None
    median_up = statistics.median(up)
    median_down = statistics.median(down)
    if abs(median_up - median_down) < _MIN_SCORE_SEPARATION:
        return None
    suggested = (median_up + median_down) / 2
    return round(suggested, 4)


def get_few_shot_examples(entries: list[FeedbackEntry]) -> list[dict[str, str]]:
    """Return the 3 most recent thumbs-up (query, answer) pairs, newest first."""
    thumbs_up = [e for e in reversed(entries) if e.rating == 1]
    return [{"query": e.query, "answer": e.answer} for e in thumbs_up[:3]]


def get_kb_gaps(entries: list[FeedbackEntry]) -> list[str]:
    """Return queries where the LLM judge found missing content."""
    return [e.query for e in entries if e.category == "missing_content"]


def apply_optimization(
    entries: list[FeedbackEntry],
    config_path: Path = Path("config/default.yaml"),
    feedback_config_path: Path = Path("feedback_config.json"),
) -> OptimizationResult:
    """Run all three optimizations and persist results to disk."""
    # 1. Retrieval param tuning
    new_min_sim = tune_retrieval_params(entries)
    if new_min_sim is not None:
        try:
            raw: dict = yaml.safe_load(config_path.read_text()) or {}
            raw.setdefault("retriever", {})["min_similarity"] = new_min_sim
            config_path.write_text(yaml.dump(raw, default_flow_style=False))
            logger.info("optimizer: updated min_similarity to %s", new_min_sim)
        except Exception as exc:
            logger.warning("optimizer: failed to update config: %s", exc)
            new_min_sim = None

    # 2. Few-shot examples
    examples = get_few_shot_examples(entries)
    feedback_config_path.write_text(
        json.dumps({"few_shot_examples": examples}, indent=2, ensure_ascii=False)
    )
    logger.info("optimizer: wrote %d few-shot examples", len(examples))

    # 3. KB gaps
    gaps = get_kb_gaps(entries)

    return OptimizationResult(
        new_min_similarity=new_min_sim,
        few_shot_count=len(examples),
        kb_gaps=gaps,
    )
