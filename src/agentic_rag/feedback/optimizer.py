"""Optimization functions: retrieval param tuning, few-shot selection, KB gap reporting."""

from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import yaml

from agentic_rag.config import resolve_config_path
from agentic_rag.feedback.store import FeedbackEntry

logger = logging.getLogger(__name__)

_MIN_SCORE_SEPARATION = 0.05  # ignore signal if medians are this close
FEEDBACK_CONFIG_PATH = Path("data/feedback_config.json")  # read by Synthesizer


@dataclass
class OptimizationResult:
    new_min_similarity: float | None  # None = no change suggested
    few_shot_count: int
    kb_gaps: list[str]


def tune_retrieval_params(entries: list[FeedbackEntry]) -> float | None:
    """Suggest a new min_similarity based on score distributions.

    Returns the midpoint between median thumbs-up and thumbs-down top_similarity
    (cosine, same scale as min_similarity), or None if there is insufficient signal.
    Entries without a similarity (web answers, legacy rows) are skipped: their
    top_score is a cross-encoder logit or web score, not a cosine.
    """
    up = [
        e.top_similarity
        for e in entries
        if e.rating == 1 and e.top_similarity is not None
    ]
    down = [
        e.top_similarity
        for e in entries
        if e.rating == -1 and e.top_similarity is not None
    ]
    if not up or not down:
        return None
    median_up = statistics.median(up)
    median_down = statistics.median(down)
    if abs(median_up - median_down) < _MIN_SCORE_SEPARATION:
        return None
    suggested = (median_up + median_down) / 2
    return round(suggested, 4)


_SECTION = re.compile(r"^retriever:\s*(#.*)?$")
_KEY = re.compile(r"^(\s+min_similarity:\s*)[^#\n]*?(\s*#.*)?$")


def set_min_similarity(yaml_text: str, value: float) -> str:
    """Set retriever.min_similarity by editing one line, keeping comments and layout.

    yaml.dump would rewrite the whole file and drop every comment. The result is
    re-parsed and checked, so an unexpected layout fails loudly instead of
    writing a broken config.
    """
    lines = yaml_text.splitlines(keepends=True)
    start = next((i for i, ln in enumerate(lines) if _SECTION.match(ln)), None)
    if start is None:
        sep = "" if not yaml_text or yaml_text.endswith("\n") else "\n"
        out = f"{yaml_text}{sep}retriever:\n  min_similarity: {value}\n"
    else:
        end = next(
            (
                i
                for i in range(start + 1, len(lines))
                if lines[i][:1] not in ("", " ", "\t", "#", "\n")
            ),
            len(lines),
        )
        hit = next((i for i in range(start + 1, end) if _KEY.match(lines[i])), None)
        if hit is not None:
            m = _KEY.match(lines[hit])
            assert m is not None
            lines[hit] = f"{m.group(1)}{value}{m.group(2) or ''}\n"
        else:
            lines.insert(start + 1, f"  min_similarity: {value}\n")
        out = "".join(lines)

    parsed = yaml.safe_load(out) or {}
    if parsed.get("retriever", {}).get("min_similarity") != value:
        raise ValueError("could not set retriever.min_similarity in config YAML")
    return out


def get_few_shot_examples(
    entries: list[FeedbackEntry], max_examples: int = 3
) -> list[dict[str, str]]:
    """Return the most recent thumbs-up (query, answer) pairs, newest first."""
    thumbs_up = [e for e in reversed(entries) if e.rating == 1]
    return [{"query": e.query, "answer": e.answer} for e in thumbs_up[:max_examples]]


def get_kb_gaps(entries: list[FeedbackEntry]) -> list[str]:
    """Return queries where the LLM judge found missing content."""
    return [e.query for e in entries if e.category == "missing_content"]


def apply_optimization(
    entries: list[FeedbackEntry],
    config_path: Path | None = None,
    feedback_config_path: Path = FEEDBACK_CONFIG_PATH,
    few_shot_max: int = 3,
) -> OptimizationResult:
    """Run all three optimizations and persist results to disk.

    ``config_path`` defaults to the file load_config() reads (RAG_CONFIG_PATH aware).
    """
    config_path = resolve_config_path(config_path)
    # 1. Retrieval param tuning
    new_min_sim = tune_retrieval_params(entries)
    if new_min_sim is not None:
        try:
            config_path.write_text(
                set_min_similarity(config_path.read_text(), new_min_sim)
            )
            logger.info("optimizer: updated min_similarity to %s", new_min_sim)
        except Exception as exc:
            logger.warning("optimizer: failed to update config: %s", exc)
            new_min_sim = None

    # 2. Few-shot examples
    examples = get_few_shot_examples(entries, max_examples=few_shot_max)
    try:
        feedback_config_path.write_text(
            json.dumps({"few_shot_examples": examples}, indent=2, ensure_ascii=False)
        )
        logger.info("optimizer: wrote %d few-shot examples", len(examples))
    except OSError as exc:
        logger.warning("optimizer: failed to write feedback config: %s", exc)
        examples = []

    # 3. KB gaps
    gaps = get_kb_gaps(entries)

    return OptimizationResult(
        new_min_similarity=new_min_sim,
        few_shot_count=len(examples),
        kb_gaps=gaps,
    )
