"""Unit tests for agentic_rag.retrieval.hybrid._rrf_merge."""

from __future__ import annotations

import pytest

from agentic_rag.retrieval.hybrid import _rrf_merge


def test_rrf_merge_empty_inputs() -> None:
    merged, scores = _rrf_merge([], [], k=60, top_n=5)
    assert merged == []
    assert scores == {}


def test_rrf_merge_vector_only() -> None:
    vector_ids = ["a", "b", "c"]
    merged, scores = _rrf_merge(vector_ids, [], k=60, top_n=3)
    assert merged == ["a", "b", "c"]
    assert set(merged) == set(scores.keys())


def test_rrf_merge_bm25_only() -> None:
    bm25_ids = ["x", "y", "z"]
    merged, scores = _rrf_merge([], bm25_ids, k=60, top_n=3)
    assert merged == ["x", "y", "z"]
    assert set(merged) == set(scores.keys())


def test_rrf_merge_shared_ids_get_higher_scores() -> None:
    # "shared" appears in both lists at rank 0 → gets two contributions
    vector_ids = ["shared", "unique_v"]
    bm25_ids = ["shared", "unique_b"]
    merged, scores = _rrf_merge(vector_ids, bm25_ids, k=60, top_n=4)

    assert "shared" in scores
    assert "unique_v" in scores
    assert "unique_b" in scores

    # shared should have a higher score than either unique ID
    assert scores["shared"] > scores["unique_v"]
    assert scores["shared"] > scores["unique_b"]


def test_rrf_merge_top_n_caps_result_count() -> None:
    vector_ids = [f"v{i}" for i in range(10)]
    bm25_ids = [f"b{i}" for i in range(10)]
    merged, scores = _rrf_merge(vector_ids, bm25_ids, k=60, top_n=5)
    assert len(merged) == 5
    assert len(scores) == 5


def test_rrf_merge_scores_only_contains_winner_ids() -> None:
    vector_ids = ["a", "b", "c", "d", "e"]
    bm25_ids = ["e", "d", "c", "b", "a"]
    merged, scores = _rrf_merge(vector_ids, bm25_ids, k=60, top_n=3)
    # scores dict should only contain the top_n winners
    assert set(scores.keys()) == set(merged)
    assert len(scores) == 3


def test_rrf_merge_ordering_descending() -> None:
    """Merged IDs should be ordered by descending RRF score."""
    vector_ids = ["best", "middle", "worst"]
    bm25_ids = ["best", "middle"]  # "best" and "middle" get extra boost
    merged, scores = _rrf_merge(vector_ids, bm25_ids, k=60, top_n=3)
    # Verify descending order
    score_values = [scores[fid] for fid in merged]
    assert score_values == sorted(score_values, reverse=True)


def test_rrf_merge_top_n_larger_than_candidates() -> None:
    """top_n larger than available candidates returns all candidates."""
    vector_ids = ["a", "b"]
    merged, scores = _rrf_merge(vector_ids, [], k=60, top_n=10)
    assert len(merged) == 2
    assert set(merged) == {"a", "b"}


def test_rrf_merge_k_affects_scores() -> None:
    """Smaller k yields larger score differences between ranks."""
    vector_ids = ["first", "second"]
    _, scores_k1 = _rrf_merge(vector_ids, [], k=1, top_n=2)
    _, scores_k100 = _rrf_merge(vector_ids, [], k=100, top_n=2)

    diff_k1 = scores_k1["first"] - scores_k1["second"]
    diff_k100 = scores_k100["first"] - scores_k100["second"]
    assert diff_k1 > diff_k100
