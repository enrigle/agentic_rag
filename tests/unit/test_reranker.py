"""Unit tests for agentic_rag.retrieval.reranker.CrossEncoderReranker."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agentic_rag.retrieval.reranker import CrossEncoderReranker


def _make_candidates(n: int) -> list[dict[str, Any]]:
    return [{"id": str(i), "content": f"doc {i}", "score": 0.0} for i in range(n)]


def _make_reranker(min_score: float | None = None) -> CrossEncoderReranker:
    """CrossEncoderReranker with a mocked CrossEncoder to avoid downloading weights."""
    with patch("agentic_rag.retrieval.reranker.CrossEncoder") as mock_cls:
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        instance = CrossEncoderReranker(
            model="mock-model", top_k=3, min_score=min_score
        )
        instance._model = mock_model
        return instance


@pytest.fixture()
def reranker() -> CrossEncoderReranker:
    return _make_reranker()


def test_rerank_empty_candidates(reranker: CrossEncoderReranker) -> None:
    result = reranker.rerank("query", [])
    assert result == []
    reranker._model.predict.assert_not_called()  # type: ignore[attr-defined]


def test_rerank_returns_top_k(reranker: CrossEncoderReranker) -> None:
    import numpy as np

    candidates = _make_candidates(5)
    reranker._model.predict.return_value = np.array([0.1, 0.9, 0.5, 0.8, 0.3])  # type: ignore[attr-defined]

    result = reranker.rerank("query", candidates)

    assert len(result) == 3  # top_k=3


def test_rerank_scores_descending(reranker: CrossEncoderReranker) -> None:
    import numpy as np

    candidates = _make_candidates(4)
    reranker._model.predict.return_value = np.array([0.2, 0.8, 0.5, 0.1])  # type: ignore[attr-defined]

    result = reranker.rerank("query", candidates)

    scores = [r["score"] for r in result]
    assert scores == sorted(scores, reverse=True)


def test_rerank_score_field_overwritten(reranker: CrossEncoderReranker) -> None:
    import numpy as np

    # More candidates than top_k=3 so the cross-encoder actually runs.
    candidates = _make_candidates(4)
    candidates[0]["score"] = 999.0
    reranker._model.predict.return_value = np.array([0.42, 0.1, 0.2, 0.3])  # type: ignore[attr-defined]

    result = reranker.rerank("query", candidates)

    # Top result is candidate 0 (highest predict score) with its score overwritten.
    assert result[0]["id"] == "0"
    assert result[0]["score"] == pytest.approx(0.42)


def test_rerank_fewer_candidates_than_top_k(reranker: CrossEncoderReranker) -> None:
    candidates = _make_candidates(2)  # top_k=3 but only 2 candidates

    result = reranker.rerank("query", candidates)

    # Skip-guard: no forward pass needed when every candidate is kept anyway.
    assert result == candidates
    reranker._model.predict.assert_not_called()  # type: ignore[attr-defined]


def test_rerank_min_score_drops_low_scores() -> None:
    import numpy as np

    gated = _make_reranker(min_score=0.0)
    candidates = _make_candidates(4)
    gated._model.predict.return_value = np.array([-5.0, 2.0, -0.1, 0.5])  # type: ignore[attr-defined]

    result = gated.rerank("query", candidates)

    assert [r["id"] for r in result] == ["1", "3"]  # only scores >= 0.0 survive


def test_rerank_min_score_can_return_empty() -> None:
    import numpy as np

    gated = _make_reranker(min_score=0.0)
    candidates = _make_candidates(3)
    gated._model.predict.return_value = np.array([-8.0, -6.5, -9.1])  # type: ignore[attr-defined]

    result = gated.rerank("query", candidates)

    assert result == []


def test_rerank_min_score_disables_skip_guard() -> None:
    import numpy as np

    # 2 candidates <= top_k=3: without a gate predict is skipped, with a gate
    # it must run so irrelevant docs can be dropped.
    gated = _make_reranker(min_score=0.0)
    candidates = _make_candidates(2)
    gated._model.predict.return_value = np.array([1.0, -1.0])  # type: ignore[attr-defined]

    result = gated.rerank("query", candidates)

    gated._model.predict.assert_called_once()  # type: ignore[attr-defined]
    assert [r["id"] for r in result] == ["0"]
