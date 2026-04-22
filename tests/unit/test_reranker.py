"""Unit tests for agentic_rag.retrieval.reranker.CrossEncoderReranker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agentic_rag.retrieval.reranker import CrossEncoderReranker


def _make_candidates(n: int) -> list[dict]:
    return [{"id": str(i), "content": f"doc {i}", "score": 0.0} for i in range(n)]


@pytest.fixture()
def reranker() -> CrossEncoderReranker:
    """CrossEncoderReranker with a mocked CrossEncoder to avoid downloading weights."""
    with patch("agentic_rag.retrieval.reranker.CrossEncoder") as mock_cls:
        mock_model = MagicMock()
        mock_cls.return_value = mock_model
        instance = CrossEncoderReranker(model="mock-model", top_k=3)
        instance._model = mock_model
        yield instance


def test_rerank_empty_candidates(reranker: CrossEncoderReranker) -> None:
    result = reranker.rerank("query", [])
    assert result == []
    reranker._model.predict.assert_not_called()


def test_rerank_returns_top_k(reranker: CrossEncoderReranker) -> None:
    import numpy as np

    candidates = _make_candidates(5)
    reranker._model.predict.return_value = np.array([0.1, 0.9, 0.5, 0.8, 0.3])

    result = reranker.rerank("query", candidates)

    assert len(result) == 3  # top_k=3


def test_rerank_scores_descending(reranker: CrossEncoderReranker) -> None:
    import numpy as np

    candidates = _make_candidates(4)
    reranker._model.predict.return_value = np.array([0.2, 0.8, 0.5, 0.1])

    result = reranker.rerank("query", candidates)

    scores = [r["score"] for r in result]
    assert scores == sorted(scores, reverse=True)


def test_rerank_score_field_overwritten(reranker: CrossEncoderReranker) -> None:
    import numpy as np

    candidates = [{"id": "a", "content": "hello", "score": 999.0}]
    reranker._model.predict.return_value = np.array([0.42])

    result = reranker.rerank("query", candidates)

    assert len(result) == 1
    assert result[0]["score"] == pytest.approx(0.42)


def test_rerank_fewer_candidates_than_top_k(reranker: CrossEncoderReranker) -> None:
    import numpy as np

    candidates = _make_candidates(2)  # top_k=3 but only 2 candidates
    reranker._model.predict.return_value = np.array([0.7, 0.3])

    result = reranker.rerank("query", candidates)

    assert len(result) == 2
