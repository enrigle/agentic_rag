"""Cross-encoder reranker for candidate document re-scoring."""

from __future__ import annotations

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Reranks candidate documents using a cross-encoder model.

    The model is loaded once at instantiation and kept in memory.
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2",
        top_k: int = 5,
        min_score: float | None = None,
    ) -> None:
        self._model: CrossEncoder = CrossEncoder(model)
        self._top_k = top_k
        self._min_score = min_score

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Score (query, doc) pairs and return the top-k by descending score.

        Args:
            query: The user query string.
            candidates: List of result dicts, each with at least a "content" key.

        Returns:
            Up to top_k dicts from candidates, sorted by cross-encoder score
            descending. The "score" field on each dict is replaced with the
            cross-encoder score. When min_score is set, candidates scoring
            below it are dropped — possibly returning an empty list — so the
            caller can treat "nothing relevant" differently from "nothing".
        """
        if not candidates:
            return []
        # ponytail: with no relevance gate, predict() can't change which docs
        # are kept when they all fit in top_k — skip the forward pass.
        if self._min_score is None and len(candidates) <= self._top_k:
            return candidates
        pairs = [(query, c["content"]) for c in candidates]
        scores: list[float] = self._model.predict(pairs).tolist()
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        kept = [
            doc | {"score": float(score)}
            for score, doc in ranked
            if self._min_score is None or score >= self._min_score
        ]
        return kept[: self._top_k]
