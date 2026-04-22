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
    ) -> None:
        self._model: CrossEncoder = CrossEncoder(model)
        self._top_k = top_k

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Score (query, doc) pairs and return the top-k by descending score.

        Args:
            query: The user query string.
            candidates: List of result dicts, each with at least a "content" key.

        Returns:
            Up to top_k dicts from candidates, sorted by cross-encoder score
            descending. The "score" field on each dict is replaced with the
            cross-encoder score.
        """
        if not candidates:
            return []
        pairs = [(query, c["content"]) for c in candidates]
        scores: list[float] = self._model.predict(pairs).tolist()
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [doc | {"score": float(score)} for score, doc in ranked[: self._top_k]]
