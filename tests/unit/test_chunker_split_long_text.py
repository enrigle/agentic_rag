"""Edge-case unit tests for agentic_rag.ingestion.chunker._split_long_text."""

from __future__ import annotations

from agentic_rag.ingestion.chunker import _split_long_text


def test_split_long_text_size_zero_returns_whole_text() -> None:
    text = "abcdef"
    assert _split_long_text(text, size=0, overlap=999) == [text]


def test_split_long_text_negative_overlap_treated_as_zero() -> None:
    text = "a" * 35
    chunks_neg = _split_long_text(text, size=10, overlap=-5)
    chunks_zero = _split_long_text(text, size=10, overlap=0)
    assert chunks_neg == chunks_zero


def test_split_long_text_overlap_larger_than_size_terminates() -> None:
    # This used to be a common pitfall for chunkers: overlap >= size can loop.
    text = "a" * 55
    chunks = _split_long_text(text, size=10, overlap=100)
    assert chunks  # non-empty
    assert all(chunks)
    assert all(len(c) <= 10 for c in chunks)
    assert chunks[0] == "a" * 10
    assert chunks[-1] == "a" * 5

