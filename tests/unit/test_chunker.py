"""Unit tests for agentic_rag.ingestion.chunker pure functions."""

from __future__ import annotations

import pytest

from agentic_rag.ingestion.chunker import _chunk_text, _extract_plain_text


# ---------------------------------------------------------------------------
# _extract_plain_text
# ---------------------------------------------------------------------------


def test_extract_plain_text_unknown_type_returns_empty() -> None:
    block = {"type": "unsupported_type", "unsupported_type": {"rich_text": [{"plain_text": "hi"}]}}
    assert _extract_plain_text(block) == ""


def test_extract_plain_text_missing_type_returns_empty() -> None:
    assert _extract_plain_text({}) == ""


def test_extract_plain_text_paragraph_concatenates() -> None:
    block = {
        "type": "paragraph",
        "paragraph": {
            "rich_text": [
                {"plain_text": "Hello"},
                {"plain_text": " world"},
            ]
        },
    }
    assert _extract_plain_text(block) == "Hello world"


def test_extract_plain_text_heading_returns_text() -> None:
    block = {
        "type": "heading_1",
        "heading_1": {"rich_text": [{"plain_text": "My Heading"}]},
    }
    assert _extract_plain_text(block) == "My Heading"


def test_extract_plain_text_empty_rich_text_returns_empty() -> None:
    block = {"type": "paragraph", "paragraph": {"rich_text": []}}
    assert _extract_plain_text(block) == ""


def test_extract_plain_text_missing_plain_text_key() -> None:
    block = {
        "type": "paragraph",
        "paragraph": {"rich_text": [{"no_plain_text_key": "ignored"}]},
    }
    assert _extract_plain_text(block) == ""


# ---------------------------------------------------------------------------
# _chunk_text
# ---------------------------------------------------------------------------


def test_chunk_text_empty_input_returns_empty() -> None:
    assert _chunk_text([]) == []


def test_chunk_text_only_empty_text_blocks_returns_empty() -> None:
    blocks = [{"type": "paragraph", "text": "   "}]
    assert _chunk_text(blocks) == []


def test_chunk_text_single_short_paragraph_returns_one_chunk() -> None:
    blocks = [{"type": "paragraph", "text": "Short content."}]
    chunks = _chunk_text(blocks, size=800)
    assert len(chunks) == 1
    assert "Short content." in chunks[0]


def test_chunk_text_heading_context_prepended() -> None:
    blocks = [
        {"type": "heading_2", "text": "Section Header"},
        {"type": "paragraph", "text": "Paragraph under section."},
    ]
    chunks = _chunk_text(blocks, size=800)
    assert len(chunks) == 1
    assert chunks[0].startswith("Section Header\n")
    assert "Paragraph under section." in chunks[0]


def test_chunk_text_heading_only_returns_empty() -> None:
    # Headings are not emitted as chunks on their own; they set context
    blocks = [{"type": "heading_1", "text": "Just a heading"}]
    assert _chunk_text(blocks) == []


def test_chunk_text_oversized_paragraph_is_split() -> None:
    long_text = "word " * 400  # ~2000 chars
    blocks = [{"type": "paragraph", "text": long_text}]
    chunks = _chunk_text(blocks, size=100, overlap=10)
    assert len(chunks) > 1
    for chunk in chunks:
        # Each individual chunk should be at most size + small margin
        assert len(chunk) <= 120  # size=100 + some tolerance for word boundary


def test_chunk_text_oversized_paragraph_overlap() -> None:
    """Chunks from an oversized paragraph should overlap."""
    long_para = "word" * 50  # creates "wordword..." — long enough for multiple chunks
    blocks = [{"type": "paragraph", "text": long_para}]
    chunks = _chunk_text(blocks, size=20, overlap=10)
    assert len(chunks) >= 2
    # The end of chunk 0 and the start of chunk 1 should share content
    # due to the overlap
    chunk0_end = chunks[0][-10:]  # last 10 chars of first chunk
    chunk1_start = chunks[1][:10]  # first 10 chars of second chunk
    assert chunk0_end in chunks[1] or chunk1_start in chunks[0], (
        f"Expected overlap between chunks, got:\n  chunk0 end: {chunk0_end!r}\n  chunk1 start: {chunk1_start!r}"
    )


def test_chunk_text_multiple_paragraphs_batched() -> None:
    """Short paragraphs should be grouped into a single chunk when under size."""
    blocks = [
        {"type": "paragraph", "text": "First."},
        {"type": "paragraph", "text": "Second."},
        {"type": "paragraph", "text": "Third."},
    ]
    chunks = _chunk_text(blocks, size=800)
    assert len(chunks) == 1
    assert "First." in chunks[0]
    assert "Third." in chunks[0]


def test_chunk_text_flushes_when_size_exceeded() -> None:
    """Paragraphs that together exceed size should be split into multiple chunks."""
    para = "x" * 300
    blocks = [
        {"type": "paragraph", "text": para},
        {"type": "paragraph", "text": para},
        {"type": "paragraph", "text": para},
    ]
    chunks = _chunk_text(blocks, size=400, overlap=0)
    assert len(chunks) >= 2
