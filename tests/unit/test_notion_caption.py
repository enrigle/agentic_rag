"""Unit tests for NotionIngester._caption_image OCR fallback."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from agentic_rag.ingestion.notion import NotionIngester


def _make_ingester() -> NotionIngester:
    with patch("agentic_rag.ingestion.notion.chromadb.PersistentClient"):
        return NotionIngester(MagicMock(), MagicMock())


def test_caption_image_ocr_fallback_when_no_vision_model() -> None:
    """ollama_client=None → Tesseract OCR path (not the vision model)."""
    ingester = _make_ingester()
    with (
        patch("agentic_rag.ingestion.notion.urllib.request.urlopen") as mock_open,
        patch("agentic_rag.ingestion.notion.Image.open"),
        patch(
            "agentic_rag.ingestion.notion.pytesseract.image_to_string",
            return_value="  hello from image  ",
        ) as mock_ocr,
    ):
        mock_open.return_value.__enter__.return_value.read.return_value = b"img"
        result = asyncio.run(ingester._caption_image(None, "http://x/y.png"))

    assert result == "hello from image"  # stripped
    mock_ocr.assert_called_once()


def test_caption_image_returns_empty_on_failure() -> None:
    """Any download/OCR error is swallowed and yields an empty caption."""
    ingester = _make_ingester()
    with patch(
        "agentic_rag.ingestion.notion.urllib.request.urlopen",
        side_effect=OSError("boom"),
    ):
        result = asyncio.run(ingester._caption_image(None, "http://x/y.png"))

    assert result == ""
