"""Pure chunking functions extracted from the original ingest.py.

These functions are stateless and perform no I/O.
"""

from dataclasses import dataclass, field
from typing import Any

# Block types that contain rich_text content worth indexing
RICH_TEXT_BLOCK_TYPES = {
    "paragraph",
    "heading_1",
    "heading_2",
    "heading_3",
    "bulleted_list_item",
    "numbered_list_item",
    "code",
    "quote",
    "toggle",
    "callout",
}

HEADING_BLOCK_TYPES = {"heading_1", "heading_2", "heading_3"}


@dataclass(slots=True)
class _ChunkBuffer:
    chunks: list[str]
    size: int
    overlap: int
    heading: str = ""
    paras: list[str] = field(default_factory=list)
    length: int = 0

    def flush(self) -> None:
        if not self.paras:
            return
        _emit_chunk(self.chunks, self.heading, self.paras)
        self.paras = []
        self.length = 0

    def set_heading(self, heading: str) -> None:
        if heading != self.heading and self.paras:
            self.flush()
        self.heading = heading

    def _overlap_carry_from_last_paragraph(self) -> str:
        if self.overlap <= 0 or not self.paras:
            return ""
        last_para = self.paras[-1]
        if len(last_para) <= self.overlap:
            return last_para
        return last_para[-self.overlap :].lstrip()

    def flush_with_overlap_seed(self, next_para: str, next_para_len: int) -> None:
        if not self.paras:
            return

        carry = self._overlap_carry_from_last_paragraph()
        self.flush()

        if carry and (len(carry) + 1 + next_para_len <= self.size):
            self.paras = [carry]
            self.length = len(carry)

    def append(self, paragraph: str, paragraph_len: int) -> None:
        self.length += paragraph_len + (1 if self.paras else 0)
        self.paras.append(paragraph)


def _extract_plain_text(block: dict[str, Any]) -> str:
    """Return plain text from a block's rich_text array, or empty string."""
    block_type = block.get("type", "")
    if block_type not in RICH_TEXT_BLOCK_TYPES:
        return ""
    rich_texts: list[dict[str, Any]] = block.get(block_type, {}).get("rich_text", [])
    return "".join(rt.get("plain_text", "") for rt in rich_texts)


def _get_title(page: dict[str, Any]) -> str:
    """Extract page title from Notion page properties."""
    for prop in page.get("properties", {}).values():
        if prop.get("type") == "title":
            rich_texts = prop.get("title", [])
            return "".join(rt.get("plain_text", "") for rt in rich_texts)
    return "Untitled"


def _emit_chunk(chunks: list[str], heading: str, paras: list[str]) -> None:
    prefix = f"{heading}\n" if heading else ""
    chunks.append(prefix + "\n".join(paras))


def _split_long_text(text: str, size: int, overlap: int) -> list[str]:
    # Split *text* into chunks of at most *size* characters, with optional *overlap*.
    if not text:
        return []
    if size <= 0:
        return [text]
    if overlap < 0:
        overlap = 0

    out: list[str] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = text.rfind(" ", start, end)
            if cut > start:
                end = cut
        if end <= start:
            end = min(start + size, n)

        chunk = text[start:end].strip()
        if chunk:
            out.append(chunk)

        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start
        while start < n and text[start].isspace():
            start += 1

    return out


def _chunk_text(
    blocks: list[dict[str, str]],
    size: int = 800,
    overlap: int = 100,
) -> list[str]:
    """Paragraph-aware chunking that prepends the last seen heading to each chunk.

    Splits on paragraph boundaries first. Falls back to word-boundary
    character-splitting only for paragraphs that exceed ``size`` on their own.
    The most recent heading is prepended to every chunk so the LLM always has
    section context.
    """
    if not blocks:
        return []

    # Build (heading_context, paragraph_text) pairs
    tagged: list[tuple[str, str]] = []
    current_heading = ""
    for block in blocks:
        block_type = block.get("type", "")
        text = block.get("text", "").strip()
        if not text:
            continue
        if block_type in HEADING_BLOCK_TYPES:
            current_heading = text
        else:
            tagged.append((current_heading, text))

    if not tagged:
        return []

    chunks: list[str] = []
    buf = _ChunkBuffer(chunks=chunks, size=size, overlap=overlap)

    for heading, para in tagged:
        buf.set_heading(heading)

        para_len = len(para)

        # Oversized single paragraph: flush buffer then character-split the para
        if para_len > size:
            buf.flush()
            for sub in _split_long_text(para, size=size, overlap=overlap):
                _emit_chunk(chunks, buf.heading, [sub])
            continue

        # Adding this paragraph would exceed the size limit: flush first
        if buf.paras and (buf.length + 1 + para_len > size):
            buf.flush_with_overlap_seed(para, para_len)

        buf.append(para, para_len)

    buf.flush()

    return chunks
