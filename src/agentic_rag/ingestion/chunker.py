"""Pure chunking functions extracted from the original ingest.py.

These functions are stateless and perform no I/O.
"""

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
    buf_paras: list[str] = []
    buf_heading = ""
    buf_len = 0

    for heading, para in tagged:
        if heading:
            buf_heading = heading

        # Oversized single paragraph: flush buffer then character-split the para
        if len(para) > size:
            if buf_paras:
                _emit_chunk(chunks, buf_heading, buf_paras)
                buf_paras = []
                buf_len = 0
            start = 0
            while start < len(para):
                end = start + size
                if end < len(para):
                    space = para.rfind(" ", start, end)
                    if space > start:
                        end = space
                sub = para[start:end].strip()
                if sub:
                    _emit_chunk(chunks, buf_heading, [sub])
                start = max(start + 1, end - overlap)
            continue

        # Adding this paragraph would exceed the size limit: flush first
        if buf_paras and buf_len + len(para) + 1 > size:
            _emit_chunk(chunks, buf_heading, buf_paras)
            # Carry the last paragraph forward as overlap context
            last = buf_paras[-1]
            buf_paras = [last] if len(last) + len(para) + 1 <= size else []
            buf_len = len(buf_paras[0]) if buf_paras else 0

        buf_paras.append(para)
        buf_len = sum(len(p) for p in buf_paras) + max(0, len(buf_paras) - 1)

    if buf_paras:
        _emit_chunk(chunks, buf_heading, buf_paras)

    return chunks
