"""Pull a JSON object out of an LLM reply that may have fences or prose around it."""

import json
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    """Decode the first JSON object in text. Raises ValueError if there is none."""
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in reply")
    try:
        value, _ = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON object in reply: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("reply JSON is not an object")
    return value
