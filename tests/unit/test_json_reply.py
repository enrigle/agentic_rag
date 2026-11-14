import pytest

from agentic_rag.llm.json_reply import extract_json_object


def test_extract_json_object_strips_fences_and_prose() -> None:
    assert extract_json_object('Sure:\n```json\n{"a": 1}\n```\nDone') == {"a": 1}


def test_extract_json_object_raises_without_object() -> None:
    with pytest.raises(ValueError):
        extract_json_object("no json here")


def test_extract_json_object_handles_nested_and_malformed() -> None:
    assert extract_json_object('x {"a": {"b": [1]}} y') == {"a": {"b": [1]}}
    with pytest.raises(ValueError):
        extract_json_object("{not json}")
