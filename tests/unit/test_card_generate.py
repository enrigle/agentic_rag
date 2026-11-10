import json
from unittest.mock import AsyncMock

import pytest

from agentic_rag.catalog.generate import (
    apply_tag_map,
    build_card_prompt,
    build_tag_merge_prompt,
    generate_card_fields,
    parse_card_reply,
    parse_tag_map,
)
from agentic_rag.catalog.store import PageCard


def _card(page_id: str, tags: list[str]) -> PageCard:
    return PageCard(page_id, "T", "u", "t", "s", tags, [], False, None, [], [0.1])


class FlakyChat:
    def __init__(self, failures: int, reply: str = '{"summary": "ok"}') -> None:
        self.failures = failures
        self.reply = reply
        self.calls = 0

    async def chat(self, prompt: str) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError("backend down")
        return self.reply


def test_parse_card_reply_drops_items_not_in_page() -> None:
    page = "Skills: tdd, grilling, caveman"
    reply = '{"summary": "A list.", "tags": ["Claude", "claude"], "items": ["tdd", "grilling", "invented"]}'
    f = parse_card_reply(reply, page)
    assert (
        f.items == ["tdd", "grilling"]
        and f.tags == ["claude"]
        and not f.items_truncated
    )


def test_parse_card_reply_item_check_is_case_insensitive_and_keeps_text() -> None:
    f = parse_card_reply('{"summary": "s", "items": ["TDD", "TDD"]}', "we like tdd")
    assert f.items == ["TDD"]


def test_parse_card_reply_caps_items_at_50() -> None:
    names = [f"skill{i:02d}" for i in range(60)]
    f = parse_card_reply(
        json.dumps({"summary": "s", "tags": [], "items": names}), " ".join(names)
    )
    assert len(f.items) == 50 and f.items_truncated


def test_parse_card_reply_empty_summary_raises() -> None:
    with pytest.raises(ValueError):
        parse_card_reply('{"summary": "  ", "tags": [], "items": []}', "page")


def test_build_prompts_contain_inputs() -> None:
    assert "My Title" in build_card_prompt("My Title", "body text")
    assert "body text" in build_card_prompt("My Title", "body text")
    assert "claude-code" in build_tag_merge_prompt(["claude-code"])


async def test_generate_card_fields_retries_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep = AsyncMock()
    monkeypatch.setattr("agentic_rag.catalog.generate.asyncio.sleep", sleep)
    llm = FlakyChat(failures=2)
    f = await generate_card_fields(llm, "T", "page")
    assert f.summary == "ok" and llm.calls == 3
    assert [c.args[0] for c in sleep.await_args_list] == [1, 2]


async def test_generate_card_fields_reraises_after_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agentic_rag.catalog.generate.asyncio.sleep", AsyncMock())
    llm = FlakyChat(failures=99)
    with pytest.raises(RuntimeError):
        await generate_card_fields(llm, "T", "page")
    assert llm.calls == 3


async def test_generate_card_fields_does_not_retry_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agentic_rag.catalog.generate.asyncio.sleep", AsyncMock())
    llm = FlakyChat(failures=0, reply="not json")
    with pytest.raises(ValueError):
        await generate_card_fields(llm, "T", "page")
    assert llm.calls == 1


def test_parse_tag_map_unknown_tags_map_to_self() -> None:
    assert parse_tag_map('{"claude-code": "Claude"}', ["claude-code", "evals"]) == {
        "claude-code": "claude",
        "evals": "evals",
    }


def test_apply_tag_map_dedupes() -> None:
    cards = [_card("a", ["claude-code", "anthropic"]), _card("b", ["x"])]
    out = apply_tag_map(cards, {"claude-code": "claude", "anthropic": "claude"})
    assert out[0].tags == ["claude"] and out[1].tags == ["x"]
    assert cards[0].tags == ["claude-code", "anthropic"]


def test_build_card_prompt_truncates_page_text() -> None:
    prompt = build_card_prompt("T", "a" * 100 + "TAIL", max_page_chars=100)
    assert "a" * 100 in prompt and "TAIL" not in prompt


async def test_items_beyond_truncation_still_accepted() -> None:
    page = "x" * 200 + " Skill Z"
    chat = FlakyChat(0, '{"summary": "ok", "items": ["Skill Z"]}')
    fields = await generate_card_fields(chat, "T", page, max_page_chars=100)
    assert fields.items == ["Skill Z"]
