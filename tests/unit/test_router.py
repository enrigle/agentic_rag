import logging

import pytest

from agentic_rag.models import ChatMessage
from agentic_rag.pipeline.router import (
    QueryRouter,
    RouteDecision,
    build_router_prompt,
    parse_route,
)


class StubLLM:
    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply
        self.prompts: list[str] = []
        self.json_flags: list[bool] = []

    async def chat(self, prompt: str, *, json_output: bool = False) -> str:
        self.prompts.append(prompt)
        self.json_flags.append(json_output)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_parse_route_valid() -> None:
    d = parse_route(
        '{"standalone_query": "list my skills", "route": "catalog", "focus_previous": false}',
        "give me the skills",
    )
    assert d == RouteDecision("list my skills", "catalog", False)


def test_parse_route_accepts_fences_and_prose() -> None:
    text = 'Sure:\n```json\n{"standalone_query": "a", "route": "chunks", "focus_previous": true}\n```'
    assert parse_route(text, "q") == RouteDecision("a", "chunks", True)


def test_parse_route_non_bool_focus_is_false() -> None:
    text = '{"standalone_query": "a", "route": "chunks", "focus_previous": "yes"}'
    assert parse_route(text, "q").focus_previous is False


def test_parse_route_bad_json_falls_back_to_both(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        assert parse_route("nope", "q") == RouteDecision("q", "both", False)
    assert "bad JSON" in caplog.text


def test_parse_route_unknown_route_falls_back() -> None:
    text = '{"standalone_query": "a", "route": "web", "focus_previous": true}'
    assert parse_route(text, "q") == RouteDecision("q", "both", False)


def test_parse_route_empty_query_falls_back() -> None:
    text = '{"standalone_query": "  ", "route": "catalog", "focus_previous": false}'
    assert parse_route(text, "q") == RouteDecision("q", "both", False)


def test_build_router_prompt_keeps_last_3_turns() -> None:
    history = []
    for i in range(5):
        history.append(ChatMessage("user", f"user-msg-{i}"))
        history.append(ChatMessage("assistant", f"asst-msg-{i}"))
    prompt = build_router_prompt("new q", history, [], 3)
    assert "user-msg-1" not in prompt and "asst-msg-1" not in prompt
    for i in (2, 3, 4):
        assert f"user-msg-{i}" in prompt and f"asst-msg-{i}" in prompt
    assert "new q" in prompt
    assert "no previous answer" in prompt.lower()


def test_build_router_prompt_lists_previous_pages_and_routes() -> None:
    prompt = build_router_prompt("summarize it", [], [("id1", "Azure ML")], 3)
    assert "Azure ML" in prompt
    for text in (
        "catalog",
        "chunks",
        "both",
        "give me all my skills",
        "tell me about plugins",
    ):
        assert text in prompt


async def test_router_llm_error_falls_back_to_both(
    caplog: pytest.LogCaptureFixture,
) -> None:
    router = QueryRouter(StubLLM(RuntimeError("down")), 3)
    with caplog.at_level(logging.WARNING):
        d = await router.route("q", [ChatMessage("user", "x")], [])
    assert d == RouteDecision("q", "both", False)
    assert "LLM error" in caplog.text


async def test_router_connection_error_falls_back_to_both(
    caplog: pytest.LogCaptureFixture,
) -> None:
    router = QueryRouter(StubLLM(ConnectionError("ollama down")), 3)
    with caplog.at_level(logging.WARNING):
        d = await router.route("q", [], [])
    assert d == RouteDecision("q", "both", False)
    assert "LLM error" in caplog.text


async def test_router_without_history_keeps_original_query() -> None:
    reply = (
        '{"standalone_query": "rewritten", "route": "catalog", "focus_previous": false}'
    )
    d = await QueryRouter(StubLLM(reply), 3).route("orig", [], [])
    assert d == RouteDecision("orig", "catalog", False)


async def test_router_with_history_uses_rewrite() -> None:
    reply = (
        '{"standalone_query": "rewritten", "route": "chunks", "focus_previous": true}'
    )
    d = await QueryRouter(StubLLM(reply), 3).route(
        "it?", [ChatMessage("user", "x")], [("1", "T")]
    )
    assert d == RouteDecision("rewritten", "chunks", True)


async def test_route_requests_json_output() -> None:
    llm = StubLLM(
        '{"standalone_query": "a", "route": "chunks", "focus_previous": false}'
    )
    await QueryRouter(llm, 3).route("q", [], [])
    assert llm.json_flags == [True]


def test_parse_route_reads_translated_query() -> None:
    text = (
        '{"standalone_query": "lower pool pH", "translated_query": "bajar pH piscina",'
        ' "route": "chunks", "focus_previous": false}'
    )
    assert parse_route(text, "q").translated_query == "bajar pH piscina"


@pytest.mark.parametrize("value", ['""', "3", '"Lower Pool pH"'])
def test_parse_route_drops_useless_translation(value: str) -> None:
    text = (
        f'{{"standalone_query": "lower pool pH", "translated_query": {value},'
        ' "route": "chunks", "focus_previous": false}'
    )
    assert parse_route(text, "q").translated_query is None


def test_build_router_prompt_asks_for_translation() -> None:
    assert "translated_query" in build_router_prompt("q", [], [], 3)


async def test_router_without_history_keeps_translation() -> None:
    reply = (
        '{"standalone_query": "x", "translated_query": "y", "route": "chunks",'
        ' "focus_previous": false}'
    )
    d = await QueryRouter(StubLLM(reply), 3).route("orig", [], [])
    assert d == RouteDecision("orig", "chunks", False, "y")


def test_build_router_prompt_keeps_question_language() -> None:
    # A Spanish follow-up rewritten into English loses its Spanish matches.
    assert "same language as the new question" in build_router_prompt("q", [], [], 3)
