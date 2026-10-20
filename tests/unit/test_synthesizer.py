"""Unit tests for Synthesizer prompt building, incl. few-shot examples."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_rag.models import PipelineContext
from agentic_rag.pipeline.synthesizer import Synthesizer
from agentic_rag.models import SearchResult


def _sr(
    id: str = "",
    title: str = "",
    source: str = "",
    content: str = "",
    score: float = 0.0,
    similarity: float | None = None,
) -> SearchResult:
    return SearchResult(id, title, source, content, score, similarity)


def _ctx() -> PipelineContext:
    return PipelineContext(
        query="q",
        chat_history=[],
        results=[_sr(title="T", source="s", content="body")],
        final_answer=None,
        error=None,
        tool_calls=0,
        max_tool_calls=3,
    )


def _llm() -> MagicMock:
    llm = MagicMock()
    llm.chat = AsyncMock(return_value="answer")
    return llm


def _prompt(llm: MagicMock) -> str:
    return str(llm.chat.await_args.args[0])


@pytest.mark.asyncio
async def test_few_shot_examples_reach_prompt(tmp_path: Path) -> None:
    path = tmp_path / "feedback_config.json"
    path.write_text(
        json.dumps({"few_shot_examples": [{"query": "EX_Q", "answer": "EX_A"}]})
    )
    llm = _llm()
    await Synthesizer(llm, few_shot_path=path).synthesize("q", _ctx())
    assert "EX_Q" in _prompt(llm)
    assert "EX_A" in _prompt(llm)


@pytest.mark.asyncio
async def test_examples_reread_each_call(tmp_path: Path) -> None:
    # Optimizer rewrites the file while the (cached) pipeline keeps running.
    path = tmp_path / "feedback_config.json"
    llm = _llm()
    synth = Synthesizer(llm, few_shot_path=path)
    await synth.synthesize("q", _ctx())
    assert "NEW_Q" not in _prompt(llm)
    path.write_text(
        json.dumps({"few_shot_examples": [{"query": "NEW_Q", "answer": "NEW_A"}]})
    )
    await synth.synthesize("q", _ctx())
    assert "NEW_Q" in _prompt(llm)


@pytest.mark.asyncio
async def test_missing_file_means_no_examples(tmp_path: Path) -> None:
    llm = _llm()
    synth = Synthesizer(llm, few_shot_path=tmp_path / "absent.json")
    assert await synth.synthesize("q", _ctx()) == "answer"
    assert "Example" not in _prompt(llm)


@pytest.mark.asyncio
async def test_malformed_file_warns_and_answers(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "feedback_config.json"
    path.write_text("{not json")
    llm = _llm()
    with caplog.at_level(logging.WARNING):
        assert (
            await Synthesizer(llm, few_shot_path=path).synthesize("q", _ctx())
            == "answer"
        )
    assert "few-shot" in caplog.text


@pytest.mark.asyncio
async def test_chat_history_rendered_in_prompt() -> None:
    from agentic_rag.models import ChatMessage

    ctx = _ctx()
    ctx.chat_history = [ChatMessage("user", "earlier q"), ChatMessage("assistant", "")]
    llm = _llm()
    await Synthesizer(llm).synthesize("q", ctx)
    assert "Conversation so far:\nuser: earlier q" in _prompt(llm)
    assert "assistant:" not in _prompt(llm)  # empty turns skipped
