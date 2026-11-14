import asyncio

import pytest

from agentic_rag.ui.event_loop import BackgroundLoop


async def _current_loop() -> asyncio.AbstractEventLoop:
    return asyncio.get_running_loop()


async def _fail() -> None:
    raise ValueError("boom")


def test_run_returns_coroutine_result() -> None:
    loop = BackgroundLoop()

    assert loop.run(asyncio.sleep(0, result=42)) == 42


def test_all_calls_share_one_loop() -> None:
    """Async clients bind to a loop; every call must reuse the same one."""
    loop = BackgroundLoop()

    first = loop.run(_current_loop())
    second = loop.run(_current_loop())

    assert first is second
    assert not first.is_closed()


def test_run_raises_coroutine_error() -> None:
    loop = BackgroundLoop()

    with pytest.raises(ValueError, match="boom"):
        loop.run(_fail())
