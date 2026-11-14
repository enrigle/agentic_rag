"""One long-lived event loop for the Streamlit app.

Async clients (httpx inside ollama/openai) bind their connections to the loop
that first used them. ``asyncio.run`` makes a new loop per call and closes it,
so a cached client fails on the second call with "Event loop is closed".
"""

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


class BackgroundLoop:
    """Runs coroutines on one event loop that lives in a daemon thread."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """Run *coro* on the shared loop and block until it returns."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()
