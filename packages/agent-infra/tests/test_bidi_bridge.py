"""Tests for the sync<->async bridging in bidi_bridge.py.

No skip needed: this module only uses asyncio/queue/threading, none of which
need the container-only ``bedrock_agentcore``/``strands`` dependencies.

Driven via plain ``asyncio.run()`` rather than ``pytest-asyncio`` markers --
this repo has no existing async-test convention to extend, and one file
alone does not justify a new dev dependency.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from bidi_bridge import SyncTextQueue, bidi_stream_to_async


class _FakeSyncStream:
    """Stands in for BidiChunkStream: a context-managed sync iterator."""

    def __init__(self, items: list[int], raise_after: int | None = None) -> None:
        self._items = items
        self._raise_after = raise_after
        self._index = 0

    def __enter__(self) -> _FakeSyncStream:
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass

    def __iter__(self) -> _FakeSyncStream:
        return self

    def __next__(self) -> int:
        if self._raise_after is not None and self._index >= self._raise_after:
            raise RuntimeError("fake stream failure")
        if self._index >= len(self._items):
            raise StopIteration
        value = self._items[self._index]
        self._index += 1
        return value


async def _collect(agen) -> list:
    return [item async for item in agen]


class TestBidiStreamToAsync:
    def test_yields_items_in_order(self) -> None:
        stream = _FakeSyncStream([1, 2, 3])
        result = asyncio.run(_collect(bidi_stream_to_async(lambda: stream)))
        assert result == [1, 2, 3]

    def test_propagates_exceptions_from_the_worker_thread(self) -> None:
        stream = _FakeSyncStream([1, 2, 3], raise_after=1)
        with pytest.raises(RuntimeError, match="fake stream failure"):
            asyncio.run(_collect(bidi_stream_to_async(lambda: stream)))

    def test_worker_thread_does_not_leak(self) -> None:
        # Not a raw threading.active_count() comparison: asyncio's default
        # run_in_executor() ThreadPoolExecutor keeps its own worker threads
        # alive across calls in the same process, which would make any
        # total-count assertion flaky for reasons unrelated to this bridge.
        # Track the bridge's own named thread instead.
        async def _run() -> None:
            stream = _FakeSyncStream([1, 2, 3])
            await _collect(bidi_stream_to_async(lambda: stream))
            await asyncio.sleep(0.05)
            leftover = [t for t in threading.enumerate() if t.name == "bidi-bridge"]
            assert not any(t.is_alive() for t in leftover)

        asyncio.run(_run())


class TestSyncTextQueue:
    def test_put_then_close_yields_all_items_then_stops(self) -> None:
        q = SyncTextQueue()
        q.put("a")
        q.put("b")
        q.close()
        assert list(q) == ["a", "b"]

    def test_iteration_blocks_until_put_or_close(self) -> None:
        q = SyncTextQueue()
        results: list[str] = []

        def _consume() -> None:
            results.extend(q)

        consumer = threading.Thread(target=_consume)
        consumer.start()
        q.put("first")
        q.close()
        consumer.join(timeout=2.0)
        assert results == ["first"]
