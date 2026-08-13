"""Bridges the synchronous ``BidiChunkStream`` into an async generator.

``BidiChunkStream`` (``tts_client.streaming``) drives its own private event
loop via ``loop.run_until_complete()`` per ``__next__()`` -- calling that
directly inside the entrypoint's async generator would block every other
in-flight session this process is serving. ``bidi_stream_to_async`` runs the
whole synchronous iteration in a worker thread and relays items through a
``queue.Queue``, drained via ``loop.run_in_executor(None, q.get)`` so the
caller's event loop never blocks.

The opposite boundary -- the agent's ``stream_async()`` is an async
generator, but ``synthesize_bidi_stream``'s ``text_source`` must be a plain
sync iterable -- is ``SyncTextQueue``, a second queue crossing the same
worker thread in reverse.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncGenerator, Callable, Iterator
from typing import Any

_DONE = object()


async def bidi_stream_to_async(open_stream: Callable[[], Any]) -> AsyncGenerator[Any, None]:
    """Bridge a context-managed sync iterator (a ``BidiChunkStream``) to async.

    ``open_stream`` is a zero-arg callable returning the stream -- deferred
    so the stream (and its private event loop) opens inside the worker
    thread, not the caller's.
    """
    q: queue.Queue = queue.Queue()
    loop = asyncio.get_event_loop()

    def _run() -> None:
        try:
            with open_stream() as stream:
                for chunk in stream:
                    q.put(("chunk", chunk))
        except Exception as exc:  # noqa: BLE001 - relayed to the async side, not swallowed
            q.put(("error", exc))
        finally:
            q.put(("done", None))

    thread = threading.Thread(target=_run, daemon=True, name="bidi-bridge")
    thread.start()

    try:
        while True:
            kind, value = await loop.run_in_executor(None, q.get)
            if kind == "done":
                break
            if kind == "error":
                raise value
            yield value
    finally:
        thread.join(timeout=5.0)


class SyncTextQueue:
    """A sync iterable fed from the async side of the bridge.

    ``put()`` is called from async code, as the agent's own stream is
    consumed; ``__iter__`` is consumed synchronously, inside the worker
    thread ``bidi_stream_to_async`` starts for the ``BidiChunkStream`` side.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()

    def put(self, fragment: str) -> None:
        self._queue.put(fragment)

    def close(self) -> None:
        self._queue.put(_DONE)

    def __iter__(self) -> Iterator[str]:
        while True:
            item = self._queue.get()
            if item is _DONE:
                return
            yield item
