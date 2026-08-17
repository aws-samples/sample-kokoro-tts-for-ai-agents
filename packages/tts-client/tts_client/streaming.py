# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Incremental text-chunk streaming over one bidi session.

:meth:`TTSClient.synthesize_bidi_stream` sends each sentence as its own
message on one open bidirectional session, rather than the whole text as a
single message the way :meth:`TTSClient.synthesize_bidi` does. Every bidi
container's handler loop already accepts a further message on the same
socket after ``synthesis_complete`` (it only stops on an explicit
``{"type": "close"}``), so nothing server-side needs to change for this;
it is purely a client-side sending pattern.

This does not overlap chunk N+1's inference with chunk N's audio still
transmitting: every bidi container's handler is a strictly sequential
``receive -> synthesize & send all audio -> synthesis_complete -> receive``
loop with no concurrency, so chunk N+1 cannot be sent (or acted on) before
chunk N's ``synthesis_complete`` arrives. What this buys instead is not
waiting for the *entire* text to be assembled before sending anything --
chunk 1 starts synthesizing the moment it is available, rather than after
chunk 3's text exists.

:class:`IncrementalSentenceChunker` is the single primitive behind both
supported input shapes: text handed over as one complete string, and text
arriving as fragments over time (e.g. from an upstream LLM's token stream).
The latter is the former fed one fragment at a time followed by a flush --
same regex, same code path, so the two cannot drift apart.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import warnings
from collections.abc import AsyncGenerator, Iterable
from typing import Any

from aws_sdk_sagemaker_runtime_http2.models import (
    RequestPayloadPart,
    RequestStreamEventPayloadPart,
)

from tts_client._bidi_transport import (
    BIDI_SAMPLE_RATE,
    _build_bidi_message,
    _close_quietly,
    _drain_until_complete,
    _open_bidi_stream,
    _pcm_to_wav,
)
from tts_client.errors import ServerError
from tts_client.types import SynthesisChunk

#: This client's own sentence splitter for incremental streaming. Kept local
#: rather than imported from a container: this package has no dependency on
#: speech-infra's containers.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_CLOSE_MESSAGE = json.dumps({"type": "close"}).encode("utf-8")


class IncrementalSentenceChunker:
    """Buffers fed text and emits complete sentences as boundaries appear.

    ``feed()`` is safe to call repeatedly with arbitrarily-sized fragments;
    the last (possibly incomplete) sentence is always held back until either
    a later ``feed()`` completes it or ``flush()`` is called at end-of-stream.
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, fragment: str) -> list[str]:
        """Add ``fragment`` to the buffer, returning any newly-completed sentences."""
        self._buffer += fragment
        parts = _SENTENCE_RE.split(self._buffer)
        if len(parts) == 1:
            return []
        self._buffer = parts[-1]
        return [s.strip() for s in parts[:-1] if s.strip()]

    def flush(self) -> str | None:
        """Return and clear whatever remains in the buffer, or ``None`` if empty.

        This is the end-of-stream step: it is what guarantees a final
        sentence with no trailing terminal punctuation still gets sent
        instead of silently dropped.
        """
        remainder = self._buffer.strip()
        self._buffer = ""
        return remainder or None


def split_sentences(text: str) -> list[str]:
    """Split a complete string into sentences, via the same chunker used for
    incremental input -- so the two input shapes are provably consistent."""
    chunker = IncrementalSentenceChunker()
    sentences = chunker.feed(text.strip())
    tail = chunker.flush()
    if tail is not None:
        sentences.append(tail)
    return sentences


def concat_chunks_to_wav(chunks: list[SynthesisChunk]) -> bytes:
    """Concatenate chunks' raw PCM, in order, into one WAV file."""
    pcm = b"".join(chunk.audio_bytes for chunk in chunks)
    return _pcm_to_wav(pcm)


async def _stream_chunks_async(
    region: str,
    endpoint: str,
    voice: str,
    text_source: str | Iterable[str],
    speed: float,
    t0: float,
    sample_rate: int | None = None,
) -> AsyncGenerator[SynthesisChunk, None]:
    fragments: Iterable[str] = (text_source,) if isinstance(text_source, str) else text_source

    resolved_rate = sample_rate if sample_rate is not None else BIDI_SAMPLE_RATE

    # Opened lazily, on the first sentence actually ready to send -- an empty
    # or all-whitespace source should raise without ever touching the network.
    stream: Any = None
    output_stream = None
    chunker = IncrementalSentenceChunker()
    seq = 0

    async def _send_sentence(sentence: str) -> SynthesisChunk:
        nonlocal stream, output_stream, seq
        if stream is None:
            stream = await _open_bidi_stream(region, endpoint)
        message = _build_bidi_message(sentence, voice, None, speed=speed, sample_rate=sample_rate)
        await stream.input_stream.send(
            RequestStreamEventPayloadPart(value=RequestPayloadPart(bytes_=message))
        )
        if output_stream is None:
            _, output_stream = await stream.await_output()
        drained = await _drain_until_complete(output_stream, t0)
        if not drained.ended_cleanly:
            raise ServerError(
                f"bidi session ended before chunk {seq} ({sentence[:40]!r}) completed"
            )
        chunk = SynthesisChunk(
            seq=seq,
            text=sentence,
            audio_bytes=drained.pcm_bytes,
            ttfab_ms=drained.ttfab_ms if seq == 0 else None,
            duration_s=len(drained.pcm_bytes) / (resolved_rate * 2),
            sample_rate=resolved_rate,
        )
        seq += 1
        return chunk

    try:
        for fragment in fragments:
            for sentence in chunker.feed(fragment):
                yield await _send_sentence(sentence)

        remainder = chunker.flush()
        if remainder is not None:
            yield await _send_sentence(remainder)

        if stream is None:
            raise ValueError("text_source produced no sentences to synthesize")

        await stream.input_stream.send(
            RequestStreamEventPayloadPart(value=RequestPayloadPart(bytes_=_CLOSE_MESSAGE))
        )
    finally:
        if stream is not None:
            await _close_quietly(stream)


class BidiChunkStream:
    """Sync iterator bridging one bidi session across several text chunks.

    Constructed via :meth:`TTSClient.synthesize_bidi_stream`. Each `next()`
    drives one send/drain round-trip on a single event loop kept alive for
    the whole session, since the HTTP/2 stream and its lock/state need to
    persist across chunks -- a fresh ``asyncio.run()`` per chunk (the pattern
    every other client method in this package uses) would tear the
    connection down between chunks.

    Use as a context manager, or exhaust the iterator, so the underlying
    connection and event loop close deterministically rather than depending
    on garbage-collection timing:

        with client.synthesize_bidi_stream(endpoint, voice, text) as stream:
            for chunk in stream:
                ...
    """

    def __init__(
        self,
        region: str,
        endpoint: str,
        voice: str,
        text_source: str | Iterable[str],
        speed: float,
        sample_rate: int | None = None,
    ) -> None:
        self._region = region
        self._endpoint = endpoint
        self._voice = voice
        self._text_source = text_source
        self._speed = speed
        self._sample_rate = sample_rate
        self._loop = asyncio.new_event_loop()
        self._agen: AsyncGenerator[SynthesisChunk, None] | None = None
        self._closed = False

    def __iter__(self) -> BidiChunkStream:
        return self

    def __next__(self) -> SynthesisChunk:
        if self._closed:
            raise StopIteration
        if self._agen is None:
            t0 = time.perf_counter()
            self._agen = _stream_chunks_async(
                self._region,
                self._endpoint,
                self._voice,
                self._text_source,
                self._speed,
                t0,
                self._sample_rate,
            )
        try:
            return self._loop.run_until_complete(self._agen.__anext__())
        except StopAsyncIteration:
            self.close()
            raise StopIteration from None
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> BidiChunkStream:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Idempotent: closes the bidi connection (if opened) and the event loop.

        Replicates the teardown ``asyncio.run()`` does for a one-shot
        coroutine (cancel any tasks the SDK scheduled on this loop, then
        ``shutdown_asyncgens``, then close) -- a bare ``run_until_complete()``
        + ``close()`` skips that step, which otherwise surfaces as "Task was
        destroyed but it is pending!" warnings from a background task the
        HTTP/2 SDK leaves scheduled on the loop.
        """
        if self._closed:
            return
        self._closed = True
        if self._agen is not None:
            self._loop.run_until_complete(self._agen.aclose())
        pending = asyncio.all_tasks(loop=self._loop)
        if pending:
            for task in pending:
                task.cancel()
            self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        self._loop.run_until_complete(self._loop.shutdown_asyncgens())
        self._loop.close()

    def __del__(self) -> None:
        if not self._closed:
            warnings.warn(
                "BidiChunkStream was never closed -- use it as a context manager "
                "or exhaust the iterator so the underlying connection and event "
                "loop close deterministically.",
                stacklevel=2,
            )
            try:
                self.close()
            except Exception:  # noqa: BLE001 - a __del__ must never raise
                pass
