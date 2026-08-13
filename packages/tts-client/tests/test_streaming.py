"""Tests for incremental text-chunk streaming over one bidi session.

Fakes follow the same pattern as ``test_client.py``'s bidi tests (patch
``tts_client._bidi_transport.SageMakerRuntimeHTTP2Client`` at the constructor
call site), extended to support several send/receive round-trips on one
session rather than just one.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from aws_sdk_sagemaker_runtime_http2.models import (
    InternalStreamFailure,
    ModelStreamError,
    ResponsePayloadPart,
    ResponseStreamEventInternalStreamFailure,
    ResponseStreamEventModelStreamError,
    ResponseStreamEventPayloadPart,
)

from tts_client.client import TTSClient
from tts_client.errors import ServerError, TTSClientError
from tts_client.streaming import IncrementalSentenceChunker, split_sentences
from tts_client.types import SampleRate

PCM_100MS = b"\x00\x01" * 2400


def _payload_event(data: bytes):
    return ResponseStreamEventPayloadPart(value=ResponsePayloadPart(bytes_=data))


def _control_frame(**fields):
    return _payload_event(json.dumps(fields).encode("utf-8"))


class _FakeInputStream:
    def __init__(self, *, on_send=None) -> None:
        self.sent: list = []
        self.closed = False
        self._on_send = on_send

    async def send(self, event) -> None:
        self.sent.append(event)
        if self._on_send is not None:
            self._on_send(len(self.sent))

    async def close(self) -> None:
        self.closed = True


class _FakeOutputStream:
    """One shared events queue: several ``_drain_until_complete`` calls each
    consume up to their own ``synthesis_complete``, then the next call picks
    up wherever the previous one left off -- exactly like one real socket
    carrying several sequential responses.
    """

    def __init__(self, events: list, *, on_receive=None) -> None:
        self._events = list(events)
        self._on_receive = on_receive
        self.receives = 0

    async def receive(self):
        self.receives += 1
        if self._on_receive is not None:
            self._on_receive(self.receives)
        if not self._events:
            return None
        return self._events.pop(0)


class _FakeStream:
    def __init__(self, events: list, *, on_receive=None, on_send=None) -> None:
        self.input_stream = _FakeInputStream(on_send=on_send)
        self.output_stream = _FakeOutputStream(events, on_receive=on_receive)
        self.closed = False
        self.await_output_calls = 0

    async def await_output(self):
        self.await_output_calls += 1
        return (object(), self.output_stream)

    async def close(self) -> None:
        self.closed = True


class _FakeBidiClientCtor:
    """Replaces ``SageMakerRuntimeHTTP2Client`` at the constructor call site."""

    def __init__(self, stream: _FakeStream) -> None:
        self._stream = stream
        self.config = None
        self.construct_count = 0

    def __call__(self, config=None):
        self.config = config
        self.construct_count += 1
        return self

    async def invoke_endpoint_with_bidirectional_stream(self, input_):
        return self._stream


def _sentence_events(*sentences: bytes) -> list:
    """One PCM chunk + ``synthesis_complete`` per sentence, concatenated."""
    events: list = []
    for pcm in sentences:
        events.append(_payload_event(pcm))
        events.append(_control_frame(type="synthesis_complete"))
    return events


def _sent_texts(stream: _FakeStream) -> list[str]:
    return [
        json.loads(event.value.bytes_.decode("utf-8")).get("text")
        for event in stream.input_stream.sent
    ]


def _sent_types(stream: _FakeStream) -> list[str | None]:
    return [
        json.loads(event.value.bytes_.decode("utf-8")).get("type")
        for event in stream.input_stream.sent
    ]


class TestIncrementalSentenceChunker:
    def test_fragment_with_no_terminal_punctuation_holds_everything(self) -> None:
        chunker = IncrementalSentenceChunker()
        assert chunker.feed("Hello there") == []

    def test_later_fragment_completes_the_held_sentence(self) -> None:
        chunker = IncrementalSentenceChunker()
        assert chunker.feed("Hello ") == []
        assert chunker.feed("there. And more") == ["Hello there."]

    def test_single_feed_with_several_sentences_returns_all_at_once(self) -> None:
        chunker = IncrementalSentenceChunker()
        assert chunker.feed("One. Two. Three incomplete") == ["One.", "Two."]

    def test_flush_returns_the_remainder_once(self) -> None:
        chunker = IncrementalSentenceChunker()
        chunker.feed("trailing fragment")
        assert chunker.flush() == "trailing fragment"
        assert chunker.flush() is None

    def test_flush_on_empty_buffer_returns_none(self) -> None:
        assert IncrementalSentenceChunker().flush() is None


class TestSplitSentences:
    def test_matches_feed_then_flush(self) -> None:
        text = "One. Two! Three? Four with no terminator"
        chunker = IncrementalSentenceChunker()
        expected = chunker.feed(text)
        tail = chunker.flush()
        if tail is not None:
            expected.append(tail)
        assert split_sentences(text) == expected

    def test_multi_sentence_text(self) -> None:
        assert split_sentences("Hello there. How are you? Fine!") == [
            "Hello there.",
            "How are you?",
            "Fine!",
        ]


class TestSynthesizeBidiStream:
    def test_whole_string_input_sends_one_chunk_per_sentence(self) -> None:
        events = _sentence_events(PCM_100MS, PCM_100MS)
        stream = _FakeStream(events)
        fake_ctor = _FakeBidiClientCtor(stream)

        with patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor):
            client = TTSClient()
            with client.synthesize_bidi_stream(
                "speech-kokoro-82m", "af_heart", "First sentence. Second sentence."
            ) as bidi_stream:
                chunks = list(bidi_stream)

        assert [c.text for c in chunks] == ["First sentence.", "Second sentence."]
        assert [c.seq for c in chunks] == [0, 1]
        assert _sent_texts(stream) == ["First sentence.", "Second sentence.", None]
        assert _sent_types(stream) == [None, None, "close"]
        assert stream.closed

    def test_fragmented_input_sends_sentences_as_they_complete(self) -> None:
        events = _sentence_events(PCM_100MS, PCM_100MS)
        stream = _FakeStream(events)
        fake_ctor = _FakeBidiClientCtor(stream)

        def fragments():
            yield "First "
            yield "sentence. Second sent"
            yield "ence."

        with patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor):
            client = TTSClient()
            with client.synthesize_bidi_stream(
                "speech-kokoro-82m", "af_heart", fragments()
            ) as bidi_stream:
                chunks = list(bidi_stream)

        assert [c.text for c in chunks] == ["First sentence.", "Second sentence."]

    def test_trailing_fragment_with_no_terminator_sent_via_flush(self) -> None:
        events = _sentence_events(PCM_100MS, PCM_100MS)
        stream = _FakeStream(events)
        fake_ctor = _FakeBidiClientCtor(stream)

        def fragments():
            yield "Complete sentence."
            yield " trailing fragment with no terminator"

        with patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor):
            client = TTSClient()
            with client.synthesize_bidi_stream(
                "speech-kokoro-82m", "af_heart", fragments()
            ) as bidi_stream:
                chunks = list(bidi_stream)

        assert [c.text for c in chunks] == [
            "Complete sentence.",
            "trailing fragment with no terminator",
        ]

    def test_chunk_two_is_not_sent_before_chunk_one_completes(self) -> None:
        send_counts_at_receive: list[int] = []
        events = _sentence_events(PCM_100MS, PCM_100MS)
        stream = _FakeStream(
            events,
            on_receive=lambda _n: send_counts_at_receive.append(len(stream.input_stream.sent)),
        )
        fake_ctor = _FakeBidiClientCtor(stream)

        with patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor):
            client = TTSClient()
            with client.synthesize_bidi_stream(
                "speech-kokoro-82m", "af_heart", "First one. Second one."
            ) as bidi_stream:
                list(bidi_stream)

        # Two receives happen per chunk (audio, then synthesis_complete); by
        # the time either of chunk 2's receives fires, only chunk 1's message
        # (not chunk 2's) has been sent -- proving strict ordering.
        assert send_counts_at_receive[:2] == [1, 1]
        assert send_counts_at_receive[2:4] == [2, 2]

    def test_ttfab_set_only_on_first_chunk(self) -> None:
        events = _sentence_events(PCM_100MS, PCM_100MS)
        stream = _FakeStream(events)
        fake_ctor = _FakeBidiClientCtor(stream)

        with patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor):
            client = TTSClient()
            with client.synthesize_bidi_stream(
                "speech-kokoro-82m", "af_heart", "First. Second."
            ) as bidi_stream:
                chunks = list(bidi_stream)

        assert chunks[0].ttfab_ms is not None
        assert chunks[1].ttfab_ms is None

    def test_await_output_called_only_once_across_the_whole_session(self) -> None:
        events = _sentence_events(PCM_100MS, PCM_100MS, PCM_100MS)
        stream = _FakeStream(events)
        fake_ctor = _FakeBidiClientCtor(stream)

        with patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor):
            client = TTSClient()
            with client.synthesize_bidi_stream(
                "speech-kokoro-82m", "af_heart", "One. Two. Three."
            ) as bidi_stream:
                list(bidi_stream)

        assert stream.await_output_calls == 1
        assert fake_ctor.construct_count == 1

    def test_mid_session_error_on_second_chunk_raises_and_closes(self) -> None:
        events = [
            _payload_event(PCM_100MS),
            _control_frame(type="synthesis_complete"),
            ResponseStreamEventModelStreamError(value=ModelStreamError("boom")),
        ]
        stream = _FakeStream(events)
        fake_ctor = _FakeBidiClientCtor(stream)

        with (
            patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor),
            pytest.raises(ServerError),
        ):
            client = TTSClient()
            with client.synthesize_bidi_stream(
                "speech-kokoro-82m", "af_heart", "First. Second."
            ) as bidi_stream:
                list(bidi_stream)

        assert stream.closed

    def test_connection_drop_before_all_chunks_sent_raises(self) -> None:
        # Only one sentence's worth of events for a two-sentence input: the
        # second drain sees `event is None` with no synthesis_complete.
        events = _sentence_events(PCM_100MS)
        stream = _FakeStream(events)
        fake_ctor = _FakeBidiClientCtor(stream)

        with (
            patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor),
            pytest.raises(ServerError, match="ended before chunk"),
        ):
            client = TTSClient()
            with client.synthesize_bidi_stream(
                "speech-kokoro-82m", "af_heart", "First. Second."
            ) as bidi_stream:
                list(bidi_stream)

    def test_empty_string_raises_value_error(self) -> None:
        fake_ctor = _FakeBidiClientCtor(_FakeStream([]))

        with (
            patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor),
            pytest.raises(ValueError, match="no sentences"),
        ):
            client = TTSClient()
            with client.synthesize_bidi_stream(
                "speech-kokoro-82m", "af_heart", "   "
            ) as bidi_stream:
                list(bidi_stream)

    def test_empty_fragment_source_raises_value_error(self) -> None:
        fake_ctor = _FakeBidiClientCtor(_FakeStream([]))

        with (
            patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor),
            pytest.raises(ValueError, match="no sentences"),
        ):
            client = TTSClient()
            with client.synthesize_bidi_stream(
                "speech-kokoro-82m", "af_heart", iter([])
            ) as bidi_stream:
                list(bidi_stream)

    def test_empty_source_never_opens_a_connection(self) -> None:
        fake_ctor = _FakeBidiClientCtor(_FakeStream([]))

        with (
            patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor),
            pytest.raises(ValueError),
        ):
            client = TTSClient()
            with client.synthesize_bidi_stream("speech-kokoro-82m", "af_heart", "") as bidi_stream:
                list(bidi_stream)

        assert fake_ctor.construct_count == 0

    def test_early_break_still_closes_the_session(self) -> None:
        events = _sentence_events(PCM_100MS, PCM_100MS, PCM_100MS)
        stream = _FakeStream(events)
        fake_ctor = _FakeBidiClientCtor(stream)

        with patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor):
            client = TTSClient()
            with client.synthesize_bidi_stream(
                "speech-kokoro-82m", "af_heart", "One. Two. Three."
            ) as bidi_stream:
                for chunk in bidi_stream:
                    if chunk.seq == 0:
                        break

        assert stream.closed
        assert bidi_stream._loop.is_closed()
        # The close frame is only sent once the source is exhausted -- an
        # early break must not send it.
        assert "close" not in _sent_types(stream)

    def test_speed_is_sent_on_every_chunk(self) -> None:
        events = _sentence_events(PCM_100MS, PCM_100MS)
        stream = _FakeStream(events)
        fake_ctor = _FakeBidiClientCtor(stream)

        with patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor):
            client = TTSClient()
            with client.synthesize_bidi_stream(
                "speech-kokoro-82m", "af_heart", "One. Two.", speed=1.5
            ) as bidi_stream:
                list(bidi_stream)

        speeds = [
            json.loads(event.value.bytes_.decode("utf-8")).get("speed")
            for event in stream.input_stream.sent
            if json.loads(event.value.bytes_.decode("utf-8")).get("type") != "close"
        ]
        assert speeds == [1.5, 1.5]

    def test_sample_rate_is_sent_on_every_chunk_when_set(self) -> None:
        events = _sentence_events(PCM_100MS, PCM_100MS)
        stream = _FakeStream(events)
        fake_ctor = _FakeBidiClientCtor(stream)

        with patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor):
            client = TTSClient()
            with client.synthesize_bidi_stream(
                "speech-kokoro-82m", "af_heart", "One. Two.", sample_rate=SampleRate.HZ_16000
            ) as bidi_stream:
                chunks = list(bidi_stream)

        rates = [
            json.loads(event.value.bytes_.decode("utf-8")).get("sample_rate")
            for event in stream.input_stream.sent
            if json.loads(event.value.bytes_.decode("utf-8")).get("type") != "close"
        ]
        assert rates == [16000, 16000]
        assert [c.sample_rate for c in chunks] == [16000, 16000]

    def test_sample_rate_omitted_from_every_chunk_by_default(self) -> None:
        events = _sentence_events(PCM_100MS, PCM_100MS)
        stream = _FakeStream(events)
        fake_ctor = _FakeBidiClientCtor(stream)

        with patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor):
            client = TTSClient()
            with client.synthesize_bidi_stream(
                "speech-kokoro-82m", "af_heart", "One. Two."
            ) as bidi_stream:
                chunks = list(bidi_stream)

        sent_messages = [
            json.loads(event.value.bytes_.decode("utf-8"))
            for event in stream.input_stream.sent
            if json.loads(event.value.bytes_.decode("utf-8")).get("type") != "close"
        ]
        assert all("sample_rate" not in m for m in sent_messages)
        assert [c.sample_rate for c in chunks] == [24000, 24000]

    def test_mid_session_internal_stream_failure_raises_server_error(self) -> None:
        events = [
            _payload_event(PCM_100MS),
            _control_frame(type="synthesis_complete"),
            ResponseStreamEventInternalStreamFailure(value=InternalStreamFailure("infra blip")),
        ]
        stream = _FakeStream(events)
        fake_ctor = _FakeBidiClientCtor(stream)

        with (
            patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor),
            pytest.raises(ServerError),
        ):
            client = TTSClient()
            with client.synthesize_bidi_stream(
                "speech-kokoro-82m", "af_heart", "First. Second."
            ) as bidi_stream:
                list(bidi_stream)

    def test_error_frame_raises_typed_error(self) -> None:
        events = [
            _payload_event(PCM_100MS),
            _control_frame(type="synthesis_complete"),
            _control_frame(type="error", message="queue_saturated: try later"),
        ]
        stream = _FakeStream(events)
        fake_ctor = _FakeBidiClientCtor(stream)

        with (
            patch("tts_client._bidi_transport.SageMakerRuntimeHTTP2Client", fake_ctor),
            pytest.raises(TTSClientError),
        ):
            client = TTSClient()
            with client.synthesize_bidi_stream(
                "speech-kokoro-82m", "af_heart", "First. Second."
            ) as bidi_stream:
                list(bidi_stream)
