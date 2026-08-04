"""Regression tests for the Kokoro GPU container's streaming path.

KPipeline yields torch.Tensor audio segments. A prior bug called the numpy-only
`.astype(np.int16)` directly on those tensors inside the streaming generator,
raising AttributeError *after* the HTTP 200 + chunked headers were already sent.
That truncated the chunked response body, which SageMaker's streaming proxy
surfaces as `ModelStreamError` (and plain `invoke_endpoint` as `ServerError(0)`).

These tests drive the real Starlette app with a fake pipeline that mimics the
model by yielding torch tensors, and assert the streamed WAV body is produced in
full (multiple chunks, valid RIFF header) without raising.

They also cover the MP3 wire format, including the lameenc constraints that
path depends on: the flush tail carries real audio, short utterances produce
output only via flush, and an encoder cannot be reused after flush.

Startup warm-up and the bidirectional binary-frame path are covered too: the
former keeps first-inference cost off the first real request, the latter is the
frame type SageMaker actually delivers.

The container deps (torch, starlette, kokoro) are not installed in the lean infra
dev environment, so the module self-skips there. It runs inside the container
image, where those deps exist.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _find_serve_py() -> Path | None:
    """Locate the kokoro serve.py in the repo layout or the container layout."""
    candidates = [
        Path(__file__).resolve().parents[1] / "containers" / "kokoro" / "serve.py",
        Path("/app/serve.py"),
    ]
    return next((p for p in candidates if p.is_file()), None)


_SERVE_PY = _find_serve_py()

_REQUIRED_DEPS = ("torch", "starlette", "lameenc", "uvicorn")

# Every module serve.py imports at top level must be probed: a missing one raises
# ModuleNotFoundError at collection instead of skipping this file.
_HAS_DEPS = _SERVE_PY is not None and all(
    importlib.util.find_spec(mod) is not None for mod in _REQUIRED_DEPS
)

pytestmark = pytest.mark.skipif(
    not _HAS_DEPS,
    reason=f"kokoro container deps ({', '.join(_REQUIRED_DEPS)}) not installed in this env",
)

SEGMENT_SAMPLES = 2400
SEGMENT_COUNT = 3
SAMPLE_RATE = 24000


def _load_serve(segment_samples: int = SEGMENT_SAMPLES, segment_count: int = SEGMENT_COUNT):
    """Import the container's serve.py with a fake tensor-yielding KPipeline.

    The fake records the text of every synthesis it is asked for on
    `module.pipeline_calls`, which is what lets the warm-up tests below observe that
    an inference happened during lifespan rather than only on the first request.
    """
    import torch

    kokoro_mod = types.ModuleType("kokoro")
    calls: list[str] = []

    class _FakePipeline:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __call__(self, text, voice=None, speed=1.0):
            calls.append(text)
            # Real KPipeline yields (graphemes, phonemes, torch.Tensor) per segment.
            for _ in range(segment_count):
                yield ("g", "p", torch.linspace(-0.1, 0.1, segment_samples))

    kokoro_mod.KPipeline = _FakePipeline
    sys.modules["kokoro"] = kokoro_mod

    spec = importlib.util.spec_from_file_location("kokoro_serve_under_test", _SERVE_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.pipeline_calls = calls
    return module


def _unload_serve() -> None:
    sys.modules.pop("kokoro", None)
    sys.modules.pop("kokoro_serve_under_test", None)


@pytest.fixture
def serve_module():
    module = _load_serve()
    try:
        yield module
    finally:
        _unload_serve()


@pytest.fixture
def make_serve_module():
    """Build a serve module with custom segment geometry (for MP3 frame-buffer cases)."""
    created = False

    def _factory(segment_samples: int, segment_count: int):
        nonlocal created
        created = True
        return _load_serve(segment_samples, segment_count)

    try:
        yield _factory
    finally:
        if created:
            _unload_serve()


def _expected_mp3_bytes(segment_samples: int, segment_count: int) -> int:
    """Encode the same audio the fake pipeline yields, to get an exact byte target.

    Asserting on exact length is what catches a dropped flush tail: the tail is
    real audio (512-590 B here), and a "len > 100" assertion passes without it.
    """
    import lameenc
    import numpy as np

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(48)
    encoder.set_in_sample_rate(SAMPLE_RATE)
    encoder.set_channels(1)
    encoder.set_quality(2)
    encoder.silence()

    segment = np.linspace(-0.1, 0.1, segment_samples).astype(np.float32)
    pcm = (segment * 32767).astype(np.int16).tobytes()
    body = b"".join(bytes(encoder.encode(pcm)) for _ in range(segment_count))
    return len(body) + len(bytes(encoder.flush()))


def test_streaming_invocations_returns_full_wav(serve_module) -> None:
    from starlette.testclient import TestClient

    with TestClient(serve_module.app) as client:
        resp = client.post(
            "/invocations",
            json={"text": "One. Two. Three.", "voice": "af_heart", "stream": True},
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    body = resp.content
    assert body[:4] == b"RIFF"
    # 3 segments * 2400 samples * 2 bytes + 44-byte header.
    assert len(body) == 44 + 3 * 2400 * 2


def test_to_numpy_converts_torch_tensor(serve_module) -> None:
    import numpy as np
    import torch

    out = serve_module._to_numpy(torch.linspace(-1.0, 1.0, 8))
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    # The failing call site: this must not raise AttributeError.
    pcm = (out * 32767).astype(np.int16).tobytes()
    assert len(pcm) == 8 * 2


def test_default_request_is_unchanged_chunked_wav(serve_module) -> None:
    """The eval harness sends no transport/format and parses raw WAV. Guard that."""
    from starlette.testclient import TestClient

    with TestClient(serve_module.app) as client:
        resp = client.post("/invocations", json={"text": "One. Two. Three."})

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content[:4] == b"RIFF"
    assert len(resp.content) == 44 + SEGMENT_COUNT * SEGMENT_SAMPLES * 2


def test_sync_wav_has_real_sizes_not_placeholders(serve_module) -> None:
    """stream=false is what static asset generation uses; sizes must be correct."""
    import struct

    from starlette.testclient import TestClient

    with TestClient(serve_module.app) as client:
        resp = client.post("/invocations", json={"text": "One.", "stream": False})

    body = resp.content
    assert body[:4] == b"RIFF"
    riff_size = struct.unpack_from("<I", body, 4)[0]
    data_size = struct.unpack_from("<I", body, 40)[0]
    assert riff_size != 0xFFFFFFFF
    assert data_size != 0xFFFFFFFF
    assert data_size == len(body) - 44
    assert riff_size == len(body) - 8


def test_binary_mp3_returns_audio_mpeg(serve_module) -> None:
    from starlette.testclient import TestClient

    with TestClient(serve_module.app) as client:
        resp = client.post("/invocations", json={"text": "One. Two. Three.", "format": "mp3"})

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    # Bare MP3 frame sync: 11 set bits, no ID3 or Xing wrapper.
    assert int.from_bytes(resp.content[:2], "big") & 0xFFE0 == 0xFFE0
    # Exact length: a dropped flush tail would still pass a "> 100" check.
    assert len(resp.content) == _expected_mp3_bytes(SEGMENT_SAMPLES, SEGMENT_COUNT)


def test_short_utterance_still_produces_mp3_via_flush(make_serve_module) -> None:
    """A 50ms segment yields nothing from encode(); the flush tail carries all audio."""
    from starlette.testclient import TestClient

    module = make_serve_module(1200, 1)
    with TestClient(module.app) as client:
        resp = client.post("/invocations", json={"text": "Hi.", "format": "mp3"})

    audio = resp.content
    assert audio, "short utterance produced no audio at all"
    assert int.from_bytes(audio[:2], "big") & 0xFFE0 == 0xFFE0
    # Most of a 50ms utterance arrives only in the flush tail (590 of 720 bytes).
    assert len(audio) == _expected_mp3_bytes(1200, 1)


def test_sequential_mp3_requests_each_get_fresh_encoder(serve_module) -> None:
    """lameenc raises 'Encoder not initialised' if reused after flush."""
    from starlette.testclient import TestClient

    with TestClient(serve_module.app) as client:
        first = client.post("/invocations", json={"text": "One.", "format": "mp3"})
        second = client.post("/invocations", json={"text": "One.", "format": "mp3"})

    assert first.status_code == second.status_code == 200
    assert len(first.content) > 100
    assert len(second.content) == len(first.content)


def test_mp3_encoder_rejects_use_after_flush(serve_module) -> None:
    import numpy as np

    encoder = serve_module.Mp3StreamEncoder()
    encoder.encode(np.zeros(2400, dtype=np.float32))
    encoder.flush()

    assert encoder.flush() == b""
    with pytest.raises(RuntimeError, match="already flushed"):
        encoder.encode(np.zeros(2400, dtype=np.float32))


def test_mp3_flush_tail_carries_audio(serve_module) -> None:
    """The tail is real audio, not padding, so it must never be dropped."""
    import numpy as np

    samples = np.linspace(-0.5, 0.5, 24000, dtype=np.float32)
    encoder = serve_module.Mp3StreamEncoder()
    body = encoder.encode(samples)
    tail = encoder.flush()

    assert len(tail) > 0
    assert len(body) + len(tail) > len(body)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"text": "One.", "format": "flac"}, "format"),
        ({"text": ""}, "text"),
    ],
)
def test_invalid_requests_are_rejected(serve_module, payload, expected) -> None:
    from starlette.testclient import TestClient

    with TestClient(serve_module.app) as client:
        resp = client.post("/invocations", json=payload)

    assert resp.status_code == 400
    assert expected in resp.json()["error"]


class TestStartupWarmup:
    """The warm-up moves first-inference cost out of the first real request.

    `/ping` answering 200 proves the pipeline is *loaded*, not that CUDA kernel
    autotune has run. Without a warm-up the first caller after a scale-out absorbs
    it — which is exactly the request arriving into a capacity shortfall.
    """

    def test_warmup_returns_a_real_sample_count(self, serve_module) -> None:
        # Guards the tests below from passing for the wrong reason: the lifespan
        # swallows warm-up failures, so a broken `_warmup` would leave them green.
        assert serve_module._warmup() == 3 * 2400

    def test_runs_one_synthesis_during_lifespan(self, serve_module, capfd) -> None:
        from starlette.testclient import TestClient

        with TestClient(serve_module.app):
            # No request has been made yet, so anything here came from lifespan.
            assert serve_module.pipeline_calls == [serve_module.WARMUP_TEXT]

        assert "Warm-up complete" in capfd.readouterr().out

    def test_warmup_output_is_discarded_not_served(self, serve_module) -> None:
        from starlette.testclient import TestClient

        with TestClient(serve_module.app) as client:
            resp = client.post("/invocations", json={"text": "Hello.", "stream": True})

        assert resp.status_code == 200
        # The warm-up audio must not be prepended to the first real response.
        assert len(resp.content) == 44 + 3 * 2400 * 2
        assert serve_module.pipeline_calls == [serve_module.WARMUP_TEXT, "Hello."]

    def test_serves_traffic_when_warmup_raises(self, serve_module, monkeypatch) -> None:
        # A container that cannot warm up can still serve; failing startup over it
        # would turn a latency problem into an outage.
        #
        # `_warmup` rather than `_synthesize_full`: patching the latter would break the
        # non-streaming serving path too, so a green result would prove less.
        monkeypatch.setattr(
            serve_module,
            "_warmup",
            lambda: (_ for _ in ()).throw(RuntimeError("CUDA OOM during warm-up")),
        )

        from starlette.testclient import TestClient

        with TestClient(serve_module.app) as client:
            resp = client.post("/invocations", json={"text": "Hello.", "stream": True})

        assert resp.status_code == 200
        assert resp.content[:4] == b"RIFF"

    def test_emits_the_full_stage_sequence(self, serve_module, capfd) -> None:
        from starlette.testclient import TestClient

        from shared.stages import Stage, parse_stage_markers

        with TestClient(serve_module.app):
            pass

        markers = parse_stage_markers(capfd.readouterr().out.splitlines())
        assert [m.name for m in markers] == [
            Stage.FRAMEWORK_INIT,
            Stage.WEIGHTS_READY,
            Stage.WARMUP_DONE,
            Stage.READY,
        ]
        # Monotonic non-decreasing: a stage that appears to finish before the one
        # before it makes the attribution `ttotal` produces nonsense.
        assert [m.elapsed_s for m in markers] == sorted(m.elapsed_s for m in markers)

    def test_marks_warmup_done_even_when_warmup_failed(
        self, serve_module, monkeypatch, capfd
    ) -> None:
        # The marker is emitted regardless so `ttotal`'s sequence stays complete and
        # the warm-up cost stays visible even when the warm-up itself did not work.
        from starlette.testclient import TestClient

        from shared.stages import Stage, parse_stage_markers

        monkeypatch.setattr(serve_module, "_warmup", lambda: (_ for _ in ()).throw(RuntimeError()))

        with TestClient(serve_module.app):
            pass

        names = [m.name for m in parse_stage_markers(capfd.readouterr().out.splitlines())]
        assert Stage.WARMUP_DONE in names
        assert names[-1] == Stage.READY


class TestBidirectionalBinaryFrames:
    """SageMaker sends binary frames; the handler must not require text ones.

    ``invoke_endpoint_with_bidirectional_stream`` forwards each
    ``RequestPayloadPart`` as a *binary* WebSocket frame. Starlette's
    ``receive_text()`` reads ``message["text"]`` unconditionally, so it raised
    ``KeyError: 'text'`` on every production request — surfacing to the client as
    an error frame whose message was the literal string ``'text'``, and to a
    benchmark as a zero-audio "success" if error frames are not classified.

    Both frame types are asserted here because the local browser demo and
    ``TestClient`` send text while SageMaker sends binary; a fix that swapped
    ``receive_text`` for ``receive_bytes`` would just invert the bug.
    """

    def _synthesize(self, serve_module, *, binary: bool) -> tuple[list[dict], int]:
        """Drive one bidi request, returning (control frames, total audio bytes)."""
        import json

        from starlette.testclient import TestClient

        message = json.dumps({"text": "One. Two.", "voice": "af_heart", "request_id": "r1"})

        frames: list[dict] = []
        audio = 0
        with TestClient(serve_module.app) as client:
            with client.websocket_connect("/invocations-bidirectional-stream") as ws:
                if binary:
                    ws.send_bytes(message.encode("utf-8"))
                else:
                    ws.send_text(message)
                while True:
                    received = ws.receive()
                    if received["type"] == "websocket.close":
                        break
                    if received.get("text") is not None:
                        frame = json.loads(received["text"])
                        frames.append(frame)
                        if frame["type"] in ("synthesis_complete", "error"):
                            break
                    elif received.get("bytes"):
                        audio += len(received["bytes"])
        return frames, audio

    def test_a_binary_frame_synthesizes_audio(self, serve_module) -> None:
        # The exact shape SageMaker delivers. Before the fix this returned an
        # error frame with message "'text'" and zero audio.
        frames, audio = self._synthesize(serve_module, binary=True)

        types_seen = [f["type"] for f in frames]
        assert "error" not in types_seen, f"error frame: {frames}"
        assert types_seen == ["synthesis_start", "synthesis_complete"]
        # 3 fake segments * 2400 samples * 2 bytes, raw PCM with no RIFF header.
        assert audio == 3 * 2400 * 2

    def test_a_text_frame_still_works(self, serve_module) -> None:
        frames, audio = self._synthesize(serve_module, binary=False)

        assert [f["type"] for f in frames] == ["synthesis_start", "synthesis_complete"]
        assert audio == 3 * 2400 * 2

    def test_a_binary_frame_reaches_the_model_with_its_text_intact(self, serve_module) -> None:
        # Not just "no error": the decoded payload must be what the model
        # synthesizes, so a mis-decode cannot pass as success.
        self._synthesize(serve_module, binary=True)

        assert serve_module.pipeline_calls[-1] == "One. Two."

    def test_an_empty_binary_frame_is_rejected_as_missing_text(self, serve_module) -> None:
        # Decoding to "" must take the handler's own "text required" path rather
        # than raising, which would look identical to the bug being fixed.
        import json

        from starlette.testclient import TestClient

        with TestClient(serve_module.app) as client:
            with client.websocket_connect("/invocations-bidirectional-stream") as ws:
                ws.send_bytes(json.dumps({"request_id": "r1"}).encode("utf-8"))
                frame = json.loads(ws.receive_text())

        assert frame["type"] == "error"
        assert frame["message"] == "text required"
        assert frame["request_id"] == "r1"

    def test_client_disconnect_closes_cleanly(self, serve_module) -> None:
        # _receive_message translates a disconnect into WebSocketDisconnect, which
        # the handler already treats as a normal end of session. Without that
        # translation it would decode a missing payload and log a spurious error.
        from starlette.testclient import TestClient

        with TestClient(serve_module.app) as client:
            with client.websocket_connect("/invocations-bidirectional-stream"):
                pass

        # Reaching here without an exception propagating out of the app is the
        # assertion; a leaked error would fail the TestClient context exit.
        assert True
