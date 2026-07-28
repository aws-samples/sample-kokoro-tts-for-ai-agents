"""Regression tests for the Kokoro GPU container's streaming path.

KPipeline yields torch.Tensor audio segments. A prior bug called the numpy-only
`.astype(np.int16)` directly on those tensors inside the streaming generator,
raising AttributeError *after* the HTTP 200 + chunked headers were already sent.
That truncated the chunked response body, which SageMaker's streaming proxy
surfaces as `ModelStreamError` (and plain `invoke_endpoint` as `ServerError(0)`).

These tests drive the real Starlette app with a fake pipeline that mimics the
model by yielding torch tensors, and assert the streamed WAV body is produced in
full (multiple chunks, valid RIFF header) without raising.

They also cover the SSE and MP3 wire formats added for the AgentCore relay,
including the lameenc constraints that path depends on: the flush tail carries
real audio, short utterances produce output only via flush, and an encoder
cannot be reused after flush.

The container deps (torch, starlette, kokoro) are not installed in the lean infra
dev environment, so the module self-skips there. It runs inside the container
image, where those deps exist.
"""

from __future__ import annotations

import base64
import importlib.util
import json
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

_HAS_DEPS = _SERVE_PY is not None and all(
    importlib.util.find_spec(mod) is not None for mod in ("torch", "starlette", "lameenc")
)

pytestmark = pytest.mark.skipif(
    not _HAS_DEPS,
    reason="kokoro container deps (torch, starlette, lameenc) not installed in this env",
)

SEGMENT_SAMPLES = 2400
SEGMENT_COUNT = 3
SAMPLE_RATE = 24000


def _load_serve(segment_samples: int = SEGMENT_SAMPLES, segment_count: int = SEGMENT_COUNT):
    """Import the container's serve.py with a fake tensor-yielding KPipeline."""
    import torch

    kokoro_mod = types.ModuleType("kokoro")

    class _FakePipeline:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __call__(self, text, voice=None, speed=1.0):
            # Real KPipeline yields (graphemes, phonemes, torch.Tensor) per segment.
            for _ in range(segment_count):
                yield ("g", "p", torch.linspace(-0.1, 0.1, segment_samples))

    kokoro_mod.KPipeline = _FakePipeline
    sys.modules["kokoro"] = kokoro_mod

    spec = importlib.util.spec_from_file_location("kokoro_serve_under_test", _SERVE_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event, payload) pairs."""
    events = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        event = None
        data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        assert event is not None, f"SSE block missing event: {block!r}"
        assert data is not None, f"SSE block missing data: {block!r}"
        events.append((event, json.loads(data)))
    return events


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


def test_sse_emits_ordered_audio_events(serve_module) -> None:
    from starlette.testclient import TestClient

    with TestClient(serve_module.app) as client:
        resp = client.post(
            "/invocations",
            json={
                "text": "One. Two. Three.",
                "transport": "sse",
                "format": "mp3",
                "request_id": "req-1",
            },
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["x-accel-buffering"] == "no"

    events = _parse_sse(resp.text)
    names = [name for name, _ in events]
    assert names[0] == "audio_stream_start"
    assert names[-1] == "audio_stream_end"
    assert "error" not in names

    chunks = [payload for name, payload in events if name == "audio_chunk"]
    assert chunks, "expected at least one audio_chunk"
    assert [c["seq"] for c in chunks] == list(range(len(chunks)))
    assert all(c["request_id"] == "req-1" for c in chunks)

    start = events[0][1]
    assert start["format"] == "mp3"
    assert start["sample_rate"] == serve_module.SAMPLE_RATE

    end = events[-1][1]
    assert end["total_chunks"] == len(chunks)
    expected_duration = SEGMENT_COUNT * SEGMENT_SAMPLES / serve_module.SAMPLE_RATE
    assert end["duration_s"] == pytest.approx(expected_duration, abs=0.001)


def test_sse_chunks_reassemble_into_valid_mp3(serve_module) -> None:
    from starlette.testclient import TestClient

    with TestClient(serve_module.app) as client:
        resp = client.post(
            "/invocations",
            json={"text": "One. Two. Three.", "transport": "sse", "format": "mp3"},
        )

    audio = b"".join(
        base64.b64decode(payload["data"])
        for name, payload in _parse_sse(resp.text)
        if name == "audio_chunk"
    )
    assert int.from_bytes(audio[:2], "big") & 0xFFE0 == 0xFFE0
    # The SSE path must carry every byte the binary path would, flush tail included.
    assert len(audio) == _expected_mp3_bytes(SEGMENT_SAMPLES, SEGMENT_COUNT)


def test_sse_wav_format_is_supported(serve_module) -> None:
    from starlette.testclient import TestClient

    with TestClient(serve_module.app) as client:
        resp = client.post(
            "/invocations",
            json={"text": "One. Two. Three.", "transport": "sse", "format": "wav"},
        )

    events = _parse_sse(resp.text)
    assert events[0][1]["format"] == "wav"
    audio = b"".join(
        base64.b64decode(payload["data"]) for name, payload in events if name == "audio_chunk"
    )
    assert audio[:4] == b"RIFF"
    assert len(audio) == 44 + SEGMENT_COUNT * SEGMENT_SAMPLES * 2


def test_short_utterance_still_produces_mp3_via_flush(make_serve_module) -> None:
    """A 50ms segment yields nothing from encode(); the flush tail carries all audio."""
    from starlette.testclient import TestClient

    module = make_serve_module(1200, 1)
    with TestClient(module.app) as client:
        resp = client.post(
            "/invocations",
            json={"text": "Hi.", "transport": "sse", "format": "mp3"},
        )

    events = _parse_sse(resp.text)
    chunks = [p for name, p in events if name == "audio_chunk"]
    assert chunks, "short utterance produced no audio at all"
    audio = b"".join(base64.b64decode(c["data"]) for c in chunks)
    assert int.from_bytes(audio[:2], "big") & 0xFFE0 == 0xFFE0
    assert events[-1][0] == "audio_stream_end"
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
        ({"text": "One.", "transport": "grpc"}, "transport"),
        ({"text": ""}, "text"),
    ],
)
def test_invalid_requests_are_rejected(serve_module, payload, expected) -> None:
    from starlette.testclient import TestClient

    with TestClient(serve_module.app) as client:
        resp = client.post("/invocations", json=payload)

    assert resp.status_code == 400
    assert expected in resp.json()["error"]
