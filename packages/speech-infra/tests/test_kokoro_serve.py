"""Regression tests for the Kokoro GPU container's streaming path.

KPipeline yields torch.Tensor audio segments. A prior bug called the numpy-only
`.astype(np.int16)` directly on those tensors inside the streaming generator,
raising AttributeError *after* the HTTP 200 + chunked headers were already sent.
That truncated the chunked response body, which SageMaker's streaming proxy
surfaces as `ModelStreamError` (and plain `invoke_endpoint` as `ServerError(0)`).

These tests drive the real Starlette app with a fake pipeline that mimics the
model by yielding torch tensors, and assert the streamed WAV body is produced in
full (multiple chunks, valid RIFF header) without raising.

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

_HAS_DEPS = _SERVE_PY is not None and all(
    importlib.util.find_spec(mod) is not None for mod in ("torch", "starlette")
)

pytestmark = pytest.mark.skipif(
    not _HAS_DEPS,
    reason="kokoro container deps (torch, starlette) not installed in this env",
)


@pytest.fixture
def serve_module():
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
            for _ in range(3):
                yield ("g", "p", torch.linspace(-0.1, 0.1, 2400))

    kokoro_mod.KPipeline = _FakePipeline
    sys.modules["kokoro"] = kokoro_mod

    spec = importlib.util.spec_from_file_location("kokoro_serve_under_test", _SERVE_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.pipeline_calls = calls
    try:
        yield module
    finally:
        sys.modules.pop("kokoro", None)
        sys.modules.pop("kokoro_serve_under_test", None)


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
