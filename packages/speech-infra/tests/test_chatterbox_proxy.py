# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Tests for the Chatterbox Turbo container's startup path.

Voice conditionals are precomputed during lifespan, but CUDA kernel autotune for the
T3 backbone happens on first `generate`. So `/ping` answering 200 proves the model is
loaded and the voices are cached, and still not that the next caller won't pay JIT —
which after a scale-out is the caller arriving into a capacity shortfall. The warm-up
moves that cost before uvicorn accepts connections, and the stage markers are what let
`tts-bench ttotal` see where the resulting startup time went.

The container deps (torch, chatterbox, starlette) are absent from the lean infra dev
env, so this module self-skips there and runs inside the image, where they exist.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _find_proxy_py() -> Path | None:
    """Locate the chatterbox proxy in the repo layout or the container layout."""
    candidates = [
        Path(__file__).resolve().parents[1] / "containers" / "chatterbox" / "streaming_proxy.py",
        Path("/app/streaming_proxy.py"),
    ]
    return next((p for p in candidates if p.is_file()), None)


_PROXY_PY = _find_proxy_py()

_HAS_DEPS = _PROXY_PY is not None and all(
    importlib.util.find_spec(mod) is not None for mod in ("torch", "starlette")
)

pytestmark = pytest.mark.skipif(
    not _HAS_DEPS,
    reason="chatterbox container deps (torch, starlette) not installed in this env",
)

#: Samples the fake model returns per generation, for the warm-up's return value.
_FAKE_SAMPLES = 2400


@pytest.fixture
def proxy_module(tmp_path):
    """Import the container's streaming_proxy.py with a fake ChatterboxTurboTTS.

    `generate` calls are recorded on `module.generate_calls`, which is what lets the
    tests observe that an inference happened during lifespan rather than only on the
    first request. VOICES_DIR points at a tmp dir holding one fake voice file so
    `_precompute_voices` and `_resolve_voice` both have something real to walk.
    """
    import torch

    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "ENG_US_F_KimW.wav").write_bytes(b"fake")

    generate_calls: list[str] = []

    chatterbox = types.ModuleType("chatterbox")
    tts_turbo = types.ModuleType("chatterbox.tts_turbo")

    class _Conditionals:
        pass

    class _FakeTTS:
        def __init__(self) -> None:
            self.conds = _Conditionals()

        @classmethod
        def from_local(cls, model_dir, device="cuda"):
            return cls()

        @classmethod
        def from_pretrained(cls, device="cuda"):
            return cls()

        def prepare_conditionals(self, audio_path, exaggeration=0.5) -> None:
            self.conds = _Conditionals()

        def generate(self, text):
            generate_calls.append(text)
            return torch.linspace(-0.1, 0.1, _FAKE_SAMPLES)

    tts_turbo.ChatterboxTurboTTS = _FakeTTS
    tts_turbo.Conditionals = _Conditionals
    chatterbox.tts_turbo = tts_turbo

    injected = {"chatterbox": chatterbox, "chatterbox.tts_turbo": tts_turbo}
    sys.modules.update(injected)

    # Read at module import time, so it must be set before exec_module.
    import os

    prior = os.environ.get("VOICES_DIR")
    os.environ["VOICES_DIR"] = str(voices)

    spec = importlib.util.spec_from_file_location("chatterbox_proxy_under_test", _PROXY_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.generate_calls = generate_calls
    try:
        yield module
    finally:
        if prior is None:
            os.environ.pop("VOICES_DIR", None)
        else:
            os.environ["VOICES_DIR"] = prior
        for name in injected:
            sys.modules.pop(name, None)
        sys.modules.pop("chatterbox_proxy_under_test", None)


class TestStartupWarmup:
    def test_warmup_returns_a_real_sample_count(self, proxy_module) -> None:
        # Guards the tests below from passing for the wrong reason: the lifespan
        # swallows warm-up failures, so a broken `_warmup` would leave them green.
        assert proxy_module._warmup() == _FAKE_SAMPLES
        assert proxy_module.generate_calls == [proxy_module.WARMUP_TEXT]

    def test_runs_one_generation_during_lifespan(self, proxy_module) -> None:
        from starlette.testclient import TestClient

        with TestClient(proxy_module.app):
            # No request has been made yet, so this came from lifespan.
            assert proxy_module.generate_calls == [proxy_module.WARMUP_TEXT]

    def test_warmup_output_is_discarded_not_served(self, proxy_module) -> None:
        from starlette.testclient import TestClient

        with TestClient(proxy_module.app) as client:
            resp = client.post("/invocations", json={"text": "Hello.", "stream": False})

        assert resp.status_code == 200
        # The warm-up audio must not be prepended to the first real response.
        assert len(resp.content) == 44 + _FAKE_SAMPLES * 2
        assert proxy_module.generate_calls == [proxy_module.WARMUP_TEXT, "Hello."]

    def test_serves_traffic_when_warmup_raises(self, proxy_module, monkeypatch) -> None:
        # A container that cannot warm up can still serve; failing startup over it
        # would turn a latency problem into an outage. `_warmup` rather than
        # `_synthesize`, since patching the latter would break serving too.
        monkeypatch.setattr(
            proxy_module,
            "_warmup",
            lambda: (_ for _ in ()).throw(RuntimeError("CUDA OOM during warm-up")),
        )

        from starlette.testclient import TestClient

        with TestClient(proxy_module.app) as client:
            resp = client.post("/invocations", json={"text": "Hello.", "stream": False})

        assert resp.status_code == 200
        assert resp.content[:4] == b"RIFF"

    def test_emits_the_full_stage_sequence(self, proxy_module, capfd) -> None:
        from starlette.testclient import TestClient

        from shared.stages import Stage, parse_stage_markers

        with TestClient(proxy_module.app):
            pass

        markers = parse_stage_markers(capfd.readouterr().out.splitlines())
        assert [m.name for m in markers] == [
            Stage.FRAMEWORK_INIT,
            Stage.WEIGHTS_READY,
            Stage.WARMUP_DONE,
            Stage.READY,
        ]
        # A stage that appears to finish before its predecessor would make the
        # attribution `ttotal` produces nonsense.
        assert [m.elapsed_s for m in markers] == sorted(m.elapsed_s for m in markers)

    def test_marks_warmup_done_even_when_warmup_failed(
        self, proxy_module, monkeypatch, capfd
    ) -> None:
        # The marker is emitted regardless so `ttotal`'s sequence stays complete and
        # the warm-up cost stays visible even when the warm-up itself did not work.
        from starlette.testclient import TestClient

        from shared.stages import Stage, parse_stage_markers

        monkeypatch.setattr(proxy_module, "_warmup", lambda: (_ for _ in ()).throw(RuntimeError()))

        with TestClient(proxy_module.app):
            pass

        names = [m.name for m in parse_stage_markers(capfd.readouterr().out.splitlines())]
        assert Stage.WARMUP_DONE in names
        assert names[-1] == Stage.READY

    def test_warmup_uses_the_default_voice_from_cache(self, proxy_module) -> None:
        # The warm-up must not be what *populates* the cache — `_precompute_voices`
        # runs first, so a cache miss here would mean the two disagree on voice id.
        from starlette.testclient import TestClient

        with TestClient(proxy_module.app):
            assert proxy_module.DEFAULT_VOICE in proxy_module._voice_cache
