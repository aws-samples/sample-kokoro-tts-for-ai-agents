# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Tests for the Kokoro CPU/ONNX container's startup and event-loop behavior.

Two things this container has to get right for `C_max` to mean anything:

1. **Warm-up.** Both the ONNX session and the misaki G2P are lazily initialized, so
   `/ping` answering 200 proves neither. Without a warm-up the first caller after a
   scale-out pays for espeak dictionary loading plus the provider's first-run graph
   optimization — i.e. the request arriving into a capacity shortfall pays most.
2. **Phonemization off the event loop.** G2P shells out to espeak synchronously. Run
   inline it blocks the loop for every concurrent request, `/ping` included, which
   SageMaker reads as an unhealthy instance and which makes any `C_max` measured
   against this container a measurement of the blocked loop rather than the model.

The container deps (kokoro_onnx, misaki, starlette) are absent from the lean infra dev
env, so this module self-skips there and runs inside the image, where they exist.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
import time
import types
from pathlib import Path

import pytest


def _find_serve_py() -> Path | None:
    """Locate the kokoro-cpu serve.py in the repo layout or the container layout."""
    candidates = [
        Path(__file__).resolve().parents[1] / "containers" / "kokoro-cpu" / "serve.py",
        Path("/app/serve.py"),
    ]
    return next((p for p in candidates if p.is_file()), None)


_SERVE_PY = _find_serve_py()

_HAS_DEPS = _SERVE_PY is not None and all(
    importlib.util.find_spec(mod) is not None for mod in ("starlette", "numpy")
)

pytestmark = pytest.mark.skipif(
    not _HAS_DEPS,
    reason="kokoro-cpu container deps (starlette, numpy) not installed in this env",
)


@pytest.fixture
def serve_module():
    """Import the container's serve.py with fake kokoro_onnx and misaki modules.

    Both are imported lazily inside `_get_model`/`_get_g2p`, so installing stand-ins in
    `sys.modules` is enough. The fake G2P records the thread it ran on and sleeps
    briefly — that is what lets the event-loop tests below distinguish an executor call
    from an inline one.
    """
    import numpy as np

    g2p_threads: list[int] = []
    synth_calls: list[str] = []

    kokoro_onnx = types.ModuleType("kokoro_onnx")

    class _FakeKokoro:
        def __init__(self, model_path, voices_path) -> None:
            pass

        def create(self, phonemes, voice, speed=1.0, is_phonemes=False):
            synth_calls.append(phonemes)
            return np.linspace(-0.1, 0.1, 2400, dtype=np.float32), 24000

    kokoro_onnx.Kokoro = _FakeKokoro

    misaki = types.ModuleType("misaki")
    en_mod = types.ModuleType("misaki.en")
    espeak_mod = types.ModuleType("misaki.espeak")

    class _FakeG2P:
        def __init__(self, trf=False, british=False, fallback=None) -> None:
            pass

        def __call__(self, text):
            g2p_threads.append(threading.get_ident())
            time.sleep(_G2P_DELAY_S)
            return f"ph({text})", None

    en_mod.G2P = _FakeG2P
    espeak_mod.EspeakFallback = lambda british=False: object()
    misaki.en = en_mod
    misaki.espeak = espeak_mod

    injected = {
        "kokoro_onnx": kokoro_onnx,
        "misaki": misaki,
        "misaki.en": en_mod,
        "misaki.espeak": espeak_mod,
    }
    sys.modules.update(injected)

    spec = importlib.util.spec_from_file_location("kokoro_cpu_under_test", _SERVE_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.g2p_threads = g2p_threads
    module.synth_calls = synth_calls
    try:
        yield module
    finally:
        for name in injected:
            sys.modules.pop(name, None)
        sys.modules.pop("kokoro_cpu_under_test", None)


#: How long the fake G2P blocks. Long enough that four concurrent calls on the event
#: loop would be unmistakable dead air, short enough to keep the suite fast.
_G2P_DELAY_S = 0.05


def _post(module, text: str):
    """Build a minimal ASGI POST /invocations and await the handler directly.

    Direct rather than via TestClient because these tests need to run other coroutines
    concurrently with the handler, which TestClient's synchronous portal prevents.
    """
    from starlette.requests import Request

    body = f'{{"text": "{text}"}}'.encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(
        {"type": "http", "method": "POST", "headers": [], "path": "/invocations"}, receive
    )
    return module.invocations(request)


class TestStartupWarmup:
    def test_warmup_exercises_both_g2p_and_the_onnx_session(self, serve_module) -> None:
        # Guards the tests below from passing for the wrong reason: the lifespan
        # swallows warm-up failures, so a broken `_warmup` would leave them green.
        assert serve_module._warmup() == 2400
        assert serve_module.synth_calls == [f"ph({serve_module.WARMUP_TEXT})"]

    def test_runs_one_synthesis_during_lifespan(self, serve_module) -> None:
        from starlette.testclient import TestClient

        with TestClient(serve_module.app):
            # No request has been made yet, so this came from lifespan.
            assert serve_module.synth_calls == [f"ph({serve_module.WARMUP_TEXT})"]

    def test_serves_traffic_when_warmup_raises(self, serve_module, monkeypatch) -> None:
        # `_warmup` rather than `_synthesize`: patching the latter would break the
        # serving path too, so a green result would prove nothing about the warm-up.
        monkeypatch.setattr(
            serve_module,
            "_warmup",
            lambda: (_ for _ in ()).throw(RuntimeError("ONNX session init failed")),
        )

        from starlette.testclient import TestClient

        with TestClient(serve_module.app) as client:
            resp = client.post("/invocations", json={"text": "Hello."})

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
        # A stage that appears to finish before its predecessor would make the
        # attribution `ttotal` produces nonsense.
        assert [m.elapsed_s for m in markers] == sorted(m.elapsed_s for m in markers)

    def test_marks_warmup_done_even_when_warmup_failed(
        self, serve_module, monkeypatch, capfd
    ) -> None:
        from starlette.testclient import TestClient

        from shared.stages import Stage, parse_stage_markers

        monkeypatch.setattr(serve_module, "_warmup", lambda: (_ for _ in ()).throw(RuntimeError()))

        with TestClient(serve_module.app):
            pass

        names = [m.name for m in parse_stage_markers(capfd.readouterr().out.splitlines())]
        assert Stage.WARMUP_DONE in names
        assert names[-1] == Stage.READY


class TestPhonemizeOffEventLoop:
    @pytest.fixture(autouse=True)
    def _ready(self, serve_module):
        """Initialize what the lifespan would, without running it."""
        serve_module._get_model()
        serve_module._get_g2p()
        serve_module.g2p_threads.clear()
        serve_module.synth_calls.clear()

    def test_invocations_returns_wav(self, serve_module) -> None:
        async def go():
            return await _post(serve_module, "Hello.")

        resp = asyncio.run(self._with_lock(serve_module, go))
        assert resp.status_code == 200
        assert resp.body[:4] == b"RIFF"
        assert serve_module.synth_calls == ["ph(Hello.)"]

    def test_phonemize_does_not_run_on_the_event_loop_thread(self, serve_module) -> None:
        loop_thread: list[int] = []

        async def go():
            loop_thread.append(threading.get_ident())
            return await _post(serve_module, "Hi there.")

        asyncio.run(self._with_lock(serve_module, go))

        assert serve_module.g2p_threads, "G2P was never called"
        assert serve_module.g2p_threads[0] != loop_thread[0], (
            f"_phonemize ran on the event loop thread ({loop_thread[0]}); it must be in "
            f"an executor or it blocks /ping under concurrency"
        )

    def test_event_loop_stays_responsive_under_concurrent_g2p(self, serve_module) -> None:
        # Measured as the largest gap between ticks, not the tick count: moving G2P off
        # the loop makes the whole batch *finish sooner*, so a total-tick threshold
        # would move the wrong way. What matters is that the loop is never blocked for
        # a whole G2P — that stall is what a concurrent /ping would land in.
        stamps: list[float] = []

        async def ticker():
            while True:
                await asyncio.sleep(0.005)
                stamps.append(time.perf_counter())

        async def go():
            task = asyncio.create_task(ticker())
            await asyncio.sleep(0.02)  # let the ticker settle
            stamps.clear()
            results = await asyncio.gather(*(_post(serve_module, f"S{i}.") for i in range(4)))
            task.cancel()
            return results

        results = asyncio.run(self._with_lock(serve_module, go))

        assert all(r.status_code == 200 for r in results)
        assert len(stamps) >= 2, "ticker never ran; the loop was blocked throughout"

        max_gap = max(b - a for a, b in zip(stamps, stamps[1:], strict=False))
        assert max_gap < _G2P_DELAY_S / 2, (
            f"event loop blocked for {max_gap * 1000:.0f}ms, longer than half a "
            f"{_G2P_DELAY_S * 1000:.0f}ms G2P call. Phonemization is running inline."
        )

    @staticmethod
    async def _with_lock(module, coro_fn):
        """Run `coro_fn` with the module's inference lock bound to the running loop.

        The lock is normally created in the lifespan; these tests drive the handler
        directly, so they must create it inside the same loop the handler awaits on.
        """
        module._inference_lock = asyncio.Lock()
        return await coro_fn()
