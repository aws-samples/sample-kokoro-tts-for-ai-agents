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
    """Import the container's serve.py with a fake tensor-yielding KPipeline."""
    import torch

    kokoro_mod = types.ModuleType("kokoro")

    class _FakePipeline:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __call__(self, text, voice=None, speed=1.0):
            # Real KPipeline yields (graphemes, phonemes, torch.Tensor) per segment.
            for _ in range(3):
                yield ("g", "p", torch.linspace(-0.1, 0.1, 2400))

    kokoro_mod.KPipeline = _FakePipeline
    sys.modules["kokoro"] = kokoro_mod

    spec = importlib.util.spec_from_file_location(
        "kokoro_serve_under_test", _SERVE_PY
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
