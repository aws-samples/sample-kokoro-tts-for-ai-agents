"""Tests for SynthesisClient utilities."""

from __future__ import annotations

import struct

from tts_eval.synthesize import ENDPOINT_MAP, wav_duration
from tts_inference.types import TTSModelName


def _make_minimal_wav(duration_s: float = 1.0, sr: int = 24000) -> bytes:
    """Create a minimal valid WAV file."""
    channels = 1
    bits_per_sample = 16
    num_samples = int(sr * duration_s)
    data_size = num_samples * channels * (bits_per_sample // 8)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        channels,
        sr,
        sr * channels * (bits_per_sample // 8),
        channels * (bits_per_sample // 8),
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + b"\x00" * data_size


class TestWavDuration:
    def test_correct_duration_calculation(self) -> None:
        wav = _make_minimal_wav(duration_s=2.5, sr=24000)
        d = wav_duration(wav)
        assert abs(d - 2.5) < 0.01

    def test_different_sample_rates(self) -> None:
        for sr in [16000, 22050, 24000, 44100, 48000]:
            wav = _make_minimal_wav(duration_s=1.0, sr=sr)
            d = wav_duration(wav)
            assert abs(d - 1.0) < 0.01

    def test_invalid_data_returns_zero(self) -> None:
        assert wav_duration(b"not a wav file") == 0.0
        assert wav_duration(b"") == 0.0
        assert wav_duration(b"RIFF" + b"\x00" * 10) == 0.0


class TestEndpointMap:
    def test_deployed_models_have_endpoints(self) -> None:
        deployed = [
            TTSModelName.ORPHEUS_3B,
            TTSModelName.KOKORO_82M,
            TTSModelName.CHATTERBOX_TURBO,
        ]
        for model in deployed:
            assert model in ENDPOINT_MAP, f"Missing endpoint for {model.value}"

    def test_endpoint_names_follow_convention(self) -> None:
        for _model, endpoint in ENDPOINT_MAP.items():
            assert endpoint.startswith("speech-"), f"{endpoint} should start with 'speech-'"
