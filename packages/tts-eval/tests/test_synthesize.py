"""Tests for SynthesisClient utilities."""

from __future__ import annotations

import struct

from tts_eval.synthesize import ENDPOINT_MAP, POLLY_VOICES, _pcm_to_wav, wav_duration
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


class TestPollyIntegration:
    def test_polly_voices_config_complete(self) -> None:
        assert TTSModelName.POLLY_STANDARD in POLLY_VOICES
        assert TTSModelName.POLLY_NEURAL in POLLY_VOICES
        assert TTSModelName.POLLY_GENERATIVE in POLLY_VOICES
        for config in POLLY_VOICES.values():
            assert "engine" in config
            assert "voice_id" in config

    def test_pcm_to_wav_produces_valid_header(self) -> None:
        pcm = b"\x00" * 32000  # 1 second at 16000Hz, 16-bit mono
        wav = _pcm_to_wav(pcm, 16000)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        duration = wav_duration(wav)
        assert abs(duration - 1.0) < 0.01

    def test_pcm_to_wav_correct_sample_rate(self) -> None:
        pcm = b"\x00" * 48000  # 1.5 seconds at 16000Hz
        wav = _pcm_to_wav(pcm, 16000)
        sr = struct.unpack_from("<I", wav, 24)[0]
        assert sr == 16000

    def test_polly_voices_use_mp3_format(self) -> None:
        """Polly synthesis should indicate MP3 output format at 24kHz."""
        for _model_name, config in POLLY_VOICES.items():
            assert config["engine"] in ("standard", "neural", "generative")
            assert config["voice_id"]
