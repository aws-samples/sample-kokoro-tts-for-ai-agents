"""Tests for tts_client.types."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tts_client.types import AudioFormat, SampleRate, SynthesisRequest, SynthesisResult, Transport


class TestSynthesisRequest:
    def test_defaults(self) -> None:
        req = SynthesisRequest(text="hello", voice="af_heart")
        assert req.speed == 1.0
        assert req.audio_format == AudioFormat.WAV
        assert req.sample_rate is None
        assert req.request_timestamp is None

    def test_accepts_string_audio_format(self) -> None:
        req = SynthesisRequest(text="hi", voice="af_heart", audio_format="mp3")
        assert req.audio_format == AudioFormat.MP3

    def test_accepts_int_sample_rate(self) -> None:
        req = SynthesisRequest(text="hi", voice="af_heart", sample_rate=16000)
        assert req.sample_rate == SampleRate.HZ_16000

    def test_rejects_a_rate_above_native(self) -> None:
        # 24000 (Kokoro's native rate) is the ceiling, not one option among
        # several -- upsampling past it is pure interpolation with no added
        # fidelity, so it's rejected the same as any other unsupported value.
        with pytest.raises(ValidationError):
            SynthesisRequest(text="hi", voice="af_heart", sample_rate=48000)

    def test_rejects_an_unsupported_rate_below_native(self) -> None:
        with pytest.raises(ValidationError):
            SynthesisRequest(text="hi", voice="af_heart", sample_rate=11025)


class TestSynthesisResult:
    def test_optional_ttfab_defaults_to_none(self) -> None:
        result = SynthesisResult(
            audio_bytes=b"abc",
            audio_format=AudioFormat.WAV,
            sample_rate=24000,
            duration_s=1.0,
            latency_ms=100.0,
            chars=3,
        )
        assert result.ttfab_ms is None
        assert result.chunks == 0


class TestTransportEnum:
    def test_values_match_the_wire_names(self) -> None:
        assert {t.value for t in Transport} == {"response-stream", "bidi"}


class TestAudioFormatEnum:
    def test_values_match_the_container_contract(self) -> None:
        # kokoro/serve.py's FORMAT_WAV / FORMAT_MP3 constants.
        assert {f.value for f in AudioFormat} == {"wav", "mp3"}


class TestSampleRateEnum:
    def test_values_match_the_container_contract(self) -> None:
        # kokoro/serve.py's SUPPORTED_SAMPLE_RATES constant.
        assert {r.value for r in SampleRate} == {8000, 16000, 22050, 24000}

    def test_native_rate_is_the_ceiling(self) -> None:
        assert max(SampleRate) == SampleRate.HZ_24000
