"""Tests for tts_client.types."""

from __future__ import annotations

from tts_client.types import AudioFormat, SynthesisRequest, SynthesisResult, Transport


class TestSynthesisRequest:
    def test_defaults(self) -> None:
        req = SynthesisRequest(text="hello", voice="af_heart")
        assert req.speed == 1.0
        assert req.audio_format == AudioFormat.WAV
        assert req.request_timestamp is None

    def test_accepts_string_audio_format(self) -> None:
        req = SynthesisRequest(text="hi", voice="af_heart", audio_format="mp3")
        assert req.audio_format == AudioFormat.MP3


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
