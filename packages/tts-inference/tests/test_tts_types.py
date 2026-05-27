"""Tests for TTS inference types."""

from tts_inference.types import (
    AudioEncoding,
    ExecutionMode,
    SynthesisRequest,
    SynthesisResult,
    TTSModelName,
    VoiceConfig,
)


class TestTTSModelName:
    def test_enum_values(self) -> None:
        assert TTSModelName.KOKORO_82M == "kokoro-82m"
        assert TTSModelName.MAYA_VEENA == "maya-veena"
        assert TTSModelName.CHATTERBOX_TURBO == "chatterbox-turbo"

    def test_from_string(self) -> None:
        assert TTSModelName("kokoro-82m") == TTSModelName.KOKORO_82M
        assert TTSModelName("maya-veena") == TTSModelName.MAYA_VEENA
        assert TTSModelName("chatterbox-turbo") == TTSModelName.CHATTERBOX_TURBO


class TestExecutionMode:
    def test_enum_values(self) -> None:
        assert ExecutionMode.LOCAL == "local"
        assert ExecutionMode.SAGEMAKER == "sagemaker"
        assert ExecutionMode.AUTO == "auto"


class TestAudioEncoding:
    def test_all_encodings(self) -> None:
        assert len(AudioEncoding) == 4
        assert AudioEncoding.WAV == "wav"
        assert AudioEncoding.RAW_PCM == "raw_pcm"


class TestVoiceConfig:
    def test_defaults(self) -> None:
        config = VoiceConfig()
        assert config.voice_id is None
        assert config.language == "en"
        assert config.speed == 1.0
        assert config.reference_audio_path is None

    def test_custom_config(self) -> None:
        config = VoiceConfig(
            voice_id="af_heart",
            language="en",
            speed=1.2,
            reference_audio_path="/tmp/ref.wav",
        )
        assert config.voice_id == "af_heart"
        assert config.speed == 1.2
        assert config.reference_audio_path == "/tmp/ref.wav"


class TestSynthesisRequest:
    def test_minimal_request(self) -> None:
        req = SynthesisRequest(text="Hello world")
        assert req.text == "Hello world"
        assert req.voice.language == "en"
        assert req.encoding == AudioEncoding.WAV
        assert req.sample_rate == 24000

    def test_with_voice(self) -> None:
        req = SynthesisRequest(
            text="Hello",
            voice=VoiceConfig(voice_id="am_michael", speed=0.8),
        )
        assert req.voice.voice_id == "am_michael"
        assert req.voice.speed == 0.8

    def test_json_serialization(self) -> None:
        req = SynthesisRequest(text="Test", voice=VoiceConfig(voice_id="test"))
        json_str = req.model_dump_json()
        restored = SynthesisRequest.model_validate_json(json_str)
        assert restored.text == "Test"
        assert restored.voice.voice_id == "test"


class TestSynthesisResult:
    def test_construction(self) -> None:
        result = SynthesisResult(
            source_text="hello",
            model_name=TTSModelName.KOKORO_82M,
            audio_bytes=b"\x00" * 48000,
            sample_rate=24000,
            duration_seconds=1.0,
            elapsed_seconds=0.5,
        )
        assert result.source_text == "hello"
        assert result.model_name == TTSModelName.KOKORO_82M
        assert result.realtime_factor == 0.5
        assert result.audio_size_bytes == 48000

    def test_realtime_factor_zero_duration(self) -> None:
        result = SynthesisResult(
            source_text="",
            model_name=TTSModelName.KOKORO_82M,
            audio_bytes=b"",
            sample_rate=24000,
            duration_seconds=0.0,
            elapsed_seconds=0.1,
        )
        assert result.realtime_factor == float("inf")

    def test_json_serialization(self) -> None:
        result = SynthesisResult(
            source_text="test",
            model_name=TTSModelName.MAYA_VEENA,
            audio_bytes=b"\x00\x01\x02",
            sample_rate=24000,
            duration_seconds=0.5,
            elapsed_seconds=0.2,
        )
        json_str = result.model_dump_json()
        restored = SynthesisResult.model_validate_json(json_str)
        assert restored.source_text == "test"
        assert restored.model_name == TTSModelName.MAYA_VEENA
        assert restored.audio_bytes == b"\x00\x01\x02"
