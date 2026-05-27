"""Tests for STT inference types."""

from stt_inference.types import (
    AudioFormat,
    AudioInput,
    ExecutionMode,
    STTModelName,
    TranscriptionResult,
    TranscriptionSegment,
    WordSegment,
)


class TestSTTModelName:
    def test_enum_values(self) -> None:
        assert STTModelName.WHISPER_LARGE_V3 == "whisper-large-v3"
        assert STTModelName.QWEN3_ASR == "qwen3-asr"

    def test_from_string(self) -> None:
        assert STTModelName("whisper-large-v3") == STTModelName.WHISPER_LARGE_V3
        assert STTModelName("qwen3-asr") == STTModelName.QWEN3_ASR


class TestExecutionMode:
    def test_enum_values(self) -> None:
        assert ExecutionMode.LOCAL == "local"
        assert ExecutionMode.SAGEMAKER == "sagemaker"
        assert ExecutionMode.AUTO == "auto"


class TestAudioFormat:
    def test_all_formats(self) -> None:
        assert len(AudioFormat) == 5
        assert AudioFormat.WAV == "wav"
        assert AudioFormat.MP3 == "mp3"


class TestWordSegment:
    def test_construction(self) -> None:
        word = WordSegment(word="hello", start=0.0, end=0.5)
        assert word.word == "hello"
        assert word.start == 0.0
        assert word.end == 0.5
        assert word.confidence is None

    def test_with_confidence(self) -> None:
        word = WordSegment(word="world", start=0.5, end=1.0, confidence=0.95)
        assert word.confidence == 0.95


class TestTranscriptionSegment:
    def test_basic_segment(self) -> None:
        segment = TranscriptionSegment(text="hello world", start=0.0, end=2.0)
        assert segment.text == "hello world"
        assert segment.words is None
        assert segment.language is None

    def test_with_words(self) -> None:
        words = [
            WordSegment(word="hello", start=0.0, end=0.5),
            WordSegment(word="world", start=0.5, end=1.0),
        ]
        segment = TranscriptionSegment(text="hello world", start=0.0, end=1.0, words=words)
        assert segment.words is not None
        assert len(segment.words) == 2


class TestTranscriptionResult:
    def test_construction(self) -> None:
        result = TranscriptionResult(
            source="test.wav",
            model_name=STTModelName.WHISPER_LARGE_V3,
            text="hello world",
            segments=[TranscriptionSegment(text="hello world", start=0.0, end=2.0)],
            duration_seconds=2.0,
            elapsed_seconds=0.5,
        )
        assert result.source == "test.wav"
        assert result.model_name == STTModelName.WHISPER_LARGE_V3
        assert result.word_count == 2
        assert not result.has_word_timestamps

    def test_with_word_timestamps(self) -> None:
        words = [WordSegment(word="hello", start=0.0, end=0.5)]
        result = TranscriptionResult(
            source="test.wav",
            model_name=STTModelName.QWEN3_ASR,
            text="hello",
            segments=[TranscriptionSegment(text="hello", start=0.0, end=0.5, words=words)],
            duration_seconds=1.0,
            elapsed_seconds=0.3,
        )
        assert result.has_word_timestamps

    def test_json_serialization(self) -> None:
        result = TranscriptionResult(
            source="test.wav",
            model_name=STTModelName.WHISPER_LARGE_V3,
            text="hello",
            segments=[TranscriptionSegment(text="hello", start=0.0, end=1.0)],
            duration_seconds=1.0,
            elapsed_seconds=0.2,
        )
        json_str = result.model_dump_json()
        restored = TranscriptionResult.model_validate_json(json_str)
        assert restored.text == "hello"
        assert restored.model_name == STTModelName.WHISPER_LARGE_V3


class TestAudioInput:
    def test_with_path(self) -> None:
        audio = AudioInput(path="/tmp/test.wav", format=AudioFormat.WAV)
        assert audio.resolved_path is not None
        assert str(audio.resolved_path) == "/tmp/test.wav"

    def test_without_path(self) -> None:
        audio = AudioInput(audio_bytes=b"fake_audio", sample_rate=16000)
        assert audio.resolved_path is None
