"""Local Whisper extractor -- wraps handler for direct in-process use."""

from stt_inference.types import ExecutionMode, STTModelName, TranscriptionResult
from stt_whisper.handler import MODEL_ID, WhisperHandler


class LocalWhisperExtractor:
    """In-process Whisper Large V3 extractor with lazy model loading."""

    def __init__(self) -> None:
        self._handler = WhisperHandler()
        self._artifacts: dict | None = None

    @property
    def model_name(self) -> STTModelName:
        return STTModelName.WHISPER_LARGE_V3

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.LOCAL

    def _ensure_loaded(self) -> None:
        if self._artifacts is None:
            self._artifacts = self._handler.model_fn(MODEL_ID)

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        self._ensure_loaded()
        assert self._artifacts is not None
        return self._handler.predict_fn(audio_path, self._artifacts)
