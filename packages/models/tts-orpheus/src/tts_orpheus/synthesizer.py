"""Local Orpheus synthesizer -- wraps handler for direct in-process use."""

import time

from tts_inference.types import (
    AudioEncoding,
    ExecutionMode,
    SynthesisRequest,
    SynthesisResult,
    TTSModelName,
)
from tts_orpheus.handler import OrpheusHandler


class LocalOrpheusSynthesizer:
    """In-process Orpheus synthesizer with lazy model loading."""

    def __init__(self) -> None:
        self._handler = OrpheusHandler()
        self._artifacts: dict | None = None

    @property
    def model_name(self) -> TTSModelName:
        return TTSModelName.ORPHEUS_3B

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.LOCAL

    def _ensure_loaded(self) -> None:
        if self._artifacts is None:
            self._artifacts = self._handler.model_fn("")

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        self._ensure_loaded()
        assert self._artifacts is not None

        start = time.perf_counter()
        wav_bytes, sample_rate, duration = self._handler.predict_fn(
            request.text, self._artifacts, request.voice
        )
        elapsed = time.perf_counter() - start

        return SynthesisResult(
            source_text=request.text,
            model_name=TTSModelName.ORPHEUS_3B,
            audio_bytes=wav_bytes,
            sample_rate=sample_rate,
            duration_seconds=duration,
            elapsed_seconds=elapsed,
            encoding=AudioEncoding.WAV,
            voice_config=request.voice,
        )
