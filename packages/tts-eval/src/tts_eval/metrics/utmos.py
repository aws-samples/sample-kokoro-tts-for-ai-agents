"""Neural MOS scorer for TTS audio quality.

Uses DNSMOS (Deep Noise Suppression MOS) from the speechmos package to predict
Mean Opinion Scores on a 1-5 scale without human listeners. Provides overall,
signal quality, and background noise scores.
"""

from __future__ import annotations

import io
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from loguru import logger
from speechmos import dnsmos

_SAMPLE_RATE = 16000


class UTMOSScorer:
    """Predict MOS scores using DNSMOS neural model.

    Despite the class name (kept for API compatibility with the eval runner),
    this uses Microsoft's DNSMOS which provides overall MOS, signal MOS, and
    background MOS on a 1-5 scale.
    """

    def __init__(self) -> None:
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        # Trigger model load by scoring a silent sample
        silence = np.zeros(int(_SAMPLE_RATE * 10), dtype=np.float32)
        dnsmos.run(silence, _SAMPLE_RATE)
        self._loaded = True
        logger.info("DNSMOS model loaded")

    def score_file(self, audio_path: str | Path) -> float:
        """Score a WAV file. Returns overall MOS estimate (1-5)."""
        self._load()
        wav, _ = librosa.load(str(audio_path), sr=_SAMPLE_RATE, mono=True)
        return self._score_array(wav)

    def score_bytes(self, wav_bytes: bytes, sample_rate: int = 24000) -> float:
        """Score WAV audio from bytes. Returns overall MOS estimate (1-5)."""
        self._load()
        audio, sr = sf.read(io.BytesIO(wav_bytes))
        if sr != _SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=_SAMPLE_RATE)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return self._score_array(audio)

    def score_batch(self, audio_paths: list[str | Path]) -> list[float]:
        """Score multiple audio files. Returns list of overall MOS estimates."""
        return [self.score_file(p) for p in audio_paths]

    def _score_array(self, audio: np.ndarray) -> float:
        """Score a numpy audio array at 16kHz. Returns overall MOS (1-5)."""
        audio = audio.astype(np.float32)
        peak = np.abs(audio).max()
        if peak > 1.0:
            audio = audio / peak
        result = dnsmos.run(audio, _SAMPLE_RATE)
        score = float(result["ovrl_mos"])
        return float(np.clip(score, 1.0, 5.0))
