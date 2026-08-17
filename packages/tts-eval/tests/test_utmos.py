# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Tests for DNSMOS-based MOS scorer."""

from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from tts_eval.metrics.utmos import UTMOSScorer


def _make_wav_bytes(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Create WAV bytes from numpy array."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _make_mp3_bytes(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Create MP3 bytes from numpy array using ffmpeg."""
    wav_bytes = _make_wav_bytes(audio, sample_rate)
    result = subprocess.run(
        ["ffmpeg", "-i", "pipe:0", "-f", "mp3", "-q:a", "2", "pipe:1"],
        input=wav_bytes,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")
    return result.stdout


class TestUTMOSScorer:
    def test_score_array_returns_float_in_range(self) -> None:
        scorer = UTMOSScorer()
        # 10s of sine wave at 16kHz (DNSMOS needs 9+ seconds)
        t = np.linspace(0, 10, 160000, dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * 300 * t)
        score = scorer._score_array(audio)
        assert isinstance(score, float)
        assert 1.0 <= score <= 5.0

    def test_score_array_normalizes_clipped_audio(self) -> None:
        scorer = UTMOSScorer()
        # Audio with values > 1.0 should be normalized, not crash
        t = np.linspace(0, 10, 160000, dtype=np.float32)
        audio = 2.5 * np.sin(2 * np.pi * 300 * t)
        assert audio.max() > 1.0
        score = scorer._score_array(audio)
        assert 1.0 <= score <= 5.0

    def test_score_bytes_handles_24khz_wav(self) -> None:
        scorer = UTMOSScorer()
        # Generate 10s at 24kHz, scorer should resample to 16kHz
        t = np.linspace(0, 10, 240000, dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * 300 * t)
        wav_bytes = _make_wav_bytes(audio, sample_rate=24000)
        score = scorer.score_bytes(wav_bytes, sample_rate=24000)
        assert 1.0 <= score <= 5.0

    def test_score_bytes_handles_mp3(self) -> None:
        scorer = UTMOSScorer()
        t = np.linspace(0, 10, 240000, dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * 300 * t)
        mp3_bytes = _make_mp3_bytes(audio, sample_rate=24000)
        score = scorer.score_bytes(mp3_bytes, sample_rate=24000)
        assert 1.0 <= score <= 5.0

    def test_score_batch_returns_list(self) -> None:
        scorer = UTMOSScorer()

        t = np.linspace(0, 10, 160000, dtype=np.float32)
        audio = 0.3 * np.sin(2 * np.pi * 300 * t)

        paths = []
        for _ in range(2):
            p = Path(tempfile.mktemp(suffix=".wav"))
            sf.write(str(p), audio, 16000, format="WAV", subtype="PCM_16")
            paths.append(p)

        try:
            scores = scorer.score_batch(paths)
            assert len(scores) == 2
            for s in scores:
                assert 1.0 <= s <= 5.0
        finally:
            for p in paths:
                p.unlink(missing_ok=True)
