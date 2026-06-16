"""Automated TTS quality metrics."""

from tts_eval.metrics.utmos import UTMOSScorer
from tts_eval.metrics.wer import WERScorer

__all__ = ["UTMOSScorer", "WERScorer"]
