# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Automated TTS quality metrics."""

from tts_eval.metrics.utmos import UTMOSScorer
from tts_eval.metrics.wer import WERScorer

__all__ = ["UTMOSScorer", "WERScorer"]
