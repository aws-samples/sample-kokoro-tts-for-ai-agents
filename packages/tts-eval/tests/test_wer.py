"""Tests for WER scoring punctuation normalization."""

from __future__ import annotations

from jiwer import wer as compute_wer

from tts_eval.metrics.wer import _PUNCT_RE


def _normalized_wer(reference: str, transcript: str) -> float:
    """Compute WER with same normalization as WERScorer.score()."""
    ref = _PUNCT_RE.sub("", reference.strip().lower())
    trans = _PUNCT_RE.sub("", transcript.strip().lower())
    if not ref:
        return 0.0
    return round(float(compute_wer(ref, trans)), 4)


class TestPunctuationNormalization:
    def test_trailing_period_ignored(self) -> None:
        assert _normalized_wer("Right away.", "Right away") == 0.0

    def test_trailing_period_both_sides(self) -> None:
        assert _normalized_wer("Right away.", "Right away.") == 0.0

    def test_comma_difference_ignored(self) -> None:
        assert _normalized_wer("One moment, please.", "One moment please") == 0.0

    def test_apostrophe_contraction_symmetric(self) -> None:
        assert _normalized_wer("Here's the chart.", "Here's the chart") == 0.0

    def test_real_word_error_still_caught(self) -> None:
        wer = _normalized_wer("Searching now.", "Search him now")
        assert wer > 0.0

    def test_garbled_output_still_caught(self) -> None:
        wer = _normalized_wer("On it.", "Ahmet")
        assert wer > 0.0

    def test_word_substitution_still_caught(self) -> None:
        wer = _normalized_wer("Here are the lab results.", "There are the lab results")
        assert wer > 0.0

    def test_perfect_match_returns_zero(self) -> None:
        assert _normalized_wer("Here are the notes.", "Here are the notes.") == 0.0

    def test_empty_reference(self) -> None:
        assert _normalized_wer("", "something") == 0.0

    def test_case_insensitive(self) -> None:
        assert _normalized_wer("Here's Your Schedule.", "here's your schedule") == 0.0
