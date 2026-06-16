"""Evaluation runner - orchestrates TTS quality assessment across models.

For each model x sample: synthesize -> score UTMOS -> score WER -> collect results.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from shared.types import TTSSample
from tts_eval.metrics.utmos import UTMOSScorer
from tts_eval.metrics.wer import WERScorer
from tts_eval.synthesize import SynthesisClient
from tts_inference.types import TTSModelName


class EvalResult:
    """Result for a single model x sample evaluation."""

    def __init__(
        self,
        model: str,
        sample_id: str,
        text: str,
        utmos: float | None = None,
        wer: float | None = None,
        transcript: str | None = None,
        latency_ms: float | None = None,
        audio_duration_s: float | None = None,
        error: str | None = None,
    ) -> None:
        self.model = model
        self.sample_id = sample_id
        self.text = text
        self.utmos = utmos
        self.wer = wer
        self.transcript = transcript
        self.latency_ms = latency_ms
        self.audio_duration_s = audio_duration_s
        self.error = error

    def to_dict(self) -> dict:
        d = {
            "model": self.model,
            "sample_id": self.sample_id,
            "text": self.text,
            "utmos": self.utmos,
            "wer": self.wer,
            "transcript": self.transcript,
            "latency_ms": self.latency_ms,
            "audio_duration_s": self.audio_duration_s,
        }
        if self.error:
            d["error"] = self.error
        return d


class EvalRunner:
    """Orchestrates evaluation across models and samples."""

    def __init__(
        self,
        models: list[str | TTSModelName],
        samples: list[TTSSample],
        output_dir: Path,
        region: str = "us-east-1",
        skip_wer: bool = False,
    ) -> None:
        self.models = [TTSModelName(m) for m in models]
        self.samples = samples
        self.output_dir = output_dir
        self.skip_wer = skip_wer

        self._client = SynthesisClient(region=region)
        self._utmos = UTMOSScorer()
        self._wer = WERScorer(region=region) if not skip_wer else None

    def run(self) -> list[EvalResult]:
        """Run full evaluation. Returns list of results."""
        results: list[EvalResult] = []
        total = len(self.models) * len(self.samples)
        completed = 0

        for model in self.models:
            model_dir = self.output_dir / "samples" / model.value
            model_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Evaluating {} ({} samples)", model.value, len(self.samples))

            for sample in self.samples:
                completed += 1
                result = self._evaluate_single(model, sample, model_dir)
                results.append(result)

                status = "OK" if not result.error else f"ERR: {result.error}"
                logger.debug(
                    "[{}/{}] {} / {} -> UTMOS={} WER={} ({})",
                    completed,
                    total,
                    model.value,
                    sample.id,
                    f"{result.utmos:.2f}" if result.utmos else "N/A",
                    f"{result.wer:.3f}" if result.wer is not None else "N/A",
                    status,
                )

        return results

    def _evaluate_single(
        self, model: TTSModelName, sample: TTSSample, model_dir: Path
    ) -> EvalResult:
        """Evaluate a single model x sample pair."""
        try:
            synthesis = self._client.synthesize(model, sample.text)
        except Exception as e:
            return EvalResult(
                model=model.value,
                sample_id=sample.id,
                text=sample.text,
                error=f"Synthesis failed: {e}",
            )

        wav_path = model_dir / f"{sample.id}.wav"
        wav_path.write_bytes(synthesis["audio_bytes"])

        utmos_score = None
        try:
            utmos_score = self._utmos.score_bytes(
                synthesis["audio_bytes"], synthesis["sample_rate"]
            )
        except Exception as e:
            logger.warning("UTMOS scoring failed for {}/{}: {}", model.value, sample.id, e)

        wer_score = None
        transcript = None
        if self._wer and not self.skip_wer:
            try:
                wer_result = self._wer.score(
                    reference_text=sample.text,
                    audio_bytes=synthesis["audio_bytes"],
                    sample_rate=synthesis["sample_rate"],
                )
                wer_score = float(wer_result["wer"])
                transcript = str(wer_result["transcript"])
            except Exception as e:
                logger.warning("WER scoring failed for {}/{}: {}", model.value, sample.id, e)

        return EvalResult(
            model=model.value,
            sample_id=sample.id,
            text=sample.text,
            utmos=utmos_score,
            wer=wer_score,
            transcript=transcript,
            latency_ms=synthesis["latency_ms"],
            audio_duration_s=synthesis["duration_s"],
        )
