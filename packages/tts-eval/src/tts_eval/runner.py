# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Evaluation runner - orchestrates TTS quality assessment across models.

For each model x sample: synthesize -> score UTMOS -> score WER -> collect results.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger

from shared.types import TTSSample
from tts_client.client import TTSClient
from tts_client.polly import PollyClient
from tts_client.types import SynthesisRequest
from tts_eval.metrics.utmos import UTMOSScorer
from tts_eval.metrics.wer import WERScorer
from tts_eval.synthesize import DEFAULT_VOICES, ENDPOINT_MAP, POLLY_VOICES, validate_kokoro_voice
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
        rtf: float | None = None,
        ttfab_ms: float | None = None,
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
        self.rtf = rtf
        self.ttfab_ms = ttfab_ms
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
            "rtf": self.rtf,
            "ttfab_ms": self.ttfab_ms,
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
        max_workers: int = 10,
        streaming_mode: str = "response-stream",
    ) -> None:
        self.models = [TTSModelName(m) for m in models]
        self.samples = samples
        self.output_dir = output_dir
        self.skip_wer = skip_wer
        self._max_workers = max_workers
        self._streaming_mode = streaming_mode

        self._client = TTSClient(region=region)
        self._polly_client = PollyClient(region=region)
        self._utmos = UTMOSScorer()
        self._wer = WERScorer(region=region) if not skip_wer else None

    def run(self) -> list[EvalResult]:
        """Run full evaluation. Returns list of results.

        Samples within each model are evaluated in parallel using a thread pool.
        Models are processed sequentially to keep logs readable.
        """
        results: list[EvalResult] = []
        total = len(self.models) * len(self.samples)
        completed = 0
        lock = threading.Lock()

        for model in self.models:
            model_dir = self.output_dir / "samples" / model.value
            model_dir.mkdir(parents=True, exist_ok=True)

            logger.info(
                "Evaluating {} ({} samples, {} workers)",
                model.value,
                len(self.samples),
                self._max_workers,
            )

            with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
                futures = {
                    pool.submit(self._evaluate_single, model, sample, model_dir): sample
                    for sample in self.samples
                }
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)

                    with lock:
                        completed += 1
                        n = completed

                    status = "OK" if not result.error else f"ERR: {result.error}"
                    logger.debug(
                        "[{}/{}] {} / {} -> UTMOS={} WER={} ({})",
                        n,
                        total,
                        model.value,
                        result.sample_id,
                        f"{result.utmos:.2f}" if result.utmos else "N/A",
                        f"{result.wer:.3f}" if result.wer is not None else "N/A",
                        status,
                    )

        return results

    def _evaluate_single(
        self, model: TTSModelName, sample: TTSSample, model_dir: Path
    ) -> EvalResult:
        """Evaluate a single model x sample pair."""
        is_polly = model in (
            TTSModelName.POLLY_STANDARD,
            TTSModelName.POLLY_NEURAL,
            TTSModelName.POLLY_GENERATIVE,
        )
        try:
            if is_polly:
                voice_config = POLLY_VOICES[model]
                polly_result = self._polly_client.synthesize(
                    voice_id=voice_config["voice_id"],
                    engine=voice_config["engine"],
                    text=sample.text,
                )
                audio_bytes = polly_result.audio_bytes
                audio_format = polly_result.audio_format.value
                latency_ms = polly_result.latency_ms
                duration_s = polly_result.duration_s
                ttfab_ms = polly_result.ttfab_ms
                sample_rate = polly_result.sample_rate
            else:
                endpoint = ENDPOINT_MAP[model]
                voice = validate_kokoro_voice(DEFAULT_VOICES[model])
                request = SynthesisRequest(text=sample.text, voice=voice)
                if self._streaming_mode == "bidirectional":
                    result = self._client.synthesize_bidi(endpoint, request)
                else:
                    result = self._client.synthesize(endpoint, request)
                audio_bytes = result.audio_bytes
                audio_format = result.audio_format.value
                latency_ms = result.latency_ms
                duration_s = result.duration_s
                ttfab_ms = result.ttfab_ms
                sample_rate = result.sample_rate
        except Exception as e:
            return EvalResult(
                model=model.value,
                sample_id=sample.id,
                text=sample.text,
                error=f"Synthesis failed: {e}",
            )

        audio_path = model_dir / f"{sample.id}.{audio_format}"
        audio_path.write_bytes(audio_bytes)

        rtf = (latency_ms / 1000) / duration_s if duration_s and duration_s > 0 else None

        utmos_score = None
        try:
            utmos_score = self._utmos.score_bytes(audio_bytes, sample_rate)
        except Exception as e:
            logger.warning("UTMOS scoring failed for {}/{}: {}", model.value, sample.id, e)

        wer_score = None
        transcript = None
        if self._wer and not self.skip_wer:
            try:
                wer_result = self._wer.score(
                    reference_text=sample.text,
                    audio_bytes=audio_bytes,
                    sample_rate=sample_rate,
                    audio_format=audio_format,
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
            latency_ms=latency_ms,
            audio_duration_s=duration_s,
            rtf=rtf,
            ttfab_ms=ttfab_ms,
        )
