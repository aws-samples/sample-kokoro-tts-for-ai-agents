# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Round-trip WER scorer using AWS Transcribe.

Pipeline: reference text -> TTS audio -> AWS Transcribe -> compare with jiwer.
Measures intelligibility: how accurately the synthesized speech can be understood.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path

import boto3
from jiwer import wer as compute_wer
from loguru import logger

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


class WERScorer:
    """Compute Word Error Rate via AWS Transcribe round-trip."""

    def __init__(self, region: str = "us-east-1") -> None:
        self._transcribe = boto3.client("transcribe", region_name=region)
        self._s3 = boto3.client("s3", region_name=region)
        self._region = region

    def score(
        self,
        reference_text: str,
        audio_path: str | Path | None = None,
        audio_bytes: bytes | None = None,
        sample_rate: int = 24000,
        audio_format: str = "wav",
    ) -> dict[str, float | str]:
        """Compute WER between reference text and transcription of audio.

        Args:
            reference_text: Original text that was synthesized.
            audio_path: Path to audio file to transcribe.
            audio_bytes: Raw audio bytes to transcribe (alternative to audio_path).
            sample_rate: Sample rate of the audio.
            audio_format: Audio format ("wav" or "mp3").

        Returns:
            Dict with 'wer' (float 0-1), 'transcript' (str), 'reference' (str).
        """
        if audio_path is not None:
            audio_bytes = Path(audio_path).read_bytes()
            audio_format = Path(audio_path).suffix.lstrip(".")
        if audio_bytes is None:
            raise ValueError("Must provide either audio_path or audio_bytes")

        transcript = self._transcribe_audio(audio_bytes, sample_rate, audio_format)
        reference_normalized = _PUNCT_RE.sub("", reference_text.strip().lower())
        transcript_normalized = _PUNCT_RE.sub("", transcript.strip().lower())

        if not reference_normalized:
            return {"wer": 0.0, "transcript": transcript, "reference": reference_text}

        error_rate = compute_wer(reference_normalized, transcript_normalized)

        return {
            "wer": round(float(error_rate), 4),
            "transcript": transcript,
            "reference": reference_text,
        }

    def _transcribe_audio(
        self, audio_bytes: bytes, sample_rate: int, audio_format: str = "wav"
    ) -> str:
        """Transcribe audio bytes using AWS Transcribe."""
        bucket = self._get_temp_bucket()
        job_name = f"tts-eval-{uuid.uuid4().hex[:12]}"
        s3_key = f"tts-eval-tmp/{job_name}.{audio_format}"

        self._s3.put_object(Bucket=bucket, Key=s3_key, Body=audio_bytes)

        try:
            self._transcribe.start_transcription_job(
                TranscriptionJobName=job_name,
                Media={"MediaFileUri": f"s3://{bucket}/{s3_key}"},
                MediaFormat=audio_format,
                MediaSampleRateHertz=sample_rate,
                LanguageCode="en-US",
            )

            transcript = self._wait_for_job(job_name)
            return transcript
        finally:
            self._s3.delete_object(Bucket=bucket, Key=s3_key)
            try:
                self._transcribe.delete_transcription_job(TranscriptionJobName=job_name)
            except Exception:
                pass

    def _wait_for_job(self, job_name: str) -> str:
        """Poll until transcription job completes."""
        while True:
            resp = self._transcribe.get_transcription_job(TranscriptionJobName=job_name)
            status = resp["TranscriptionJob"]["TranscriptionJobStatus"]

            if status == "COMPLETED":
                uri = resp["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
                import urllib.request

                with urllib.request.urlopen(uri) as response:
                    result = json.loads(response.read().decode())
                transcripts = result["results"]["transcripts"]
                return str(transcripts[0]["transcript"]) if transcripts else ""

            if status == "FAILED":
                reason = resp["TranscriptionJob"].get("FailureReason", "Unknown")
                logger.error("Transcription failed: {}", reason)
                return ""

            time.sleep(2)

    def _get_temp_bucket(self) -> str:
        """Get or determine the S3 bucket for temporary audio files."""
        sts = boto3.client("sts")
        account_id = sts.get_caller_identity()["Account"]
        return f"speech-model-weights-{account_id}-{self._region}"
