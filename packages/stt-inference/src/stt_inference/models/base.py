"""Base protocol and factory for STT extractors.

Usage:
    extractor = get_transcriber(STTModelName.WHISPER_LARGE_V3, mode=ExecutionMode.LOCAL)
    result = extractor.transcribe("audio.wav")
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from stt_inference.types import ExecutionMode, STTModelName, TranscriptionResult

MODEL_IDS: dict[STTModelName, str] = {
    STTModelName.WHISPER_LARGE_V3: "openai/whisper-large-v3",
    STTModelName.QWEN3_ASR: "Qwen/Qwen3-ASR-1.7B",
}

SUBPROCESS_PACKAGES: dict[STTModelName, tuple[str, str]] = {
    STTModelName.WHISPER_LARGE_V3: ("stt-whisper", "stt_whisper"),
    STTModelName.QWEN3_ASR: ("stt-qwen3-asr", "stt_qwen3_asr"),
}


class STTExtractor(Protocol):
    """Protocol for STT extraction -- works with local or remote models."""

    @property
    def model_name(self) -> STTModelName:
        ...

    @property
    def mode(self) -> ExecutionMode:
        ...

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        ...


class SubprocessSTTExtractor:
    """STT extractor that invokes a model package in its own isolated venv."""

    def __init__(self, model: STTModelName) -> None:
        if model not in SUBPROCESS_PACKAGES:
            raise ValueError(f"No subprocess package configured for {model}")
        pkg_dir, module_name = SUBPROCESS_PACKAGES[model]
        self._model = model
        self._module_name = module_name

        project_root = Path(__file__).resolve().parents[5]
        self._project_path = str(project_root / "packages" / "models" / pkg_dir)

    @property
    def model_name(self) -> STTModelName:
        return self._model

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.LOCAL

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        cmd = [
            "uv",
            "run",
            "--project",
            self._project_path,
            "python",
            "-m",
            self._module_name,
            audio_path,
        ]
        logger.info("Invoking {} via subprocess", self._model.value)
        start = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error("stderr: {}", result.stderr)
            raise RuntimeError(
                f"{self._model.value} subprocess failed (exit {result.returncode}): "
                f"{result.stderr[:500]}"
            )

        elapsed = time.perf_counter() - start
        logger.info("{} subprocess completed in {:.1f}s", self._model.value, elapsed)
        return TranscriptionResult.model_validate(json.loads(result.stdout))


def get_model_id(model: STTModelName) -> str:
    return MODEL_IDS[model]


def get_transcriber(
    model: STTModelName | str,
    mode: ExecutionMode | str = ExecutionMode.AUTO,
    **kwargs: Any,
) -> STTExtractor:
    if isinstance(model, str):
        model = STTModelName(model)
    if isinstance(mode, str):
        mode = ExecutionMode(mode)

    if mode == ExecutionMode.AUTO:
        mode = ExecutionMode.LOCAL

    if mode == ExecutionMode.LOCAL:
        return SubprocessSTTExtractor(model)

    if mode == ExecutionMode.SAGEMAKER:
        raise NotImplementedError("SageMaker mode not yet implemented for STT")

    raise ValueError(f"Invalid mode: {mode}")


def list_available_models() -> dict[str, list[str]]:
    return {
        "local": [m.value for m in STTModelName],
        "sagemaker": [],
    }
