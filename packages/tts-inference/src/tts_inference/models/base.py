"""Base protocol and factory for TTS synthesizers.

Usage:
    synth = get_synthesizer(TTSModelName.KOKORO_82M, mode=ExecutionMode.LOCAL)
    result = synth.synthesize(SynthesisRequest(text="Hello world"))
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

from tts_inference.types import (
    ExecutionMode,
    SynthesisRequest,
    SynthesisResult,
    TTSModelName,
)

MODEL_IDS: dict[TTSModelName, str] = {
    TTSModelName.KOKORO_82M: "hexgrad/Kokoro-82M",
}

SUBPROCESS_PACKAGES: dict[TTSModelName, tuple[str, str]] = {
    TTSModelName.KOKORO_82M: ("tts-kokoro", "tts_kokoro"),
}


class TTSSynthesizer(Protocol):
    """Protocol for TTS synthesis -- works with local or remote models."""

    @property
    def model_name(self) -> TTSModelName:
        ...

    @property
    def mode(self) -> ExecutionMode:
        ...

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        ...


class SubprocessTTSSynthesizer:
    """TTS synthesizer that invokes a model package in its own isolated venv."""

    def __init__(self, model: TTSModelName) -> None:
        if model not in SUBPROCESS_PACKAGES:
            raise ValueError(f"No subprocess package configured for {model}")
        pkg_dir, module_name = SUBPROCESS_PACKAGES[model]
        self._model = model
        self._module_name = module_name

        project_root = Path(__file__).resolve().parents[5]
        self._project_path = str(project_root / "packages" / "models" / pkg_dir)

    @property
    def model_name(self) -> TTSModelName:
        return self._model

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.LOCAL

    def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        request_json = request.model_dump_json()
        cmd = [
            "uv",
            "run",
            "--project",
            self._project_path,
            "python",
            "-m",
            self._module_name,
            "--request",
            request_json,
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
        return SynthesisResult.model_validate(json.loads(result.stdout))


def get_model_id(model: TTSModelName) -> str:
    return MODEL_IDS[model]


def get_synthesizer(
    model: TTSModelName | str,
    mode: ExecutionMode | str = ExecutionMode.AUTO,
    **kwargs: Any,
) -> TTSSynthesizer:
    if isinstance(model, str):
        model = TTSModelName(model)
    if isinstance(mode, str):
        mode = ExecutionMode(mode)

    if mode == ExecutionMode.AUTO:
        mode = ExecutionMode.LOCAL

    if mode == ExecutionMode.LOCAL:
        return SubprocessTTSSynthesizer(model)

    if mode == ExecutionMode.SAGEMAKER:
        raise NotImplementedError("SageMaker mode not yet implemented for TTS")

    raise ValueError(f"Invalid mode: {mode}")


def list_available_models() -> dict[str, list[str]]:
    return {
        "local": [m.value for m in TTSModelName],
        "sagemaker": [],
    }
