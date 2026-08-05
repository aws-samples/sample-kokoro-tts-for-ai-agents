"""Model catalog for TTS evaluation.

Which endpoint and default voice each model resolves to. The actual
synthesis clients live in ``tts_client``: SageMaker-endpoint synthesis
(response-stream, bidirectional) goes through ``tts_client.client.TTSClient``,
and Amazon Polly goes through ``tts_client.polly.PollyClient`` — Polly is a
managed API with no SageMaker endpoint, so it has its own client rather than
being a mode on ``TTSClient``. Callers resolve a model to the
endpoint/voice/voice_id/engine values those clients need from the catalogs
below.
"""

from __future__ import annotations

from tts_inference.types import TTSModelName

ENDPOINT_MAP: dict[str, str] = {
    TTSModelName.ORPHEUS_3B: "speech-orpheus-3b",
    TTSModelName.KOKORO_82M: "speech-kokoro-82m",
    TTSModelName.KOKORO_82M_CPU: "speech-kokoro-82m-cpu",
    TTSModelName.CHATTERBOX_TURBO: "speech-chatterbox-turbo",
}

POLLY_VOICES: dict[str, dict[str, str]] = {
    TTSModelName.POLLY_STANDARD: {"engine": "standard", "voice_id": "Salli"},
    TTSModelName.POLLY_NEURAL: {"engine": "neural", "voice_id": "Joanna"},
    TTSModelName.POLLY_GENERATIVE: {"engine": "generative", "voice_id": "Ruth"},
}

DEFAULT_VOICES: dict[str, str] = {
    TTSModelName.ORPHEUS_3B: "tara",
    TTSModelName.KOKORO_82M: "af_heart",
    TTSModelName.KOKORO_82M_CPU: "af_heart",
    TTSModelName.CHATTERBOX_TURBO: "ENG_US_F_KimW",
    TTSModelName.POLLY_STANDARD: "Salli",
    TTSModelName.POLLY_NEURAL: "Joanna",
    TTSModelName.POLLY_GENERATIVE: "Ruth",
}
