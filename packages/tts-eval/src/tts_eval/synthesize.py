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

from enum import StrEnum

from pydantic import TypeAdapter

from tts_inference.types import TTSModelName

ENDPOINT_MAP: dict[str, str] = {
    TTSModelName.KOKORO_82M: "speech-kokoro-82m",
}


class KokoroVoice(StrEnum):
    """The 20 voices that actually work against the deployed Kokoro endpoint.

    The container loads one KPipeline, for lang_code="a" (American English)
    only -- see packages/speech-infra/containers/kokoro/serve.py's module
    docstring. Kokoro's real model supports 54 voices across 9 languages, but
    only these 20 (af_*/am_*) synthesize correctly against this deployment;
    any other voice would either 404 deep in kokoro's own hf_hub_download, or
    silently load a wrong-language style vector into this English-only
    phonemization pipeline with no error at all. Duplicated as a plain
    frozenset in serve.py -- see that file's _VALID_VOICES for why (no
    shared import across the Docker build boundary).
    """

    AF_HEART = "af_heart"
    AF_ALLOY = "af_alloy"
    AF_AOEDE = "af_aoede"
    AF_BELLA = "af_bella"
    AF_JESSICA = "af_jessica"
    AF_KORE = "af_kore"
    AF_NICOLE = "af_nicole"
    AF_NOVA = "af_nova"
    AF_RIVER = "af_river"
    AF_SARAH = "af_sarah"
    AF_SKY = "af_sky"
    AM_ADAM = "am_adam"
    AM_ECHO = "am_echo"
    AM_ERIC = "am_eric"
    AM_FENRIR = "am_fenrir"
    AM_LIAM = "am_liam"
    AM_MICHAEL = "am_michael"
    AM_ONYX = "am_onyx"
    AM_PUCK = "am_puck"
    AM_SANTA = "am_santa"


_KOKORO_VOICE_ADAPTER = TypeAdapter(KokoroVoice)


def validate_kokoro_voice(voice: str) -> str:
    """Validate ``voice`` against :class:`KokoroVoice`, before any network call.

    Raises:
        pydantic.ValidationError: If ``voice`` isn't one of the 20 supported
            voices -- the message already lists them, since Pydantic's own
            enum validation produces that for free.
    """
    return str(_KOKORO_VOICE_ADAPTER.validate_python(voice).value)


POLLY_VOICES: dict[str, dict[str, str]] = {
    TTSModelName.POLLY_STANDARD: {"engine": "standard", "voice_id": "Salli"},
    TTSModelName.POLLY_NEURAL: {"engine": "neural", "voice_id": "Joanna"},
    TTSModelName.POLLY_GENERATIVE: {"engine": "generative", "voice_id": "Ruth"},
}

DEFAULT_VOICES: dict[str, str] = {
    TTSModelName.KOKORO_82M: "af_heart",
    TTSModelName.POLLY_STANDARD: "Salli",
    TTSModelName.POLLY_NEURAL: "Joanna",
    TTSModelName.POLLY_GENERATIVE: "Ruth",
}
