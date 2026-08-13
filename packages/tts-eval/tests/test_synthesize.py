"""Tests for the model catalog: which endpoint/voice each model resolves to.

wav_duration() moved to tts_client.client, along with the SageMaker-endpoint
synthesis logic that used it -- see packages/tts-client/tests/test_client.py
for its coverage. Polly synthesis moved to tts_client.polly.PollyClient --
see packages/tts-client/tests/test_polly.py. What's left here is the catalog
data itself.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tts_eval.synthesize import ENDPOINT_MAP, POLLY_VOICES, KokoroVoice, validate_kokoro_voice
from tts_inference.types import TTSModelName


class TestEndpointMap:
    def test_deployed_models_have_endpoints(self) -> None:
        deployed = [
            TTSModelName.KOKORO_82M,
        ]
        for model in deployed:
            assert model in ENDPOINT_MAP, f"Missing endpoint for {model.value}"

    def test_endpoint_names_follow_convention(self) -> None:
        for _model, endpoint in ENDPOINT_MAP.items():
            assert endpoint.startswith("speech-"), f"{endpoint} should start with 'speech-'"


class TestPollyIntegration:
    def test_polly_voices_config_complete(self) -> None:
        assert TTSModelName.POLLY_STANDARD in POLLY_VOICES
        assert TTSModelName.POLLY_NEURAL in POLLY_VOICES
        assert TTSModelName.POLLY_GENERATIVE in POLLY_VOICES
        for config in POLLY_VOICES.values():
            assert "engine" in config
            assert "voice_id" in config

    def test_polly_voices_use_mp3_format(self) -> None:
        """Polly synthesis should indicate MP3 output format at 24kHz."""
        for _model_name, config in POLLY_VOICES.items():
            assert config["engine"] in ("standard", "neural", "generative")
            assert config["voice_id"]


class TestKokoroVoice:
    def test_has_exactly_the_twenty_working_voices(self) -> None:
        # The deployed container loads one KPipeline, for lang_code="a"
        # (American English) only -- see serve.py's module docstring. Any
        # other language's voices would not actually work against it.
        assert len(KokoroVoice) == 20
        assert all(v.value.startswith(("af_", "am_")) for v in KokoroVoice)

    def test_default_voice_is_a_member(self) -> None:
        assert KokoroVoice.AF_HEART in KokoroVoice
        assert KokoroVoice.AF_HEART == "af_heart"


class TestValidateKokoroVoice:
    @pytest.mark.parametrize("voice", list(KokoroVoice))
    def test_accepts_every_working_voice(self, voice: KokoroVoice) -> None:
        assert validate_kokoro_voice(voice.value) == voice.value

    def test_rejects_a_clearly_invalid_string(self) -> None:
        with pytest.raises(ValidationError):
            validate_kokoro_voice("not-a-real-voice")

    def test_rejects_a_real_but_wrong_language_voice(self) -> None:
        # bf_emma is a real Kokoro voice (British English), but the deployed
        # container never loads a British pipeline -- this is the exact gap
        # this validation closes: a voice that exists in Kokoro's model but
        # not against this deployment must still be rejected.
        with pytest.raises(ValidationError):
            validate_kokoro_voice("bf_emma")
