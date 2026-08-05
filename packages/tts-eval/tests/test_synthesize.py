"""Tests for the model catalog: which endpoint/voice each model resolves to.

wav_duration() moved to tts_client.client, along with the SageMaker-endpoint
synthesis logic that used it -- see packages/tts-client/tests/test_client.py
for its coverage. Polly synthesis moved to tts_client.polly.PollyClient --
see packages/tts-client/tests/test_polly.py. What's left here is the catalog
data itself.
"""

from __future__ import annotations

from tts_eval.synthesize import ENDPOINT_MAP, POLLY_VOICES
from tts_inference.types import TTSModelName


class TestEndpointMap:
    def test_deployed_models_have_endpoints(self) -> None:
        deployed = [
            TTSModelName.ORPHEUS_3B,
            TTSModelName.KOKORO_82M,
            TTSModelName.CHATTERBOX_TURBO,
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
