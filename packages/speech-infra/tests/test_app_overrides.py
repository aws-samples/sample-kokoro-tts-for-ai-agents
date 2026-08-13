"""Tests for `app.py`'s CDK context overrides.

The one behaviour here worth testing directly is the per-model `instance_type`
override. It exists because evaluating a candidate instance type should be a flag
rather than an edit-commit-deploy cycle: the benchmark harness has to be re-runnable
per configuration, and instance type is the dimension that moves most often.

Two properties matter, and both are about *containment*:

- **It is scoped to one model.** A global `-c instance_type=...` would re-type every
  stack in the app at once, silently, which is how a measurement run turns into an
  unintended production change.
- **It is revalidated, not copied.** `model_copy(update=...)` skips validators in
  Pydantic v2, so a typo would reach CloudFormation, be accepted, and leave the
  endpoint in `Updating` with no `FailureReason`.

`_apply_instance_type_override` is called directly rather than through `create_app`,
which would build every stack in the app and need a Docker daemon for the container
assets.
"""

from __future__ import annotations

import aws_cdk as cdk
import pytest
from pydantic import ValidationError

from speech_infra.app import _apply_instance_type_override
from speech_infra.config import TTS_MODEL_CONFIGS, ContainerType, ModelEndpointConfig

#: A real type no model is declared on, so an override to it cannot be confused with the
#: config already saying so. Multi-GPU on purpose: that is the case the override is for.
_OVERRIDE = "ml.g6.12xlarge"


def _config(**overrides) -> ModelEndpointConfig:
    defaults = {
        "model_name": "kokoro-82m",
        "hf_model_id": "hexgrad/Kokoro-82M",
        "instance_type": "ml.g5.xlarge",
        "container_type": ContainerType.PYTORCH_CUSTOM,
    }
    defaults.update(overrides)
    return ModelEndpointConfig(**defaults)


def _app(**context) -> cdk.App:
    return cdk.App(context=context)


class TestInstanceTypeOverride:
    def test_no_context_leaves_the_config_alone(self) -> None:
        config = _config()
        assert _apply_instance_type_override(_app(), config) is config

    def test_the_scoped_key_re_types_the_model(self) -> None:
        app = _app(**{"kokoro-82m:instance_type": "ml.g6.12xlarge"})
        assert _apply_instance_type_override(app, _config()).instance_type == "ml.g6.12xlarge"

    def test_another_models_key_does_not_apply(self) -> None:
        # The containment property: overriding another model must not move kokoro.
        app = _app(**{"other-model:instance_type": "ml.g6.12xlarge"})
        assert _apply_instance_type_override(app, _config()).instance_type == "ml.g5.xlarge"

    def test_an_unscoped_key_does_not_apply(self) -> None:
        # A bare `-c instance_type=...` re-typing every stack at once is exactly the
        # accident this key shape prevents, so it must be inert.
        app = _app(instance_type="ml.g6.12xlarge")
        assert _apply_instance_type_override(app, _config()).instance_type == "ml.g5.xlarge"

    def test_a_typo_fails_at_synth_not_at_deploy(self) -> None:
        # Revalidation, which `model_copy(update=...)` would skip. CloudFormation
        # accepts a bad type and then gives no FailureReason to read.
        app = _app(**{"kokoro-82m:instance_type": "g6.xlarge"})
        with pytest.raises(ValidationError, match="must start with 'ml.'"):
            _apply_instance_type_override(app, _config())

    def test_everything_else_is_carried_across(self) -> None:
        # The scaling numbers are the point of the config; losing one to a partial copy
        # would deploy an endpoint that scales differently from the one measured.
        # Read from TTS_MODEL_CONFIGS rather than `_config()` so the assertion covers the
        # fields that are only set there -- and asserts against a type the declared config
        # is not already on, so it cannot pass by the override doing nothing.
        declared = TTS_MODEL_CONFIGS["kokoro-82m"]
        assert declared.instance_type != _OVERRIDE
        app = _app(**{"kokoro-82m:instance_type": _OVERRIDE})

        result = _apply_instance_type_override(app, declared)

        assert result.instance_type == _OVERRIDE
        assert result.model_dump(exclude={"instance_type"}) == declared.model_dump(
            exclude={"instance_type"}
        )

    def test_the_declared_config_is_not_mutated(self) -> None:
        # TTS_MODEL_CONFIGS is module-level state shared with every other importer,
        # including tts-bench's registry-consistency tests. Compared against the value
        # read beforehand rather than a literal, so this keeps testing mutation after the
        # declared type changes -- which it does, that being the point of the override.
        declared_before = TTS_MODEL_CONFIGS["kokoro-82m"].instance_type
        app = _app(**{"kokoro-82m:instance_type": _OVERRIDE})

        _apply_instance_type_override(app, TTS_MODEL_CONFIGS["kokoro-82m"])

        assert TTS_MODEL_CONFIGS["kokoro-82m"].instance_type == declared_before
