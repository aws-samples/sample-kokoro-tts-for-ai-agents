# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Tests for reading measured references off ``tts-bench`` artifacts.

This module is a seam between two packages that cannot import each other, so the tests
are written against the *file format* rather than against ``QMaxReport``/``ScalingPlan``:
JSON documents with a model name and the few keys each function reads. ``tts-bench``'s own
``test_types.py`` owns the other half — that the real reports actually serialize those
keys — and the two together are what make the seam checkable without a cycle.

Most cases here are ways for the measurement to be missing, because that is the behaviour
that matters: the caller omits the resource rather than inventing a threshold, and synth
has to keep working in a checkout where no benchmark has ever run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from speech_infra import measurements

MODEL = "kokoro-82m"


def _write(directory: Path, name: str, payload: Any) -> Path:
    path = directory / name
    path.write_text(json.dumps(payload))
    return path


def _artifact(
    directory: Path,
    *,
    model_name: str = MODEL,
    c1_ms: Any = 164.5,
    slug: str = "g5xlarge-139b9068",
    filename_model: str | None = None,
) -> Path:
    """A ``qmax`` artifact carrying the two keys this module reads.

    ``filename_model`` defaults to ``model_name``; the two differ only in the tests that
    check which of the filename and the document decides.
    """
    stem = filename_model or model_name
    return _write(
        directory,
        f"qmax-{stem}-bidi-{slug}.json",
        {"model_name": model_name, "ttfab_p95_at_c1_ms": c1_ms},
    )


class TestTtfabP95AtC1Ms:
    def test_reads_the_field(self, tmp_path: Path) -> None:
        _artifact(tmp_path, c1_ms=164.5)
        assert measurements.ttfab_p95_at_c1_ms(MODEL, artifact_dir=tmp_path) == 164.5

    def test_missing_directory_is_not_an_error(self, tmp_path: Path) -> None:
        # The fresh-clone case: no artifacts/ at all. Raising here would mean a
        # repository could not synth until someone had run a 40-minute ladder.
        absent = tmp_path / "never-created"
        assert measurements.ttfab_p95_at_c1_ms(MODEL, artifact_dir=absent) is None

    def test_empty_directory_is_not_an_error(self, tmp_path: Path) -> None:
        assert measurements.ttfab_p95_at_c1_ms(MODEL, artifact_dir=tmp_path) is None

    def test_only_this_models_artifacts_are_read(self, tmp_path: Path) -> None:
        # Service time is a property of a model on an instance type, so another
        # model's number is not a worse threshold — it is a threshold for something
        # that was never deployed here.
        _artifact(tmp_path, model_name="other-model")
        assert measurements.ttfab_p95_at_c1_ms(MODEL, artifact_dir=tmp_path) is None

    def test_a_filename_prefix_match_is_not_a_model_match(self, tmp_path: Path) -> None:
        # A hypothetical kokoro-82m-v2 config would have a filename that matches a
        # glob for kokoro-82m. Deciding on the filename would threshold this model's
        # alarm with a different model's service time, which is why the document's
        # own model_name is what decides.
        _artifact(tmp_path, model_name="kokoro-82m-v2", c1_ms=900.0, slug="g5xlarge-abcdef01")
        assert measurements.ttfab_p95_at_c1_ms(MODEL, artifact_dir=tmp_path) is None

    def test_a_renamed_file_still_reads_its_own_model(self, tmp_path: Path) -> None:
        # The other direction of the same rule: --output takes any path, so the
        # filename is a convention and the document is the fact. A file named for
        # kokoro-82m that records having measured something else is not usable here.
        _artifact(tmp_path, model_name="other-model", filename_model=MODEL)
        assert measurements.ttfab_p95_at_c1_ms(MODEL, artifact_dir=tmp_path) is None

    def test_a_document_without_a_model_name_is_rejected(self, tmp_path: Path) -> None:
        # model_name is required on every report the tool writes, so its absence means
        # this is not one — some other JSON that happened to match the glob.
        _write(tmp_path, f"qmax-{MODEL}-bidi-g5xlarge-139b9068.json", {"ttfab_p95_at_c1_ms": 164.5})
        assert measurements.ttfab_p95_at_c1_ms(MODEL, artifact_dir=tmp_path) is None

    def test_newest_by_mtime_wins(self, tmp_path: Path) -> None:
        old = _artifact(tmp_path, c1_ms=500.0, slug="g5xlarge-aaaa1111")
        new = _artifact(tmp_path, c1_ms=164.5, slug="g5xlarge-bbbb2222")
        os.utime(old, (1_000_000, 1_000_000))
        os.utime(new, (2_000_000, 2_000_000))
        assert measurements.ttfab_p95_at_c1_ms(MODEL, artifact_dir=tmp_path) == 164.5

    def test_falls_through_to_an_older_artifact(self, tmp_path: Path) -> None:
        # A truncated newest file must not mask a good older one. The loop continues
        # rather than returning on the first unreadable path, which is the difference
        # between "one killed run cost us the alarm" and "cost us nothing".
        good = _artifact(tmp_path, c1_ms=500.0, slug="g5xlarge-aaaa1111")
        bad = tmp_path / f"qmax-{MODEL}-bidi-g5xlarge-bbbb2222.json"
        bad.write_text("{truncated")
        os.utime(good, (1_000_000, 1_000_000))
        os.utime(bad, (2_000_000, 2_000_000))
        assert measurements.ttfab_p95_at_c1_ms(MODEL, artifact_dir=tmp_path) == 500.0

    def test_falls_through_past_another_models_artifact(self, tmp_path: Path) -> None:
        # Same fall-through, but for the prefix collision: a hypothetical
        # kokoro-82m-v2's artifact being the newest must not hide kokoro-82m's own.
        mine = _artifact(tmp_path, c1_ms=164.5)
        theirs = _artifact(
            tmp_path, model_name="kokoro-82m-v2", c1_ms=900.0, slug="g5xlarge-abcdef01"
        )
        os.utime(mine, (1_000_000, 1_000_000))
        os.utime(theirs, (2_000_000, 2_000_000))
        assert measurements.ttfab_p95_at_c1_ms(MODEL, artifact_dir=tmp_path) == 164.5

    def test_unreadable_json_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / f"qmax-{MODEL}-bidi-g5xlarge-139b9068.json").write_text("{not json")
        assert measurements.ttfab_p95_at_c1_ms(MODEL, artifact_dir=tmp_path) is None

    def test_a_json_scalar_is_not_an_artifact(self, tmp_path: Path) -> None:
        # Valid JSON, wrong shape. `.get` on a list would raise, taking synth with it.
        _write(tmp_path, f"qmax-{MODEL}-bidi-g5xlarge-139b9068.json", [1, 2, 3])
        assert measurements.ttfab_p95_at_c1_ms(MODEL, artifact_dir=tmp_path) is None

    def test_an_older_artifact_without_the_field_returns_none(self, tmp_path: Path) -> None:
        # The state of this repo before the field existed: real artifacts are on disk
        # and parse fine, but none of them measured N=1 as a named scalar.
        _write(
            tmp_path,
            f"qmax-{MODEL}-bidi-g5xlarge-139b9068.json",
            {"model_name": MODEL, "q_max": 41, "slo_ms": 3000},
        )
        assert measurements.ttfab_p95_at_c1_ms(MODEL, artifact_dir=tmp_path) is None

    def test_a_null_field_returns_none(self, tmp_path: Path) -> None:
        # What a ladder with no N=1 rung serializes. Explicitly null, not absent.
        _artifact(tmp_path, c1_ms=None)
        assert measurements.ttfab_p95_at_c1_ms(MODEL, artifact_dir=tmp_path) is None

    def test_zero_is_rejected(self, tmp_path: Path) -> None:
        # A threshold of 0 alarms on every request forever. Nothing real measures a
        # 0ms p95, so this is a corrupt figure and treated as no measurement.
        _artifact(tmp_path, c1_ms=0)
        assert measurements.ttfab_p95_at_c1_ms(MODEL, artifact_dir=tmp_path) is None

    def test_a_string_is_rejected(self, tmp_path: Path) -> None:
        _artifact(tmp_path, c1_ms="164")
        assert measurements.ttfab_p95_at_c1_ms(MODEL, artifact_dir=tmp_path) is None

    def test_true_is_not_a_latency(self, tmp_path: Path) -> None:
        # `isinstance(True, int)` holds in Python, so a bool would otherwise pass the
        # numeric check and threshold the alarm at 1 microsecond.
        _artifact(tmp_path, c1_ms=True)
        assert measurements.ttfab_p95_at_c1_ms(MODEL, artifact_dir=tmp_path) is None

    def test_an_int_is_accepted_as_a_float(self, tmp_path: Path) -> None:
        # json.dumps writes 164.0 as "164.0", but a hand-edited artifact may hold an
        # int, and the return type is float either way.
        _artifact(tmp_path, c1_ms=164)
        value = measurements.ttfab_p95_at_c1_ms(MODEL, artifact_dir=tmp_path)
        assert value == 164.0
        assert isinstance(value, float)


class TestArtifactDir:
    def test_default_points_at_the_speech_infra_artifacts_dir(self) -> None:
        # Two parents up from measurements.py: speech_infra -> src -> packages/speech-infra.
        # Locked down because the walk is positional: moving this module one directory
        # would silently start reading the wrong place, and the failure would look like
        # "no measurement" — which is a legitimate state, so nothing else would complain.
        assert measurements.ARTIFACT_DIR.name == "artifacts"
        assert (measurements.ARTIFACT_DIR.parent / "src" / "speech_infra").is_dir()


def _plan(
    directory: Path,
    *,
    model_name: str = MODEL,
    filename: str = "plan-kokoro-82m.json",
    c_scale_max_in_cw_units: Any = 30.75,
    verdict: str = "ok",
) -> Path:
    """A ``plan`` artifact carrying the shape ``scaling_thresholds_measured`` reads.

    ``scale_report.plan_to_dict`` writes ``{"plan": {...}, "verdict": ..., ...}`` — the
    nested ``plan`` key is what tells this apart from a ``qmax``/``ttotal`` artifact,
    since ``plan --output`` leaves the filename itself free-form (no glob prefix to
    check first, unlike ``qmax-*.json``).
    """
    return _write(
        directory,
        filename,
        {
            "plan": {"model_name": model_name, "c_scale_max_in_cw_units": c_scale_max_in_cw_units},
            "verdict": verdict,
        },
    )


class TestScalingThresholdsMeasured:
    def test_reads_a_measured_feasible_plan(self, tmp_path: Path) -> None:
        _plan(tmp_path)
        assert measurements.scaling_thresholds_measured(MODEL, artifact_dir=tmp_path) is True

    def test_missing_directory_is_not_an_error(self, tmp_path: Path) -> None:
        absent = tmp_path / "never-created"
        assert measurements.scaling_thresholds_measured(MODEL, artifact_dir=absent) is False

    def test_empty_directory_is_not_an_error(self, tmp_path: Path) -> None:
        assert measurements.scaling_thresholds_measured(MODEL, artifact_dir=tmp_path) is False

    def test_a_qmax_artifact_alone_is_not_a_plan(self, tmp_path: Path) -> None:
        # qmax-*.json has no top-level "plan" key -- the shape this function requires --
        # so a ladder having run is not read as a plan having run.
        _write(tmp_path, "qmax-kokoro-82m-bidi-g5xlarge-139b9068.json", {"model_name": MODEL})
        assert measurements.scaling_thresholds_measured(MODEL, artifact_dir=tmp_path) is False

    def test_the_filename_is_not_the_convention_the_document_is(self, tmp_path: Path) -> None:
        # plan --output takes any path. A file named for one model whose document
        # measured another is not usable for the named one.
        _plan(tmp_path, model_name="other-model", filename="plan-kokoro-82m.json")
        assert measurements.scaling_thresholds_measured(MODEL, artifact_dir=tmp_path) is False

    def test_a_renamed_file_still_reads_its_own_model(self, tmp_path: Path) -> None:
        # The other direction: a file not named after kokoro-82m at all, whose document
        # measured it, is still usable -- the document is the fact.
        _plan(tmp_path, model_name=MODEL, filename="whatever-i-called-it.json")
        assert measurements.scaling_thresholds_measured(MODEL, artifact_dir=tmp_path) is True

    def test_an_infeasible_verdict_is_not_measured(self, tmp_path: Path) -> None:
        # A plan that stopped on its own findings must not be read as "measured, deploy
        # it" just because the file parses.
        _plan(tmp_path, verdict="INFEASIBLE")
        assert measurements.scaling_thresholds_measured(MODEL, artifact_dir=tmp_path) is False

    def test_a_warn_verdict_is_still_measured(self, tmp_path: Path) -> None:
        # A warning is not a stop -- only INFEASIBLE gates the resource.
        _plan(tmp_path, verdict="warn (1)")
        assert measurements.scaling_thresholds_measured(MODEL, artifact_dir=tmp_path) is True

    def test_no_cloudwatch_conversion_is_not_measured(self, tmp_path: Path) -> None:
        # A plan run without --cloudwatch carries the client-measured occupancy but no
        # conversion into the units the alarm reads. Deploying that unconverted number
        # is the exact defect this gate exists to close.
        _plan(tmp_path, c_scale_max_in_cw_units=None)
        assert measurements.scaling_thresholds_measured(MODEL, artifact_dir=tmp_path) is False

    def test_another_models_plan_does_not_apply(self, tmp_path: Path) -> None:
        _plan(tmp_path, model_name="other-model", filename="plan-other-model.json")
        assert measurements.scaling_thresholds_measured(MODEL, artifact_dir=tmp_path) is False

    def test_an_unreadable_plan_falls_through(self, tmp_path: Path) -> None:
        (tmp_path / "plan-kokoro-82m.json").write_text("{not json")
        assert measurements.scaling_thresholds_measured(MODEL, artifact_dir=tmp_path) is False

    def test_a_json_scalar_is_not_a_plan(self, tmp_path: Path) -> None:
        _write(tmp_path, "plan-kokoro-82m.json", [1, 2, 3])
        assert measurements.scaling_thresholds_measured(MODEL, artifact_dir=tmp_path) is False

    def test_newest_by_mtime_wins(self, tmp_path: Path) -> None:
        # Same tie-break rule as qmax artifacts: a rerun after a redeploy must not be
        # masked by an older plan still sitting in the directory.
        old = _plan(tmp_path, filename="plan-kokoro-82m-old.json", verdict="INFEASIBLE")
        new = _plan(tmp_path, filename="plan-kokoro-82m-new.json", verdict="ok")
        os.utime(old, (1_000_000, 1_000_000))
        os.utime(new, (2_000_000, 2_000_000))
        assert measurements.scaling_thresholds_measured(MODEL, artifact_dir=tmp_path) is True
