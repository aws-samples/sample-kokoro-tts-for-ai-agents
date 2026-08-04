"""Read measured references off ``tts-bench`` artifacts at synth time.

Two numbers cross from the benchmark into the deployed infrastructure without going
through ``config.py``.

The ``FirstChunkLatencyP95`` alarm's threshold cannot be the SLO — that alarm watches
*service* time on an instance already serving, where a request has spent none of its
queue allowance, so an SLO-sized threshold fires only once the endpoint is roughly 10x
past keeping up. And it cannot be a second hand-set config field, which is what it used
to be: a 300ms ``ttfab_budget_ms`` sitting beside a 3000ms ``ttfab_slo_ms`` with nothing
tying either to the other or to a measurement. So it comes from the ladder's own ``N=1``
rung, which measures exactly that quantity and re-measures it on every rerun.

Whether the *scaling* thresholds are measured at all cannot be answered from
``ModelEndpointConfig`` alone — ``scaling_target_value`` and ``scale_in_threshold`` are
plain floats there, so a hand-set number and a `plan`-computed one are indistinguishable
by type. That indistinguishability is how ``0.713`` — a client occupancy deployed against
a server statistic, satisfiable by no positive arrival rate — reached this endpoint
without anything refusing to synth it. :func:`scaling_thresholds_measured` answers the
question the type cannot: does a `plan` artifact exist for this model, current enough to
trust. ``speech_infra.stacks.endpoint`` reads it to decide whether to build any scaling
resources at all, mirroring the alarm's own "no measurement, no alarm" rule.

This module is the seam for both. It parses JSON rather than importing
:mod:`tts_bench.types`, deliberately: ``tts-bench`` depends on ``speech-infra`` for
:class:`~speech_infra.config.ModelEndpointConfig`, so importing back the other way would
be a cycle. The fields read here are present in every artifact those tools write.

**A missing artifact is not an error.** A checkout with no ``artifacts/`` still has to
synth — CI, a fresh clone, a stack for a model nobody has measured — so both functions
return ``None``/``False`` and the caller omits the resource. A threshold has to come
from somewhere real; no measurement means no resource, not a guess.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

#: Where ``tts-bench`` writes by default, relative to the repository root. Four parents
#: up from this file: ``speech_infra`` -> ``src`` -> ``speech-infra`` -> ``packages``.
ARTIFACT_DIR = Path(__file__).resolve().parents[4] / "artifacts"

#: Filename prefix ``tts-bench qmax`` uses. The rest of the stem is the configuration
#: fingerprint, which is why a glob is needed rather than a fixed name — see
#: :func:`_artifacts_for`.
ARTIFACT_PREFIX = "qmax"


def _candidate_paths(model_name: str, *, artifact_dir: Path | None = None) -> list[Path]:
    """Filenames that could be a ``qmax`` artifact for this model, newest first.

    ``tts-bench`` names artifacts ``qmax-<model>-<transport>-<config slug>.json``, where
    the slug is the instance type plus the container image digest. That is a deliberate
    part of the workflow — measuring a second configuration must not overwrite the first —
    so a model can legitimately have several, and this orders them by mtime.

    Newest rather than "the one matching the deployed configuration": at synth time the
    image digest is not yet known (CDK builds the asset during this very synth), so
    matching on it is impossible. The alarm is diagnostic and not in the capacity path, so
    a threshold from the previous image is a stale alarm rather than a wrong policy.
    ``tts-bench plan`` is where a fingerprint mismatch is a hard refusal.

    A prefilter only, and deliberately loose: the glob cannot separate ``kokoro-82m`` from
    ``kokoro-82m-cpu``, which are different models on different instance families. See
    :func:`_load`, which reads the model name out of the document and is what actually
    decides.
    """
    directory = artifact_dir or ARTIFACT_DIR
    if not directory.is_dir():
        return []
    return sorted(
        directory.glob(f"{ARTIFACT_PREFIX}-{model_name}-*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _load(path: Path, model_name: str) -> dict[str, Any] | None:
    """Parse one artifact, returning ``None`` unless it measured ``model_name``.

    Synth must not die on a truncated JSON file — a killed benchmark run can leave one,
    and the failure mode should be "no alarm" rather than "no deploy".

    The model check is on the document rather than the filename because the filename
    cannot be trusted to disambiguate: ``qmax-kokoro-82m-cpu-...json`` matches a glob for
    ``kokoro-82m``, and reading it would threshold the GPU model's alarm with a service
    time measured on an ``ml.c5.2xlarge``. ``model_name`` is a required field on every
    report the tool writes, so requiring it here rejects only documents that were never
    one.
    """
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable qmax artifact {}: {}", path, exc)
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get("model_name") != model_name:
        logger.debug(
            "Skipping {}: it measured {!r}, not {!r}",
            path.name,
            raw.get("model_name"),
            model_name,
        )
        return None
    return raw


def ttfab_p95_at_c1_ms(model_name: str, *, artifact_dir: Path | None = None) -> float | None:
    """Measured p95 first-byte time at one outstanding request, milliseconds.

    The ``FirstChunkLatencyP95`` threshold. ``None`` when no artifact exists, when the
    newest one predates this field, or when its ladder had no ``N=1`` rung — all three are
    "not measured", and the caller omits the alarm rather than inventing a number.

    Args:
        model_name: Config's ``model_name``. Matched against both the filename and the
            ``model_name`` recorded inside the artifact; the second is what decides.
        artifact_dir: Override the search directory. For tests; production reads
            :data:`ARTIFACT_DIR`.
    """
    for path in _candidate_paths(model_name, artifact_dir=artifact_dir):
        raw = _load(path, model_name)
        if raw is None:
            continue
        value = raw.get("ttfab_p95_at_c1_ms")
        if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
            logger.info(
                "{}: FirstChunkLatencyP95 threshold {}ms, measured at N=1 in {}",
                model_name,
                round(float(value)),
                path.name,
            )
            return float(value)
        logger.warning(
            "{}: {} has no usable ttfab_p95_at_c1_ms (ladder had no N=1 rung?); "
            "the latency alarm will be omitted",
            model_name,
            path.name,
        )
    return None


def _plan_candidate_paths(*, artifact_dir: Path | None = None) -> list[Path]:
    """Every JSON file that could be a ``plan`` artifact, newest first.

    Unlike ``qmax``/``ttotal``, ``tts-bench plan`` writes nothing unless ``--output`` is
    given, and that path is free-form — there is no filename convention to glob on the
    way :func:`_candidate_paths` does. So this scans every ``*.json`` in the directory and
    lets :func:`_load_plan` decide by document shape, the same "the filename is a
    convention, the document is the fact" rule :func:`_load` already applies to ``qmax``.
    """
    directory = artifact_dir or ARTIFACT_DIR
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def _load_plan(path: Path, model_name: str) -> dict[str, Any] | None:
    """Parse one file, returning its ``plan`` object iff it is a ``plan`` artifact for ``model_name``.

    A ``plan`` artifact is ``{"plan": {...}, "verdict": ..., ...}`` — see
    ``scale_report.plan_to_dict``. That nested shape is what distinguishes it from a
    ``qmax``/``ttotal`` artifact structurally, so this needs no filename prefix: a
    ``qmax-*.json`` file has no top-level ``plan`` key and is silently skipped rather than
    misread as a plan with no thresholds.
    """
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable artifact {}: {}", path, exc)
        return None
    if not isinstance(raw, dict):
        return None
    plan = raw.get("plan")
    if not isinstance(plan, dict):
        return None
    if plan.get("model_name") != model_name:
        return None
    plan["_verdict"] = raw.get("verdict")
    return plan


def scaling_thresholds_measured(model_name: str, *, artifact_dir: Path | None = None) -> bool:
    """Whether a real, feasible, unit-converted scaling plan exists for this model.

    Gates whether ``speech_infra.stacks.endpoint`` builds any scaling resources at all —
    the same "no measurement, no resource" rule :func:`ttfab_p95_at_c1_ms` already applies
    to the latency alarm. ``ModelEndpointConfig.scaling_target_value`` is a plain float, so
    a hand-set number and a ``plan``-computed one are indistinguishable by type; that
    indistinguishability is how ``0.713`` reached this endpoint without anything refusing
    to synth it. Reading the artifact is what closes the gap: a config field can be
    anything, but only a real ``plan`` run produces a document with this shape.

    Three conditions, all required:

    - A ``plan`` artifact exists for ``model_name`` (see :func:`_load_plan` for how a
      plan artifact is told apart from a ``qmax``/``ttotal`` one — there is no filename
      convention to check first).
    - Its verdict is not ``INFEASIBLE``: a plan that stopped on its own findings must not
      be read as "measured, deploy it".
    - ``c_scale_max_in_cw_units`` is not ``None``: the plan carries the CloudWatch-unit
      conversion, not just the client-measured occupancy beside it. A plan run without
      ``--cloudwatch`` has no conversion, and deploying the unconverted occupancy is the
      defect this whole gate exists to close.

    Args:
        model_name: Config's ``model_name``, matched against the plan's own
            ``model_name`` — never against a filename, which ``plan --output`` leaves
            free-form.
        artifact_dir: Override the search directory. For tests; production reads
            :data:`ARTIFACT_DIR`.
    """
    for path in _plan_candidate_paths(artifact_dir=artifact_dir):
        plan = _load_plan(path, model_name)
        if plan is None:
            continue
        if plan.get("_verdict") == "INFEASIBLE":
            logger.warning(
                "{}: {} is INFEASIBLE; no scaling resources will be synthesized",
                model_name,
                path.name,
            )
            return False
        if plan.get("c_scale_max_in_cw_units") is None:
            logger.warning(
                "{}: {} has no CloudWatch-unit conversion (run without --cloudwatch?); "
                "no scaling resources will be synthesized",
                model_name,
                path.name,
            )
            return False
        logger.info("{}: scaling thresholds measured in {}", model_name, path.name)
        return True
    logger.warning(
        "{}: no plan artifact found; no scaling resources will be synthesized. Run "
        "`tts-bench qmax`, `tts-bench ttotal`, then `tts-bench plan --output ...`.",
        model_name,
    )
    return False
