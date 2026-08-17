# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Tests for the ``T_total`` stage decomposition.

Split by testability, the same way ``test_observe.py`` is. The timeline rules —
reconstructing a missing ``container_start``, bounding recovery, choosing the dominant
stage — are pure functions and are tested directly. The fetch paths use botocore
``Stubber``, which validates every response against the real service model, so a
``describe_log_streams`` reply missing a field fails here rather than against a live
endpoint.

``LOG_LINES`` is copied from a real ``speech-kokoro-82m`` stream, banner and all. Its
notable property is what it is *missing*: no ``container_start`` marker, because the
CUDA base image prints its banner before our entrypoint runs. That is the ordinary case,
not the edge case, which is why the reconstruction path is the one that has to work.

Two things this suite is written *against*, because both shipped and both were silent:

- ``t_total_bounded`` read ``recovered.bounded``, which :func:`assemble_timeline` sets on
  every run, so it was a constant ``True`` and every cleanly-recovered run was reported
  as a floor. :class:`TestTotalIsBoundedOnlyWithoutARecoveryEndpoint` pins both directions.
- ``render_text`` referenced a ``policy_lag_bound_s`` attribute that does not exist, so
  the entire success path raised ``AttributeError``. :class:`TestRenderText` renders a
  report with a measurable ``t_total_s``, which is the case that never ran.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from typing import Any

import boto3
import botocore.exceptions
import pytest
from botocore.stub import Stubber
from loguru import logger

from shared.stages import format_stage_marker, parse_stage_markers
from tts_bench import fixture as fixture_mod
from tts_bench import ttotal as ttotal_mod
from tts_bench.fixture import SUSPEND_ALL
from tts_bench.loadgen import LoadEvent
from tts_bench.observe import LogStream
from tts_bench.ttotal import (
    POLICY_LAG_BOUND_S,
    RECOVERY_MIN_SAMPLES,
    RECOVERY_TOLERANCE,
    TRIGGER_FORCE_DESIRED,
    ScaleEvent,
    StageTime,
    TimelineStage,
    TTotalError,
    TTotalReport,
    _explain_no_scale_out,
    assemble_timeline,
    collect_timeline,
    container_timeline,
    measure,
    read_capacity,
    recovery_bound,
    recovery_references,
    render_text,
    restore_desired_count,
    wait_for_scale_out,
)

ENDPOINT = "speech-kokoro-82m"
VARIANT = "primary"
RID = f"endpoint/{ENDPOINT}/variant/{VARIANT}"
STREAM = "primary/i-0abc123def4567890"
T0 = datetime(2026, 7, 30, 11, 0, 0, tzinfo=UTC)

#: A ``Q_max`` ladder shaped like the measured kokoro one, with rungs at 5 and 10 — the
#: pair the halving test needs. Both sides of "recovery" come from here, so a ladder
#: without both rungs is a re-run of ``qmax`` rather than something ``ttotal`` can patch up.
LADDER = {1: 92.0, 5: 379.0, 10: 667.0, 20: 1243.0, 50: 2910.0}

#: What the probe should read at N=10 while one instance serves it, and the level it must
#: drop to once a second instance splits the traffic. Written as the arithmetic rather
#: than as 473.75 so a change to RECOVERY_TOLERANCE moves the fixture with the module.
P95_AT_PROBE_MS = LADDER[10]
RECOVERED_TARGET_MS = LADDER[5] * RECOVERY_TOLERANCE

#: Verbatim from a live stream. The banner lines matter: they are why the parser has to
#: search each line rather than match it, and why `container_start` is absent.
LOG_LINES = [
    "==========",
    "== CUDA ==",
    "==========",
    "CUDA Version 12.4.1",
    "Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES.",
    "INFO:     Started server process [1]",
    "INFO:     Waiting for application startup.",
    "=== STAGE framework_init t=2026-07-30T11:02:03.018Z elapsed_s=0.018 ===",
    "Loading Kokoro pipeline for lang_code=a",
    "=== STAGE weights_ready t=2026-07-30T11:02:08.868Z elapsed_s=5.868 ===",
    "Running warm-up inference",
    "=== STAGE warmup_done t=2026-07-30T11:02:09.924Z elapsed_s=6.924 ===",
    "=== STAGE ready t=2026-07-30T11:02:09.924Z elapsed_s=6.924 ===",
    "INFO:     Application startup complete.",
    "INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)",
]


def _epoch_ms(at: datetime) -> int:
    return int(at.timestamp() * 1000)


def _event(*, first_byte_at: datetime | None, ttfab_ms: float | None) -> LoadEvent:
    ts = first_byte_at.timestamp() if first_byte_at else None
    return LoadEvent(
        run_id="run",
        step_index=0,
        seq=0,
        worker_index=0,
        concurrency=10,
        model="kokoro-82m",
        endpoint=ENDPOINT,
        dispatch_ts=(ts or 0.0) - 0.1,
        first_byte_ts=ts,
        end_ts=(ts + 0.5) if ts else None,
        ttfab_ms=ttfab_ms,
        latency_ms=500.0,
        outcome="ok",
        http_status=200,
        error_class=None,
        error_message=None,
        chars=30,
        audio_bytes=4800,
        audio_duration_s=0.2,
        rtf=0.4,
        in_flight_at_dispatch=10,
        instance_count=1,
    )


def _events(
    start: datetime, count: int, *, ttfab_ms: float, every_s: float = 1.0
) -> list[LoadEvent]:
    return [
        _event(first_byte_at=start + timedelta(seconds=i * every_s), ttfab_ms=ttfab_ms)
        for i in range(count)
    ]


@pytest.fixture
def sagemaker() -> Any:
    client = boto3.client("sagemaker", region_name="us-east-1")
    stub = Stubber(client)
    stub.activate()
    yield client, stub
    stub.deactivate()


@pytest.fixture
def appscaling() -> Any:
    client = boto3.client("application-autoscaling", region_name="us-east-1")
    stub = Stubber(client)
    stub.activate()
    yield client, stub
    stub.deactivate()


@pytest.fixture
def logs() -> Any:
    client = boto3.client("logs", region_name="us-east-1")
    stub = Stubber(client)
    stub.activate()
    yield client, stub
    stub.deactivate()


@pytest.fixture
def logged() -> Any:
    """Captured loguru messages.

    ``caplog`` does not see these — loguru does not propagate to the stdlib ``logging``
    tree, so a ``caplog`` assertion would pass whether or not anything was logged.
    """
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(message.record["message"]), level="WARNING")
    try:
        yield records
    finally:
        logger.remove(sink_id)


def _variant(*, desired: int, current: int, status: str = "InService") -> dict[str, Any]:
    return {
        "EndpointName": ENDPOINT,
        "EndpointArn": f"arn:aws:sagemaker:us-east-1:1234:endpoint/{ENDPOINT}",
        "EndpointConfigName": f"{ENDPOINT}-config",
        "EndpointStatus": status,
        "CreationTime": T0,
        "LastModifiedTime": T0,
        "ProductionVariants": [
            {
                "VariantName": VARIANT,
                "DesiredInstanceCount": desired,
                "CurrentInstanceCount": current,
            }
        ],
    }


class TestContainerTimeline:
    def test_parses_real_log_lines(self) -> None:
        markers = parse_stage_markers(LOG_LINES)
        entries = container_timeline(markers)

        by_stage = {e.stage: e for e in entries}
        assert by_stage[str(TimelineStage.FRAMEWORK_INIT)].at == datetime(
            2026, 7, 30, 11, 2, 3, 18000, tzinfo=UTC
        )
        assert by_stage[str(TimelineStage.WEIGHTS_READY)].at == datetime(
            2026, 7, 30, 11, 2, 8, 868000, tzinfo=UTC
        )
        assert by_stage[str(TimelineStage.READY)].at == datetime(
            2026, 7, 30, 11, 2, 9, 924000, tzinfo=UTC
        )
        # weights_fetched is legitimately absent: kokoro bakes weights into the image,
        # so the marker would always read zero. Its absence must not be an error.
        assert str(TimelineStage.WEIGHTS_FETCHED) not in by_stage

    def test_fixture_matches_the_emitter_format(self) -> None:
        # Ties the hardcoded log lines to the format contract. If an emitter changes,
        # `shared.stages` tests catch it there and this catches the drift here, so a
        # stale fixture cannot make ttotal look healthy against a format nothing emits.
        marker = parse_stage_markers(LOG_LINES)[0]
        rendered = format_stage_marker(marker.name, marker.at, marker.elapsed_s)
        assert rendered == LOG_LINES[7]

    def test_reconstructs_a_missing_container_start_from_elapsed_s(self) -> None:
        # The ordinary case. framework_init is 0.018s in, so container start is 18ms
        # before it — a figure only elapsed_s can supply.
        entries = container_timeline(parse_stage_markers(LOG_LINES))
        start = next(e for e in entries if e.stage == str(TimelineStage.CONTAINER_STARTED))

        assert start.at == datetime(2026, 7, 30, 11, 2, 3, tzinfo=UTC)
        assert start.bounded is True
        assert "elapsed_s" in (start.note or "")

    def test_prefers_a_reported_container_start_over_reconstruction(self) -> None:
        lines = [
            "=== STAGE container_start t=2026-07-30T11:02:00.000Z elapsed_s=0.000 ===",
            *LOG_LINES,
        ]
        entries = container_timeline(parse_stage_markers(lines))
        starts = [e for e in entries if e.stage == str(TimelineStage.CONTAINER_STARTED)]

        assert len(starts) == 1
        assert starts[0].at == datetime(2026, 7, 30, 11, 2, 0, tzinfo=UTC)
        assert starts[0].bounded is False

    def test_flags_a_reconstruction_that_predates_the_stream(self) -> None:
        # A container claiming it started before its stream had any events means early
        # lines aged out. Worth stating rather than correcting: the elapsed_s is still
        # the container's own measurement.
        opened_at = datetime(2026, 7, 30, 11, 2, 5, tzinfo=UTC)
        entries = container_timeline(parse_stage_markers(LOG_LINES), stream_opened_at=opened_at)
        start = next(e for e in entries if e.stage == str(TimelineStage.CONTAINER_STARTED))

        assert start.at is not None and start.at < opened_at
        assert "predates" in (start.note or "")

    def test_keeps_an_unrecognized_stage_name(self) -> None:
        # A container image may be ahead of this module. Dropping the marker would
        # silently lose the only record of a stage someone deliberately added.
        lines = ["=== STAGE tokenizer_ready t=2026-07-30T11:02:04.000Z elapsed_s=1.000 ==="]
        (entry,) = (e for e in container_timeline(parse_stage_markers(lines)) if not e.bounded)

        assert entry.stage == "tokenizer_ready"
        assert "not known" in (entry.note or "")

    def test_no_markers_yields_no_entries(self) -> None:
        # The image predates the markers, or its startup lines aged out. Neither raises.
        assert container_timeline(parse_stage_markers(LOG_LINES[:7])) == []
        assert container_timeline([]) == []


class TestRecoveryReferences:
    """The two reference points the halving test compares against.

    Both are rungs on the ``Q_max`` ladder, measured on the same configuration by the
    same run. Nothing here is derived from the SLO: at 3000 ms an overloaded probe is
    already "inside budget", so the SLO cannot tell an overload from a recovery.
    """

    def test_the_probes_rung_and_its_half_with_tolerance(self) -> None:
        expected_before, recovered_target = recovery_references(LADDER, probe_concurrency=10)

        assert expected_before == pytest.approx(LADDER[10])
        assert recovered_target == pytest.approx(LADDER[5] * RECOVERY_TOLERANCE)

    @pytest.mark.parametrize("probe_concurrency", [1, 0, 7, -2])
    def test_a_concurrency_that_cannot_halve_is_refused(self, probe_concurrency: int) -> None:
        # There is no "half of 7" rung to compare against, and at N=1 there is nothing to
        # split: p95 is already healthy before the scale-out, so recovery would be
        # declared at the first sample after in_service and the stage would measure nothing.
        with pytest.raises(TTotalError, match="must be an even number of at least 2 to halve"):
            recovery_references(LADDER, probe_concurrency=probe_concurrency)

    def test_a_ladder_missing_the_half_rung_is_refused_not_interpolated(self) -> None:
        # Why it refuses instead of falling back to the nearest rung: a probe held at 10
        # compared against the ladder's value at 20 would still print a p95 target that
        # looks measured. The fix is a re-run of `qmax`, so the message says exactly that.
        with pytest.raises(TTotalError) as excinfo:
            recovery_references({1: 92.0, 10: 667.0, 20: 1243.0}, probe_concurrency=10)

        message = str(excinfo.value)
        assert "concurrency [5]" in message
        # The rungs it does have, so the reader can see what to add rather than re-deriving it.
        assert "measured rungs: [1, 10, 20]" in message
        assert "--concurrency including 5,10" in message

    def test_a_ladder_missing_the_probes_own_rung_is_refused_too(self) -> None:
        # Without it there is nothing to check the probe *against*: a probe that never
        # reached its own expected p95 is a bad probe, not a failed recovery, and the two
        # need opposite responses.
        with pytest.raises(TTotalError, match=r"no usable p95 at concurrency \[10\]"):
            recovery_references({1: 92.0, 5: 379.0}, probe_concurrency=10)


class TestRecoveryBound:
    def test_finds_the_first_sustained_window_inside_budget(self) -> None:
        in_service = T0 + timedelta(seconds=10)
        events = [
            *_events(T0, 8, ttfab_ms=900.0),  # the surge, before the new instance
            *_events(in_service, 3, ttfab_ms=900.0),  # still bad just after
            *_events(in_service + timedelta(seconds=5), 10, ttfab_ms=120.0),
        ]

        at, note = recovery_bound(events, in_service_at=in_service, budget_ms=300.0)

        assert at == in_service + timedelta(seconds=5)
        assert "held at or under 300ms" in note

    def test_reports_no_bound_when_p95_never_recovers(self) -> None:
        # The important negative result: T_total is then a floor, and the report has to
        # say so rather than quietly ending at in_service.
        in_service = T0 + timedelta(seconds=10)
        events = _events(in_service, 30, ttfab_ms=1500.0)

        at, note = recovery_bound(events, in_service_at=in_service, budget_ms=300.0)

        assert at is None
        assert "never held under 300ms" in note
        assert "floor" in note

    def test_refuses_to_bound_on_too_few_samples(self) -> None:
        in_service = T0 + timedelta(seconds=10)
        events = _events(in_service, RECOVERY_MIN_SAMPLES - 1, ttfab_ms=50.0)

        at, note = recovery_bound(events, in_service_at=in_service, budget_ms=300.0)

        assert at is None
        assert "need" in note or "are needed" in note

    def test_ignores_completions_from_before_the_scale_event(self) -> None:
        # A fleet that was healthy before the surge must not have its earlier good
        # latency counted as the new instance recovering.
        in_service = T0 + timedelta(seconds=60)
        events = [*_events(T0, 20, ttfab_ms=50.0), *_events(in_service, 20, ttfab_ms=2000.0)]

        at, _ = recovery_bound(events, in_service_at=in_service, budget_ms=300.0)

        assert at is None

    def test_ignores_events_that_never_produced_audio(self) -> None:
        in_service = T0 + timedelta(seconds=10)
        failed = [
            _event(first_byte_at=in_service + timedelta(seconds=i), ttfab_ms=None)
            for i in range(20)
        ]

        at, note = recovery_bound(failed, in_service_at=in_service, budget_ms=300.0)

        assert at is None
        assert "0 completion(s)" in note

    def test_the_budget_has_to_discriminate_overload_from_recovery(self) -> None:
        # Why this threshold is the ladder's *measured* half-concurrency p95 and not the
        # 3s end-to-end SLO the planner promises against. Kokoro's probe at N=10 reaches a
        # p95 TTFAB of 818ms -- already inside 3000ms. At that threshold the very first
        # window passes and recovery is dated to in_service, so the stage measures nothing;
        # at the 300ms budget the overloaded windows fail and the real boundary is found.
        in_service = T0 + timedelta(seconds=10)
        events = [
            *_events(in_service, 60, ttfab_ms=818.0, every_s=2.0),
            *_events(in_service + timedelta(seconds=120), 60, ttfab_ms=165.0, every_s=2.0),
        ]

        inside_the_slo, _ = recovery_bound(events, in_service_at=in_service, budget_ms=3000.0)
        inside_the_budget, _ = recovery_bound(events, in_service_at=in_service, budget_ms=300.0)

        # Dated to in_service itself: the very first window already passed, so every second
        # of the overload was scored as recovered.
        assert inside_the_slo == in_service
        # Against the measured budget the overloaded windows fail, and the boundary lands
        # where the latency actually changed. Asserted as a span rather than an instant
        # because the 60s window straddles the transition, so the first passing window
        # opens one sample before the last bad completion.
        assert inside_the_budget is not None
        assert inside_the_budget - in_service > timedelta(seconds=100)

    def test_a_sparse_tail_does_not_read_as_a_recovery(self) -> None:
        # Two good requests at the very end of a run would otherwise satisfy any budget
        # once every later window is shorter than min_samples.
        in_service = T0
        events = [
            *_events(in_service, 10, ttfab_ms=900.0),
            *_events(in_service + timedelta(seconds=20), 2, ttfab_ms=10.0),
        ]

        at, _ = recovery_bound(events, in_service_at=in_service, budget_ms=300.0, min_samples=5)

        assert at is None


#: The trigger, 30s after the probe started holding N=10. Named because the clock starts
#: here and not at the probe's start, and every span assertion below is read against it.
DESIRED_SET_AT = T0 + timedelta(seconds=30)
IN_SERVICE_AT = T0 + timedelta(seconds=240)
RECOVERED_AT = T0 + timedelta(seconds=330)


def _full_timeline() -> list[StageTime]:
    """A timeline with every stage observed.

    The timestamps are chosen so the two candidate spans differ visibly: 330s from the
    probe's start, 300s from the trigger. ``t_total_s`` is the second one.
    """
    return assemble_timeline(
        load_applied_at=T0,
        desired_set_at=DESIRED_SET_AT,
        stream=LogStream(
            name=STREAM,
            first_event_at=T0 + timedelta(seconds=180),
            last_event_at=T0 + timedelta(seconds=400),
        ),
        container_stages=container_timeline(
            parse_stage_markers(
                [
                    "=== STAGE container_start t=2026-07-30T11:03:05.000Z elapsed_s=0.000 ===",
                    "=== STAGE weights_ready t=2026-07-30T11:03:11.000Z elapsed_s=6.000 ===",
                    "=== STAGE ready t=2026-07-30T11:03:12.000Z elapsed_s=7.000 ===",
                ]
            )
        ),
        in_service_at=IN_SERVICE_AT,
        recovered_at=RECOVERED_AT,
        recovery_note="p95 TTFAB held at or under 474ms across 40 completions",
    )


def _report(**overrides: Any) -> TTotalReport:
    report = TTotalReport(
        model_name="kokoro-82m",
        endpoint=ENDPOINT,
        run_id="run123",
        trigger=TRIGGER_FORCE_DESIRED,
        from_instances=1,
        to_instances=2,
    )
    report.timeline = _full_timeline()
    for key, value in overrides.items():
        setattr(report, key, value)
    return report


def _without(report: TTotalReport, *stages: TimelineStage) -> None:
    """Blank the timestamps of ``stages`` in place, keeping the entries themselves.

    Blanked rather than dropped, because that is what an unobserved stage looks like
    coming out of :func:`assemble_timeline` — it degrades to a gap, never to an absence.
    """
    names = {str(stage) for stage in stages}
    report.timeline = [
        StageTime(e.stage, None, bounded=e.bounded, source=e.source, note=e.note)
        if e.stage in names
        else e
        for e in report.timeline
    ]


class TestTimelineAssembly:
    def test_the_probe_is_established_before_the_trigger(self) -> None:
        # load_applied is on the timeline to record that the recovery comparison had a
        # pre-scale steady state to compare against — not to start the clock.
        entry = next(e for e in _full_timeline() if e.stage == str(TimelineStage.LOAD_APPLIED))
        assert entry.at == T0
        assert "not the start of the clock" in (entry.note or "")

    def test_emits_the_stages_in_the_order_they_normally_occur(self) -> None:
        # The emitted order, before any sorting by observed time: trigger, then the
        # instance's first external sign, then the container, then the fleet, then traffic.
        stages = [e.stage for e in _full_timeline()]
        assert stages[:3] == [
            str(TimelineStage.LOAD_APPLIED),
            str(TimelineStage.DESIRED_SET),
            str(TimelineStage.INSTANCE_LOGGING),
        ]
        assert stages[-2:] == [str(TimelineStage.IN_SERVICE), str(TimelineStage.TRAFFIC_RECOVERED)]

    def test_marks_instance_logging_as_bounded(self) -> None:
        # A container cannot time its own image pull, so the stream opening is the only
        # visible bound on it — and it is a bound, not a measurement.
        entry = next(e for e in _full_timeline() if e.stage == str(TimelineStage.INSTANCE_LOGGING))
        assert entry.bounded is True
        assert "image pull" in (entry.note or "")

    def test_marks_traffic_recovered_as_bounded_even_when_observed(self) -> None:
        # It is inferred from client latency, never from SageMaker attributing a request
        # to an instance. Observed is not the same as measured here.
        entry = next(e for e in _full_timeline() if e.stage == str(TimelineStage.TRAFFIC_RECOVERED))
        assert entry.bounded is True

    def test_a_container_that_never_reported_ready_leaves_a_stated_gap(self) -> None:
        # Every image emits `ready`, unlike the other container stages, so its absence is
        # stated rather than omitted — otherwise a report with no container half at all
        # reads as complete.
        timeline = assemble_timeline(
            load_applied_at=T0,
            desired_set_at=DESIRED_SET_AT,
            stream=None,
            container_stages=[],
            in_service_at=IN_SERVICE_AT,
            recovered_at=None,
            recovery_note="not evaluated",
        )
        entry = next(e for e in timeline if e.stage == str(TimelineStage.READY))
        assert entry.at is None
        assert "no container reported becoming ready" in (entry.note or "")

    def test_orders_by_observed_time_not_by_enum(self) -> None:
        # Container stage order differs by image; imposing this module's order would
        # produce negative durations on the S3-syncing containers.
        stages = [e.stage for e in _report().observed_stages]
        assert stages.index(str(TimelineStage.CONTAINER_STARTED)) < stages.index(
            str(TimelineStage.IN_SERVICE)
        )
        assert stages[0] == str(TimelineStage.LOAD_APPLIED)
        assert stages[-1] == str(TimelineStage.TRAFFIC_RECOVERED)

    def test_durations_are_non_negative_and_consecutive(self) -> None:
        durations = _report().durations
        assert all(d.seconds >= 0 for d in durations)
        for prev, nxt in zip(durations, durations[1:], strict=False):
            assert prev.to_stage == nxt.from_stage


class TestTTotalReport:
    def test_t_total_spans_the_trigger_to_recovery(self) -> None:
        # 300s from desired_set, not the 330s from the probe's start: the warm-up is this
        # module's choice, so folding it in would inflate a number the policy is scaled by.
        assert _report().t_total_s == pytest.approx(300.0)

    def test_the_warmup_before_the_trigger_is_on_the_timeline_but_not_in_the_number(self) -> None:
        # The stage list telescopes across the whole observed span, warm-up included, so it
        # sums to 330s. T_total is 300s because it starts at the trigger: how long the probe
        # was held first is this module's own choice, and billing it to the lag would inflate
        # the figure the policy is sized by.
        report = _report()
        assert sum(d.seconds for d in report.durations) == pytest.approx(330.0)
        assert report.t_total_s == pytest.approx(300.0)
        first = report.durations[0]
        assert (first.from_stage, first.to_stage) == (
            str(TimelineStage.LOAD_APPLIED),
            str(TimelineStage.DESIRED_SET),
        )

    def test_the_planning_figure_adds_the_policy_lag_as_a_separate_term(self) -> None:
        # Production scales out via the policy, which this trigger bypasses. The bound is
        # added for planning and kept in its own field, because one term is measured and
        # the other is arithmetic off the deployed configuration.
        report = _report()
        assert report.t_total_with_policy_bound_s == pytest.approx(300.0 + POLICY_LAG_BOUND_S)

    def test_dominant_stage_is_the_longest(self) -> None:
        dominant = _report().dominant_stage
        assert dominant is not None
        # Trigger -> stream open: provisioning and image pull, the stage that dominates a
        # real kokoro scale-out and the only one worth attacking to shrink T_total.
        assert dominant.from_stage == str(TimelineStage.DESIRED_SET)
        assert dominant.to_stage == str(TimelineStage.INSTANCE_LOGGING)
        assert dominant.seconds == pytest.approx(150.0)

    def test_splits_the_aws_half_from_the_container_half(self) -> None:
        report = _report()
        # desired_set (T0+30s) to container_start (11:03:05 = T0+185s), not to the stream
        # opening: the container's own claim is tighter when it is available.
        assert report.aws_share_s == pytest.approx(155.0)
        # container_start -> ready.
        assert report.container_share_s == pytest.approx(7.0)

    def test_container_half_falls_back_to_the_stream_when_markers_are_missing(self) -> None:
        report = _report()
        report.timeline = [
            e
            for e in report.timeline
            if e.stage
            not in {
                str(TimelineStage.CONTAINER_STARTED),
                str(TimelineStage.WEIGHTS_READY),
                str(TimelineStage.READY),
            }
        ]

        # Stream opening to in_service: looser, but it is what remains observable.
        assert report.container_share_s == pytest.approx(60.0)
        assert report.aws_share_s == pytest.approx(150.0)

    def test_missing_stages_are_listed_not_dropped(self) -> None:
        # A stage with no timestamp stays on the timeline as a gap. Dropping it would make
        # a report whose container half was never read look like one that had none.
        report = _report()
        _without(report, TimelineStage.INSTANCE_LOGGING)
        assert str(TimelineStage.INSTANCE_LOGGING) in report.missing_stages

    def test_provenance_records_the_trigger(self) -> None:
        from tts_bench.types import Origin

        prov = _report().provenance()
        assert prov.origin is Origin.MEASURED
        assert prov.run_id == "run123"
        assert prov.endpoint == ENDPOINT
        assert TRIGGER_FORCE_DESIRED in (prov.note or "")

    def test_provenance_warns_that_the_policy_half_is_only_bounded(self) -> None:
        # The guard against a capacity-only figure being planned with as a full T_total.
        prov = _report().provenance()
        assert "capacity half only" in (prov.note or "")
        assert f"{POLICY_LAG_BOUND_S:.0f}s rather than measured" in (prov.note or "")

    def test_provenance_measures_from_the_trigger(self) -> None:
        prov = _report().provenance()
        assert prov.measured_at == DESIRED_SET_AT.isoformat()

    def test_to_dict_is_json_serializable_with_shares(self) -> None:
        import json

        payload = _report().to_dict()
        assert json.loads(json.dumps(payload))["t_total_s"] == pytest.approx(300.0)
        shares = [d["share"] for d in payload["durations"]]
        assert all(0.0 <= s <= 1.0 for s in shares)


class TestTotalIsBoundedOnlyWithoutARecoveryEndpoint:
    """``t_total_bounded`` means "no recovery endpoint", not "recovery was inferred".

    The defect this pins shipped: the property read ``recovered.bounded``, and
    :func:`assemble_timeline` sets that flag on *every* run — recovery is always an
    inference, since SageMaker never says which instance served a request. So the
    property was a constant ``True``, and the report's FLOOR line, ``provenance()``,
    ``scale_report``'s FLOOR line and the planner all called cleanly-recovered runs a
    floor. The two facts are separate and both are asserted here.
    """

    def test_a_recovered_run_is_not_a_floor(self) -> None:
        report = _report()
        assert report.at(TimelineStage.TRAFFIC_RECOVERED) is not None
        assert report.t_total_bounded is False

    def test_the_recovered_stage_stays_marked_as_inferred(self) -> None:
        # Reintroducing `return recovered.bounded` passes the pair above only if this
        # flag is also dropped — and dropping it would print a bound as a measurement.
        entry = _report().entry(TimelineStage.TRAFFIC_RECOVERED)
        assert entry is not None and entry.bounded is True

    def test_a_run_with_no_recovery_endpoint_is_a_floor(self) -> None:
        report = _report()
        _without(report, TimelineStage.TRAFFIC_RECOVERED)

        assert report.t_total_bounded is True
        # Falls back to in_service rather than reporting nothing at all: 210s from the
        # trigger, strictly earlier than the instance serving, so it under-reports.
        assert report.t_total_s == pytest.approx(210.0)
        assert "floor" in (report.provenance().note or "")


class TestRenderText:
    def test_it_renders_a_measurable_run(self) -> None:
        # The defect this covers: render_text read a `report.policy_lag_bound_s` that does
        # not exist, so every successful run raised AttributeError on the way to stdout.
        # Only the not-measurable path had a test, and it never reached that line.
        text = render_text(_report())

        assert "300.0s from capacity requested to traffic recovered" in text
        # The planning figure, labelled a BOUND rather than a measurement: it is part
        # measured and part arithmetic, and folding the two is how a bound gets quoted.
        assert f"{POLICY_LAG_BOUND_S:.0f}s BOUND" in text
        assert "360.0s including" in text

    def test_it_marks_bounded_stages(self) -> None:
        text = render_text(_report())
        assert "T_total for speech-kokoro-82m" in text
        assert "dominant stage" in text
        assert "~ marks a stage bounded by inference" in text

    def test_a_run_that_never_recovered_is_labelled_a_floor(self) -> None:
        report = _report()
        _without(report, TimelineStage.TRAFFIC_RECOVERED)

        assert "FLOOR" in render_text(report)

    def test_it_survives_an_empty_timeline(self) -> None:
        # A report with nothing observed still has to print. It is the outcome of a
        # scale event whose APIs all came back empty, which is worth seeing.
        report = _report()
        report.timeline = []
        text = render_text(report)
        assert "not measurable" in text


class TestTheReportCarriesItsConfiguration:
    """A lag belongs to a configuration, the same way a ``Q_max`` ladder does.

    ``plan`` consumes the two together, so a fingerprint on only one side would leave
    the pairing check with nothing to compare — and container start, which dominates
    this measurement, is a property of the image the digest pins.
    """

    def test_the_slug_comes_off_the_fingerprint(self) -> None:
        report = _report()
        report.deployed_config = {
            "instance_type": "ml.g6.12xlarge",
            "image_digest": "139b9068c5eb1f03",
            "container_env": {},
        }
        assert report.config_slug == "g612xlarge-139b9068"

    def test_no_fingerprint_cannot_pass_for_a_real_one(self) -> None:
        # A report rebuilt from a historical window has no configuration to read. It
        # must not compare equal to a measured fingerprint, so the slug says so.
        assert _report().config_slug == "unknown-nodigest"

    def test_to_dict_emits_both_the_fingerprint_and_the_slug(self) -> None:
        report = _report()
        report.deployed_config = {
            "instance_type": "ml.g5.xlarge",
            "image_digest": "deadbeefcafe",
            "container_env": {"MAX_REQUEST_AGE_S": "56"},
        }
        payload = report.to_dict()
        # The dict for the machine check, the slug so a human reading the JSON can see
        # what it was measured against without reassembling it.
        assert payload["deployed_config"]["container_env"] == {"MAX_REQUEST_AGE_S": "56"}
        assert payload["config_slug"] == "g5xlarge-deadbeef"


class TestWaitForScaleOut:
    def _clock(self, step_s: float = 10.0) -> Any:
        """A monotonic fake clock that advances only when ``sleep`` is called."""
        state = {"now": T0}

        def now() -> datetime:
            return state["now"]

        def sleep(seconds: float) -> None:
            state["now"] = state["now"] + timedelta(seconds=seconds or step_s)

        return now, sleep

    def test_returns_when_current_count_rises(self, sagemaker: Any) -> None:
        client, stub = sagemaker
        stub.add_response("describe_endpoint", _variant(desired=1, current=1))
        stub.add_response("describe_endpoint", _variant(desired=2, current=1))
        stub.add_response("describe_endpoint", _variant(desired=2, current=2))
        now, sleep = self._clock()

        event = wait_for_scale_out(
            client,
            endpoint=ENDPOINT,
            from_instances=1,
            poll_interval_s=10.0,
            now=now,
            sleep=sleep,
        )

        assert event.occurred is True
        assert event.to_instances == 2
        # Desired changing is recorded separately from the fleet reaching it: the gap
        # between the two *is* provisioning plus container startup.
        assert event.desired_changed_at == T0 + timedelta(seconds=10)
        assert event.in_service_at == T0 + timedelta(seconds=20)

    def test_timeout_is_a_result_not_an_error(self, sagemaker: Any) -> None:
        client, stub = sagemaker
        for _ in range(4):
            stub.add_response("describe_endpoint", _variant(desired=1, current=1))
        now, sleep = self._clock()

        event = wait_for_scale_out(
            client,
            endpoint=ENDPOINT,
            from_instances=1,
            max_wait_s=30.0,
            poll_interval_s=10.0,
            now=now,
            sleep=sleep,
        )

        assert event.occurred is False
        assert event.in_service_at is None

    def test_missing_variant_raises(self, sagemaker: Any) -> None:
        client, stub = sagemaker
        response = _variant(desired=1, current=1)
        response["ProductionVariants"][0]["VariantName"] = "other"
        stub.add_response("describe_endpoint", response)

        with pytest.raises(TTotalError, match="has no variant primary"):
            read_capacity(client, ENDPOINT, VARIANT)


class TestRestoreDesiredCount:
    def test_no_call_when_already_at_the_starting_count(self, sagemaker: Any) -> None:
        client, stub = sagemaker
        stub.add_response("describe_endpoint", _variant(desired=1, current=1))

        restore_desired_count(client, endpoint=ENDPOINT, to_instances=1)

        stub.assert_no_pending_responses()

    def test_puts_capacity_back(self, sagemaker: Any) -> None:
        client, stub = sagemaker
        stub.add_response("describe_endpoint", _variant(desired=2, current=2))
        stub.add_response(
            "update_endpoint_weights_and_capacities",
            {"EndpointArn": f"arn:aws:sagemaker:us-east-1:1234:endpoint/{ENDPOINT}"},
            {
                "EndpointName": ENDPOINT,
                "DesiredWeightsAndCapacities": [
                    {"VariantName": VARIANT, "DesiredInstanceCount": 1}
                ],
            },
        )

        restore_desired_count(client, endpoint=ENDPOINT, to_instances=1)

        stub.assert_no_pending_responses()

    def test_a_failure_is_logged_not_raised(self, sagemaker: Any) -> None:
        # Restoration runs in a `finally`. Raising here would mask whatever actually
        # went wrong with the measurement.
        client, stub = sagemaker
        stub.add_client_error("describe_endpoint", service_error_code="ValidationException")

        restore_desired_count(client, endpoint=ENDPOINT, to_instances=1)

    def test_it_waits_out_an_updating_endpoint(self, sagemaker: Any) -> None:
        # The failure this exists for: a run that times out mid-scale-out leaves the
        # endpoint Updating, and SageMaker answers a capacity change in that state with
        # "Cannot update in-progress endpoint" — so the restore failed at the one moment
        # it mattered and the fleet kept billing at four instances.
        client, stub = sagemaker
        stub.add_response("describe_endpoint", _variant(desired=4, current=1, status="Updating"))
        stub.add_response("describe_endpoint", _variant(desired=4, current=1, status="Updating"))
        stub.add_response("describe_endpoint", _variant(desired=4, current=1))
        stub.add_response(
            "update_endpoint_weights_and_capacities",
            {"EndpointArn": f"arn:aws:sagemaker:us-east-1:1234:endpoint/{ENDPOINT}"},
            {
                "EndpointName": ENDPOINT,
                "DesiredWeightsAndCapacities": [
                    {"VariantName": VARIANT, "DesiredInstanceCount": 1}
                ],
            },
        )
        slept: list[float] = []

        restore_desired_count(
            client, endpoint=ENDPOINT, to_instances=1, sleep=slept.append, poll_interval_s=5.0
        )

        stub.assert_no_pending_responses()
        assert slept == [5.0, 5.0]

    def test_it_gives_up_and_says_what_to_run(self, sagemaker: Any, logged: list[str]) -> None:
        # Waiting forever in a `finally` would hang the CLI. Giving up silently would
        # leave an endpoint at the raised count with nothing to shrink it, since
        # DesiredInstanceCount is not what a scale-in policy reads.
        client, stub = sagemaker
        for _ in range(3):
            stub.add_response(
                "describe_endpoint", _variant(desired=4, current=1, status="Updating")
            )

        restore_desired_count(
            client,
            endpoint=ENDPOINT,
            to_instances=1,
            sleep=lambda _: None,
            max_wait_s=0.0,
        )

        assert any("still billing at the raised count" in line for line in logged)
        assert any("update-endpoint-weights-and-capacities" in line for line in logged)


def _stub_collect(logs: Any, *, log_lines: list[str], new_stream: bool = True) -> Any:
    """Queue one full pass of ``collect_timeline``'s reads, in call order.

    Logs only. The metric, alarm and scaling-activity reads this used to queue were
    attribution for a policy-driven trigger; under a freeze no policy runs, so those
    APIs are silent by design and querying them would invite reading meaning into it.
    """
    logs_client, logs_stub = logs

    streams = [
        {
            "logStreamName": "primary/i-0oldinstance00000",
            "firstEventTimestamp": _epoch_ms(T0 - timedelta(days=2)),
            "lastEventTimestamp": _epoch_ms(T0 + timedelta(seconds=400)),
        }
    ]
    if new_stream:
        streams.insert(
            0,
            {
                "logStreamName": STREAM,
                "firstEventTimestamp": _epoch_ms(T0 + timedelta(seconds=180)),
                "lastEventTimestamp": _epoch_ms(T0 + timedelta(seconds=400)),
            },
        )
    logs_stub.add_response("describe_log_streams", {"logStreams": streams})
    if new_stream:
        logs_stub.add_response(
            "get_log_events",
            {
                "events": [
                    {
                        "timestamp": _epoch_ms(T0 + timedelta(seconds=180 + i)),
                        "message": line,
                        "ingestionTime": _epoch_ms(T0 + timedelta(seconds=181 + i)),
                    }
                    for i, line in enumerate(log_lines)
                ],
                "nextForwardToken": "f/1",
            },
        )
        # CloudWatch returns the same token at the end of a stream rather than omitting
        # it; this second response is what the read loop's guard has to terminate on.
        logs_stub.add_response("get_log_events", {"events": [], "nextForwardToken": "f/1"})
    return logs_client


class TestCollectTimeline:
    """Assembling one observed scale event from the logs, with the probe's own p95.

    The default probe series is the shape a working scale-out produces: 105 completions
    at the ladder's N=10 value while one instance serves, then a drop to its N=5 value
    ten seconds after ``in_service``. That drop is the only evidence a second instance
    took traffic, so it is also what dates ``traffic_recovered``.
    """

    def _event(self) -> ScaleEvent:
        return ScaleEvent(
            from_instances=1,
            to_instances=2,
            desired_changed_at=T0 + timedelta(seconds=40),
            in_service_at=IN_SERVICE_AT,
        )

    def _halving_events(self, halves_at: datetime) -> list[LoadEvent]:
        """A saturated probe that drops to the ladder's half-concurrency p95 at an instant.

        The overloaded samples keep coming at a 5s cadence right up to ``halves_at``. A gap
        there wider than the 60s recovery window would make every candidate window too
        sparse to judge, and the bound would come back empty for want of samples rather
        than for want of a halving — the wrong reason, and indistinguishable on the report.
        """
        held_after = int((halves_at - IN_SERVICE_AT).total_seconds() // 5)
        return [
            *_events(DESIRED_SET_AT, 105, ttfab_ms=LADDER[10], every_s=2.0),
            *_events(IN_SERVICE_AT, held_after, ttfab_ms=LADDER[10], every_s=5.0),
            *_events(halves_at, 40, ttfab_ms=LADDER[5], every_s=6.0),
        ]

    def _call(self, lg: Any, **overrides: Any) -> TTotalReport:
        kwargs: dict[str, Any] = {
            "logs": lg,
            "endpoint": ENDPOINT,
            "variant": VARIANT,
            "model_name": "kokoro-82m",
            "run_id": "run123",
            "trigger": TRIGGER_FORCE_DESIRED,
            "recovered_budget_ms": RECOVERED_TARGET_MS,
            "p95_expected_before_ms": P95_AT_PROBE_MS,
            "probe_concurrency": 10,
            "load_applied_at": T0,
            "desired_set_at": DESIRED_SET_AT,
            "window_start": T0 - timedelta(seconds=5),
            "window_end": T0 + timedelta(seconds=500),
            "streams_before": [
                LogStream(
                    name="primary/i-0oldinstance00000",
                    first_event_at=T0 - timedelta(days=2),
                    last_event_at=T0 + timedelta(seconds=400),
                )
            ],
            "event": self._event(),
            "load_events": self._halving_events(T0 + timedelta(seconds=250)),
        }
        kwargs.update(overrides)
        return collect_timeline(**kwargs)

    def test_assembles_every_stage_it_can_read(self, logs: Any) -> None:
        lg = _stub_collect(logs, log_lines=LOG_LINES)

        report = self._call(lg)

        assert report.at(TimelineStage.LOAD_APPLIED) == T0
        assert report.at(TimelineStage.DESIRED_SET) == DESIRED_SET_AT
        assert report.at(TimelineStage.INSTANCE_LOGGING) == T0 + timedelta(seconds=180)
        assert report.at(TimelineStage.READY) is not None
        assert report.at(TimelineStage.IN_SERVICE) == IN_SERVICE_AT
        assert report.at(TimelineStage.TRAFFIC_RECOVERED) == T0 + timedelta(seconds=250)
        # 250s - 30s: from the trigger, not from the probe's start.
        assert report.t_total_s == pytest.approx(220.0)

    def test_names_the_new_instance_by_log_stream_diff(self, logs: Any) -> None:
        # SageMaker does not report which instance it added; the set difference over
        # stream names is the only way to identify it.
        lg = _stub_collect(logs, log_lines=LOG_LINES)

        report = self._call(lg)

        assert report.instance_id == "i-0abc123def4567890"

    def test_records_the_degradation_being_recovered_from(self, logs: Any) -> None:
        # Without the before figure a recovery has nothing to be a recovery *from*, and
        # the two ladder rungs on the report are what make the halving checkable.
        lg = _stub_collect(logs, log_lines=LOG_LINES)

        report = self._call(lg)

        assert report.p95_before_ms == pytest.approx(LADDER[10])
        assert report.p95_after_ms == pytest.approx(LADDER[5])
        assert report.p95_expected_before_ms == pytest.approx(LADDER[10])
        assert report.p95_recovered_target_ms == pytest.approx(RECOVERED_TARGET_MS)
        assert report.requests_before > 0 and report.requests_after > 0

    def test_a_series_that_halves_at_a_known_instant_recovers_there(self, logs: Any) -> None:
        # The measurement itself, against a series whose transition is placed by hand:
        # recovery must land on the first completion at the halved level, not on the
        # window that merely contains it.
        halves_at = IN_SERVICE_AT + timedelta(seconds=90)
        lg = _stub_collect(logs, log_lines=LOG_LINES)

        report = self._call(lg, load_events=self._halving_events(halves_at))

        assert report.at(TimelineStage.TRAFFIC_RECOVERED) == halves_at
        assert report.t_total_bounded is False
        assert report.t_total_s == pytest.approx((halves_at - DESIRED_SET_AT).total_seconds())

    def test_a_series_that_never_halves_reports_a_floor(self, logs: Any) -> None:
        # Either the added instance never took traffic, or the probe's p95 was not what
        # the ladder measured. Both end the span at in_service, which is strictly earlier
        # than the instance serving — so the number under-reports and has to say so.
        lg = _stub_collect(logs, log_lines=LOG_LINES)

        report = self._call(
            lg, load_events=_events(DESIRED_SET_AT, 160, ttfab_ms=LADDER[10], every_s=2.0)
        )

        assert report.t_total_bounded is True
        assert str(TimelineStage.TRAFFIC_RECOVERED) in report.missing_stages
        assert any("floor" in note for note in report.notes)
        # Still reports a number, ending at in_service: 240s - 30s.
        assert report.t_total_s == pytest.approx(210.0)

    def test_a_probe_that_was_never_saturating_says_to_raise_the_concurrency(
        self, logs: Any
    ) -> None:
        # A probe whose pre-scale p95 is already at the recovered level gives the halving
        # nothing to detect: traffic_recovered lands on the first sample after in_service,
        # which is a tautology rather than a measurement. Asserted here so the run says
        # which knob to move instead of leaving it to be inferred from two numbers.
        lg = _stub_collect(logs, log_lines=LOG_LINES)

        report = self._call(
            lg,
            load_events=[
                *_events(DESIRED_SET_AT, 105, ttfab_ms=LADDER[1], every_s=2.0),
                *_events(IN_SERVICE_AT, 40, ttfab_ms=LADDER[1], every_s=6.0),
            ],
        )

        note = next(n for n in report.notes if "never saturating" in n)
        assert "--probe-concurrency" in note
        assert f"expected {LADDER[10]:.0f}ms at N=10" in note
        # The tautology the note exists to warn about, made visible.
        assert report.at(TimelineStage.TRAFFIC_RECOVERED) == IN_SERVICE_AT

    def test_a_probe_that_never_ran_reports_recovery_missing(self, logs: Any) -> None:
        # A run whose probe died leaves no completions to bound recovery with. Ending
        # silently at in_service would look like a complete measurement, so the stage is
        # a stated gap and the span is labelled a floor.
        lg = _stub_collect(logs, log_lines=LOG_LINES)

        report = self._call(lg, load_events=[], load_applied_at=None)

        assert str(TimelineStage.TRAFFIC_RECOVERED) in report.missing_stages
        assert report.t_total_bounded is True
        entry = report.entry(TimelineStage.LOAD_APPLIED)
        assert entry is not None and "no probe ran" in (entry.note or "")
        recovered = report.entry(TimelineStage.TRAFFIC_RECOVERED)
        assert recovered is not None and "completion(s)" in (recovered.note or "")

    def test_the_policy_half_is_named_as_bounded_on_every_report(self, logs: Any) -> None:
        # The trigger's known omission, stated on the artifact rather than only in the
        # docs: a capacity-only figure planned with as a full T_total is silent.
        lg = _stub_collect(logs, log_lines=LOG_LINES)

        report = self._call(lg)

        note = next(n for n in report.notes if TRIGGER_FORCE_DESIRED in n)
        assert f"bounded separately at {POLICY_LAG_BOUND_S:.0f}s" in note
        assert "t_total_with_policy_bound_s" in note

    def test_stamps_the_configuration_it_was_given(self, logs: Any) -> None:
        lg = _stub_collect(logs, log_lines=LOG_LINES)

        report = self._call(
            lg,
            deployed_config={
                "instance_type": "ml.g5.xlarge",
                "image_digest": "139b9068c5eb",
                "container_env": {},
            },
        )

        assert report.config_slug == "g5xlarge-139b9068"

    def test_a_rebuild_with_no_configuration_still_assembles(self, logs: Any) -> None:
        # Optional so a report can be rebuilt from a window whose endpoint has since
        # changed. The slug then refuses to match a real fingerprint rather than
        # inventing one, which is what makes the planner's pairing check safe.
        lg = _stub_collect(logs, log_lines=LOG_LINES)

        report = self._call(lg)

        assert report.deployed_config == {}
        assert report.config_slug == "unknown-nodigest"

    def test_a_stream_with_no_markers_is_a_gap_not_a_failure(self, logs: Any) -> None:
        # An image built before the markers existed, or one whose startup lines aged
        # out. The AWS half is still fully attributed.
        lg = _stub_collect(logs, log_lines=LOG_LINES[:7])

        report = self._call(lg)

        assert str(TimelineStage.READY) in report.missing_stages
        assert any("no STAGE markers" in n for n in report.notes)
        assert report.at(TimelineStage.INSTANCE_LOGGING) is not None
        assert report.t_total_s == pytest.approx(220.0)

    def test_no_new_stream_is_a_gap_not_a_failure(self, logs: Any) -> None:
        # Logs lag the endpoint by a minute or two, so this is a routine outcome.
        lg = _stub_collect(logs, log_lines=LOG_LINES, new_stream=False)

        report = self._call(lg)

        assert report.instance_id is None
        assert str(TimelineStage.INSTANCE_LOGGING) in report.missing_stages
        assert any("no new log stream" in n for n in report.notes)

    def test_a_fleet_that_never_grew_does_not_evaluate_recovery(self, logs: Any) -> None:
        # No in_service means there is nothing to have recovered from. Running the bound
        # anyway would date recovery from the probe's own start and report a span that
        # measured the warm-up.
        lg = _stub_collect(logs, log_lines=LOG_LINES, new_stream=False)

        report = self._call(
            lg,
            event=ScaleEvent(
                from_instances=1, to_instances=1, desired_changed_at=None, in_service_at=None
            ),
        )

        entry = report.entry(TimelineStage.TRAFFIC_RECOVERED)
        assert entry is not None and "never reached the new instance count" in (entry.note or "")
        assert report.p95_before_ms is None
        assert report.t_total_bounded is True


class TestExplainNoScaleOut:
    """The error text after a timeout.

    A run that reaches this has already spent ``max_wait_s`` of real load, so the message
    has to name which outcome it was or the operator pays for another run to find out.
    Two outcomes, and they cost very different things to fix: ``Updating`` means SageMaker
    took the change and cannot place the instance, while a return to ``InService`` at the
    old count means it gave up without recording a failure anywhere.

    Pure now — no client. Capacity was set directly under a freeze, so no policy is in the
    path and there is no activity log worth reading.
    """

    def _explain(self, **overrides: Any) -> str:
        kwargs: dict[str, Any] = {
            "endpoint": ENDPOINT,
            "from_instances": 1,
            "max_wait_s": 1500.0,
        }
        kwargs.update(overrides)
        return _explain_no_scale_out(**kwargs)

    def test_still_updating_means_the_change_was_taken_but_not_placed(self) -> None:
        # The live outcome, and the one a longer --max-wait cannot fix: AWS accepted the
        # count, reserved the quota, and no instance ever arrived. The advice has to point
        # at hardware availability rather than at patience.
        message = self._explain(endpoint_status="Updating")

        assert "accepted the change and has not placed the instance" in message
        assert "instance capacity for the type" in message
        assert "another instance type or region" in message

    def test_back_in_service_means_aws_abandoned_the_change(self) -> None:
        # The observed ending: SageMaker returns the endpoint to InService at the old
        # count and records no FailureReason anywhere. Nothing else in the account says so.
        message = self._explain(endpoint_status="InService")

        assert "abandoned the change without recording a failure" in message

    @pytest.mark.parametrize("endpoint_status", ["Updating", "InService", None])
    def test_every_message_states_the_count_and_window_it_never_passed(
        self, endpoint_status: str | None
    ) -> None:
        # Whichever branch it takes, the timeout is the thing that happened. A diagnosis
        # that omits the count and the wait cannot be judged against the run's own flags.
        message = self._explain(from_instances=2, max_wait_s=600.0, endpoint_status=endpoint_status)

        assert "more than 2 instance(s) within 600s" in message

    @pytest.mark.parametrize("endpoint_status", ["Updating", "InService", None])
    def test_no_message_blames_a_policy(self, endpoint_status: str | None) -> None:
        # Nothing scaling was live: the freeze suspended the deployed policy and capacity
        # was set directly, so the scaling activity log is silent by design. Reading "the
        # policy never acted" out of that silence would send the operator to check a
        # threshold this run never used.
        message = self._explain(endpoint_status=endpoint_status)

        assert "the policy never acted" not in message
        assert "C_target" not in message
        assert "set directly" in message
        assert "scaling config" in message or "silent by design" in message


def _endpoint_config() -> dict[str, Any]:
    """The endpoint config ``fingerprint_or_registry`` reads the instance type from."""
    return {
        "EndpointConfigName": f"{ENDPOINT}-config",
        "EndpointConfigArn": (
            f"arn:aws:sagemaker:us-east-1:1234:endpoint-config/{ENDPOINT}-config"
        ),
        "ProductionVariants": [
            {
                "VariantName": VARIANT,
                "ModelName": "kokoro-model",
                # ml.g5.xlarge is what cost.MODEL_INSTANCE_TYPES says for kokoro, so the
                # fingerprint read stays silent and the test is about the freeze.
                "InstanceType": "ml.g5.xlarge",
                "InitialInstanceCount": 1,
            }
        ],
        "CreationTime": T0,
    }


def _model() -> dict[str, Any]:
    return {
        "ModelName": "kokoro-model",
        "ModelArn": "arn:aws:sagemaker:us-east-1:1234:model/kokoro-model",
        "CreationTime": T0,
        "PrimaryContainer": {
            "Image": "1234.dkr.ecr.us-east-1.amazonaws.com/cdk-assets:139b9068c5eb1f03",
            "Environment": {"MAX_REQUEST_AGE_S": "56"},
        },
    }


def _scalable_target(*, suspended: bool) -> dict[str, Any]:
    return {
        "ServiceNamespace": "sagemaker",
        "ResourceId": RID,
        "ScalableDimension": "sagemaker:variant:DesiredInstanceCount",
        "MinCapacity": 1,
        "MaxCapacity": 4,
        "RoleARN": "arn:aws:iam::1234:role/aws-service-role/sagemaker.application-autoscaling",
        "SuspendedState": {
            "DynamicScalingInSuspended": suspended,
            "DynamicScalingOutSuspended": suspended,
            "ScheduledScalingSuspended": suspended,
        },
        "CreationTime": T0,
    }


def _scaling_policy() -> dict[str, Any]:
    return {
        "PolicyARN": f"arn:aws:autoscaling:us-east-1:1234:scalingPolicy:abc:resource/{RID}",
        "PolicyName": "TrackConcurrency",
        "ServiceNamespace": "sagemaker",
        "ResourceId": RID,
        "ScalableDimension": "sagemaker:variant:DesiredInstanceCount",
        "PolicyType": "TargetTrackingScaling",
        "CreationTime": T0,
    }


def _stub_capture(aas_stub: Stubber, sm_stub: Stubber, *, suspended: bool) -> None:
    """Queue the three calls one ``fixture.capture()`` makes, in order."""
    aas_stub.add_response(
        "describe_scalable_targets", {"ScalableTargets": [_scalable_target(suspended=suspended)]}
    )
    aas_stub.add_response("describe_scaling_policies", {"ScalingPolicies": [_scaling_policy()]})
    sm_stub.add_response("describe_endpoint", _variant(desired=1, current=1))


def _record_calls(client: Any, name: str, calls: list[tuple[str, str]]) -> None:
    """Append ``(client_name, operation)`` for every call this client makes.

    The Stubbers still validate each request and response; this only records the
    *interleaving across two clients*, which is the claim being made — suspension before
    the capacity change, and both restores after the failure. Two separate response
    queues cannot express an ordering between them.
    """
    client.meta.events.register(
        "before-parameter-build.*.*",
        lambda **kwargs: calls.append((name, kwargs["model"].name)),
    )


class TestMeasureFreezesBeforeItRaisesCapacity:
    """The ordering that keeps a second cause out of the measurement.

    Application Auto Scaling suspension does not block our own
    ``UpdateEndpointWeightsAndCapacities``, so the freeze can come first and the trigger
    still works. If it did not, the probe's own load would fire the live policy — the
    deployed alarm reads ``ConcurrentRequestsPerModel``/``Maximum`` against 0.713, which
    any probe clears tenfold, and target tracking then jumps straight to ``max_capacity``.
    Instances arriving from two causes inside one measurement is the best explanation for
    the 1->4 jump observed on 2026-07-31, and it is invisible in the resulting number.
    """

    def _run(self, aas: Any, sm: Any, lg: Any, **overrides: Any) -> Any:
        kwargs: dict[str, Any] = {
            "model_name": "kokoro-82m",
            "endpoint": ENDPOINT,
            "variant": VARIANT,
            "ladder_p95_ms": LADDER,
            "texts": ["The birch canoe slid on the smooth planks."],
            "voice": "af_heart",
            "probe_concurrency": 10,
            # No warm-up and a token settle: the waits are this module's own choices, and
            # a test that slept through them would only be measuring time.sleep.
            "warmup_s": 0.0,
            "settle_s": 0.01,
            "poll_interval_s": 0.0,
            "appscaling": aas,
            "sagemaker": sm,
            "logs": lg,
        }
        kwargs.update(overrides)
        return measure(**kwargs)

    @pytest.fixture
    def no_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Replace the probe with a thread that exits at once, so no traffic is issued.

        `measure` joins whatever it is handed, so this has to be a started thread rather
        than a stand-in object.
        """

        def fake_start_probe(**_: Any) -> tuple[threading.Thread, datetime]:
            thread = threading.Thread(target=lambda: None, name="ttotal-probe-stub", daemon=True)
            thread.start()
            return thread, T0

        monkeypatch.setattr(ttotal_mod, "_start_probe", fake_start_probe)

    @pytest.fixture
    def no_scalable_precondition(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Make a ``require_scalable`` call fail the test rather than reach AWS.

        That precondition demands a *live* policy with headroom, which is exactly what
        this mode suspends. Calling it would both refuse a run that can work and reopen
        the mid-run scale-out this freeze exists to lock out.
        """
        called: list[str] = []

        def refuse(*args: Any, **kwargs: Any) -> Any:
            called.append("require_scalable")
            raise AssertionError("require_scalable has no place under a freeze")

        monkeypatch.setattr(fixture_mod, "require_scalable", refuse)
        return called

    def _queue_the_run(self, aas_stub: Stubber, sm_stub: Stubber, logs_stub: Stubber) -> None:
        """Queue every call of one run that fails while polling for the new instance.

        The failure is injected at the poll, which is the earliest point after the
        capacity change — so the restores below are exercised on the exception path
        rather than on a clean exit.
        """
        # fingerprint_or_registry: the configuration is read before anything is frozen,
        # because most of T_total is container start and a redeploy mid-run would
        # otherwise be attributed to whichever image replaced the one under test.
        sm_stub.add_response("describe_endpoint", _variant(desired=1, current=1))
        sm_stub.add_response("describe_endpoint_config", _endpoint_config())
        sm_stub.add_response("describe_model", _model())

        # freeze(): capture, suspend, pin (already at 1), verify.
        _stub_capture(aas_stub, sm_stub, suspended=False)
        aas_stub.add_response(
            "register_scalable_target",
            {},
            {
                "ServiceNamespace": "sagemaker",
                "ResourceId": RID,
                "ScalableDimension": "sagemaker:variant:DesiredInstanceCount",
                "SuspendedState": SUSPEND_ALL,
            },
        )
        sm_stub.add_response("describe_endpoint", _variant(desired=1, current=1))
        _stub_capture(aas_stub, sm_stub, suspended=True)
        # require_frozen(): the enforcement point, and a third capture.
        _stub_capture(aas_stub, sm_stub, suspended=True)
        # read_capacity() for the starting count, then the streams to diff against.
        sm_stub.add_response("describe_endpoint", _variant(desired=1, current=1))
        logs_stub.add_response("describe_log_streams", {"logStreams": []})

        # The trigger.
        sm_stub.add_response(
            "update_endpoint_weights_and_capacities",
            {"EndpointArn": f"arn:aws:sagemaker:us-east-1:1234:endpoint/{ENDPOINT}"},
            {
                "EndpointName": ENDPOINT,
                "DesiredWeightsAndCapacities": [
                    {"VariantName": VARIANT, "DesiredInstanceCount": 2}
                ],
            },
        )
        sm_stub.add_client_error("describe_endpoint", service_error_code="ThrottlingException")

        # restore_desired_count(), from the finally.
        sm_stub.add_response("describe_endpoint", _variant(desired=2, current=1))
        sm_stub.add_response(
            "update_endpoint_weights_and_capacities",
            {"EndpointArn": f"arn:aws:sagemaker:us-east-1:1234:endpoint/{ENDPOINT}"},
            {
                "EndpointName": ENDPOINT,
                "DesiredWeightsAndCapacities": [
                    {"VariantName": VARIANT, "DesiredInstanceCount": 1}
                ],
            },
        )
        # thaw(), from frozen.__exit__: the captured state, not a blanket resume.
        aas_stub.add_response(
            "register_scalable_target",
            {},
            {
                "ServiceNamespace": "sagemaker",
                "ResourceId": RID,
                "ScalableDimension": "sagemaker:variant:DesiredInstanceCount",
                "SuspendedState": {
                    "DynamicScalingInSuspended": False,
                    "DynamicScalingOutSuspended": False,
                    "ScheduledScalingSuspended": False,
                },
            },
        )

    def test_it_suspends_scaling_before_it_raises_the_count_and_restores_both_after(
        self,
        appscaling: Any,
        sagemaker: Any,
        logs: Any,
        no_probe: None,
        no_scalable_precondition: list[str],
    ) -> None:
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        lg, logs_stub = logs
        calls: list[tuple[str, str]] = []
        _record_calls(aas, "appscaling", calls)
        _record_calls(sm, "sagemaker", calls)
        self._queue_the_run(aas_stub, sm_stub, logs_stub)

        with pytest.raises(botocore.exceptions.ClientError):
            self._run(aas, sm, lg)

        suspended_at = calls.index(("appscaling", "RegisterScalableTarget"))
        raised_at = calls.index(("sagemaker", "UpdateEndpointWeightsAndCapacities"))
        assert suspended_at < raised_at
        # Both restores run on the way out of a failure, capacity first: it has to wait out
        # an `Updating` endpoint, which thaw's own unconditional restore does not.
        assert calls[-2:] == [
            ("sagemaker", "UpdateEndpointWeightsAndCapacities"),
            ("appscaling", "RegisterScalableTarget"),
        ]
        aas_stub.assert_no_pending_responses()
        sm_stub.assert_no_pending_responses()
        assert no_scalable_precondition == []

    def test_a_ladder_without_the_halving_rungs_is_refused_before_anything_is_touched(
        self,
        appscaling: Any,
        sagemaker: Any,
        logs: Any,
        no_probe: None,
        no_scalable_precondition: list[str],
    ) -> None:
        # Checked first because finding out afterwards costs the whole run: a real
        # scale-out, a fleet briefly parked at a raised count, and the probe's traffic.
        # No response is queued on any stub, so any AWS call fails this test.
        aas, aas_stub = appscaling
        sm, sm_stub = sagemaker
        lg, _ = logs

        with pytest.raises(TTotalError, match="no usable p95 at concurrency"):
            self._run(aas, sm, lg, ladder_p95_ms={1: 92.0, 10: 667.0})

        aas_stub.assert_no_pending_responses()
        sm_stub.assert_no_pending_responses()

    def test_another_trigger_is_refused(self, appscaling: Any, sagemaker: Any, logs: Any) -> None:
        # There is one trigger. Driving load past the deployed policy was removed because
        # it let the policy add instances during the measurement; accepting the string
        # again would silently restore that.
        aas, _ = appscaling
        sm, _ = sagemaker
        lg, _ = logs

        with pytest.raises(ValueError, match="trigger must be 'force-desired'"):
            self._run(aas, sm, lg, trigger="drive-load")
