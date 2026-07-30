"""Tests for the ``T_total`` stage decomposition.

Split by testability, the same way ``test_observe.py`` is. The timeline rules —
reconstructing a missing ``container_start``, bounding recovery, choosing the dominant
stage — are pure functions and are tested directly. The fetch paths use botocore
``Stubber``, which validates every response against the real service model, so a
``describe_log_streams`` reply missing a field or a ``GetMetricStatistics`` call with an
illegal period fails here rather than against a live endpoint.

``LOG_LINES`` is copied from a real ``speech-kokoro-82m`` stream, banner and all. Its
notable property is what it is *missing*: no ``container_start`` marker, because the
CUDA base image prints its banner before our entrypoint runs. That is the ordinary case,
not the edge case, which is why the reconstruction path is the one that has to work.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import boto3
import pytest
from botocore.stub import Stubber
from loguru import logger

from shared.stages import format_stage_marker, parse_stage_markers
from tts_bench.loadgen import LoadEvent
from tts_bench.observe import LogStream
from tts_bench.ttotal import (
    RECOVERY_MIN_SAMPLES,
    TRIGGER_DRIVE_LOAD,
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
    deployed_target_value,
    policy_alarm_names,
    read_capacity,
    recovery_bound,
    render_text,
    restore_desired_count,
    wait_for_scale_out,
)

ENDPOINT = "speech-kokoro-82m"
VARIANT = "primary"
RID = f"endpoint/{ENDPOINT}/variant/{VARIANT}"
STREAM = "primary/i-0abc123def4567890"
T0 = datetime(2026, 7, 30, 11, 0, 0, tzinfo=UTC)

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
        offered_rps=5.0,
        model="kokoro-82m",
        endpoint=ENDPOINT,
        scheduled_ts=(ts or 0.0) - 0.1,
        dispatch_ts=(ts or 0.0) - 0.1,
        first_byte_ts=ts,
        end_ts=(ts + 0.5) if ts else None,
        dispatch_delay_ms=0.0,
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
        in_flight_at_dispatch=2,
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
def cloudwatch() -> Any:
    client = boto3.client("cloudwatch", region_name="us-east-1")
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
        (entry,) = [e for e in container_timeline(parse_stage_markers(lines)) if not e.bounded]

        assert entry.stage == "tokenizer_ready"
        assert "not known" in (entry.note or "")

    def test_no_markers_yields_no_entries(self) -> None:
        # The image predates the markers, or its startup lines aged out. Neither raises.
        assert container_timeline(parse_stage_markers(LOG_LINES[:7])) == []
        assert container_timeline([]) == []


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


def _full_timeline() -> list[StageTime]:
    """A timeline with every stage observed, spanning 300s."""
    return assemble_timeline(
        load_applied_at=T0,
        metric_published_at=T0 + timedelta(seconds=20),
        metric_note=None,
        alarm_fired_at=T0 + timedelta(seconds=50),
        alarm_note="alarm TargetTracking-AlarmHigh",
        activity_started_at=T0 + timedelta(seconds=55),
        activity_note="Setting desired instance count to 2. (Successful)",
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
        in_service_at=T0 + timedelta(seconds=240),
        recovered_at=T0 + timedelta(seconds=300),
        recovery_note="p95 TTFAB held at or under 300ms across 40 completions",
    )


def _report(**overrides: Any) -> TTotalReport:
    report = TTotalReport(
        model_name="kokoro-82m",
        endpoint=ENDPOINT,
        run_id="run123",
        trigger=TRIGGER_DRIVE_LOAD,
        from_instances=1,
        to_instances=2,
    )
    report.timeline = _full_timeline()
    for key, value in overrides.items():
        setattr(report, key, value)
    return report


class TestTimelineAssembly:
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
    def test_t_total_spans_load_applied_to_recovery(self) -> None:
        assert _report().t_total_s == pytest.approx(300.0)

    def test_the_narrower_reading_starts_at_metric_publication(self) -> None:
        # The two differ by the client's own ramp, which a plan still has to absorb —
        # hence both, rather than picking one and losing the distinction.
        assert _report().t_total_from_metric_s == pytest.approx(280.0)

    def test_is_not_the_sum_of_its_stages(self) -> None:
        # The span, deliberately: summing would swallow any gap between two APIs' clocks
        # and under-report T_total, which is the dangerous direction.
        report = _report()
        assert report.t_total_s is not None
        assert sum(d.seconds for d in report.durations) <= report.t_total_s

    def test_flags_bounded_when_recovery_was_never_observed(self) -> None:
        report = _report()
        report.timeline = [
            e if e.stage != str(TimelineStage.TRAFFIC_RECOVERED) else StageTime(e.stage, None)
            for e in report.timeline
        ]

        assert report.t_total_bounded is True
        # Falls back to in_service rather than reporting nothing at all.
        assert report.t_total_s == pytest.approx(240.0)

    def test_dominant_stage_is_the_longest(self) -> None:
        dominant = _report().dominant_stage
        assert dominant is not None
        # activity start -> stream open: provisioning and image pull, the stage that
        # dominates a real kokoro scale-out.
        assert dominant.from_stage == str(TimelineStage.ACTIVITY_STARTED)
        assert dominant.seconds == pytest.approx(125.0)

    def test_splits_the_aws_half_from_the_container_half(self) -> None:
        report = _report()
        # Ends at container_start (11:03:05 = T0+185s), not at the stream opening.
        assert report.aws_share_s == pytest.approx(185.0)
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
        assert report.aws_share_s == pytest.approx(180.0)

    def test_missing_stages_are_listed_not_dropped(self) -> None:
        report = _report()
        report.timeline = [
            e if e.stage != str(TimelineStage.ALARM_FIRED) else StageTime(e.stage, None)
            for e in report.timeline
        ]
        assert str(TimelineStage.ALARM_FIRED) in report.missing_stages

    def test_provenance_records_the_trigger(self) -> None:
        from tts_bench.types import Origin

        prov = _report().provenance()
        assert prov.origin is Origin.MEASURED
        assert prov.run_id == "run123"
        assert prov.endpoint == ENDPOINT
        assert TRIGGER_DRIVE_LOAD in (prov.note or "")

    def test_provenance_warns_that_force_desired_is_only_half(self) -> None:
        # The guard against a container-only figure being planned with as a full T_total.
        prov = _report(trigger=TRIGGER_FORCE_DESIRED).provenance()
        assert "container half only" in (prov.note or "")

    def test_to_dict_is_json_serializable_with_shares(self) -> None:
        import json

        payload = _report().to_dict()
        assert json.loads(json.dumps(payload))["t_total_s"] == pytest.approx(300.0)
        shares = [d["share"] for d in payload["durations"]]
        assert all(0.0 <= s <= 1.0 for s in shares)

    def test_render_text_marks_bounded_stages(self) -> None:
        text = render_text(_report())
        assert "T_total for speech-kokoro-82m" in text
        assert "dominant stage" in text
        assert "~ marks a stage bounded by inference" in text

    def test_render_text_survives_an_empty_timeline(self) -> None:
        # A report with nothing observed still has to print. It is the outcome of a
        # scale event whose APIs all came back empty, which is worth seeing.
        report = _report()
        report.timeline = []
        text = render_text(report)
        assert "not measurable" in text


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


def _policy(*, target_value: float | None = 0.713, alarms: list[str] | None = None) -> dict:
    policy: dict[str, Any] = {
        "PolicyARN": f"arn:aws:autoscaling:us-east-1:1234:scalingPolicy:abc:resource/{RID}",
        "PolicyName": "TrackConcurrency",
        "ServiceNamespace": "sagemaker",
        "ResourceId": RID,
        "ScalableDimension": "sagemaker:variant:DesiredInstanceCount",
        "PolicyType": "TargetTrackingScaling",
        "CreationTime": T0,
        "Alarms": [
            {"AlarmName": name, "AlarmARN": f"arn:aws:cloudwatch:us-east-1:1234:alarm:{name}"}
            for name in (alarms or ["TargetTracking-AlarmHigh"])
        ],
    }
    if target_value is not None:
        policy["TargetTrackingScalingPolicyConfiguration"] = {
            "TargetValue": target_value,
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": ("SageMakerVariantConcurrentRequestsPerModelHighResolution")
            },
            "DisableScaleIn": True,
        }
    return policy


class TestPolicyReads:
    def test_reads_the_deployed_target_value(self, appscaling: Any) -> None:
        # Read off the deployment, not from config: a config that has moved ahead of the
        # last `cdk deploy` would put metric_published at the wrong threshold.
        client, stub = appscaling
        stub.add_response("describe_scaling_policies", {"ScalingPolicies": [_policy()]})

        assert deployed_target_value(client, endpoint=ENDPOINT) == pytest.approx(0.713)

    def test_no_target_tracking_policy_returns_none(self, appscaling: Any) -> None:
        client, stub = appscaling
        step_only = _policy(target_value=None)
        step_only["PolicyType"] = "StepScaling"
        stub.add_response("describe_scaling_policies", {"ScalingPolicies": [step_only]})

        assert deployed_target_value(client, endpoint=ENDPOINT) is None

    def test_alarm_names_come_from_the_policy(self, appscaling: Any) -> None:
        # Target tracking generates its alarm names, so they cannot be predicted from
        # config — they have to be read back.
        client, stub = appscaling
        stub.add_response(
            "describe_scaling_policies",
            {"ScalingPolicies": [_policy(alarms=["AlarmHigh", "AlarmLow"])]},
        )

        assert policy_alarm_names(client, endpoint=ENDPOINT) == ["AlarmHigh", "AlarmLow"]

    def test_a_read_failure_degrades_to_no_attribution(self, appscaling: Any) -> None:
        client, stub = appscaling
        stub.add_client_error("describe_scaling_policies", service_error_code="AccessDenied")

        assert policy_alarm_names(client, endpoint=ENDPOINT) == []


#: Verbatim from the ``StatusMessage`` of the activity that blocked the first live run.
#: The wording is AWS's, which is why the module quotes it rather than classifying it.
QUOTA_REFUSAL = (
    "Failed to set desired instance count to 4. Reason: The account-level service limit "
    "'ml.g5.xlarge for endpoint usage' is 4 Instances, with current utilization of 3 "
    "Instances and a request delta of 3 Instances. Please use AWS Service Quotas to "
    "request an increase for this quota."
)


def _activity(
    *,
    at_s: float,
    status: str = "Successful",
    activity_id: str = "act-1",
    description: str = "Setting desired instance count to 2.",
    status_message: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "ActivityId": activity_id,
        "ServiceNamespace": "sagemaker",
        "ResourceId": RID,
        "ScalableDimension": "sagemaker:variant:DesiredInstanceCount",
        "Description": description,
        "Cause": "monitor alarm TargetTracking-AlarmHigh in state ALARM",
        "StartTime": T0 + timedelta(seconds=at_s),
        "StatusCode": status,
    }
    if status == "Successful":
        entry["EndTime"] = T0 + timedelta(seconds=at_s + 185)
    if status_message is not None:
        entry["StatusMessage"] = status_message
    return entry


def _stub_collect(
    cloudwatch: Any,
    appscaling: Any,
    logs: Any,
    *,
    log_lines: list[str],
    concurrency: float = 2.0,
    new_stream: bool = True,
    activities: list[dict[str, Any]] | None = None,
) -> tuple[Any, Any, Any]:
    """Queue one full pass of ``collect_timeline``'s reads, in call order."""
    cw_client, cw_stub = cloudwatch
    aas_client, aas_stub = appscaling
    logs_client, logs_stub = logs

    cw_stub.add_response(
        "get_metric_statistics",
        {
            "Datapoints": [
                {
                    "Timestamp": T0 + timedelta(seconds=20),
                    "Maximum": concurrency,
                    "Average": concurrency,
                    "Unit": "None",
                }
            ]
        },
    )
    aas_stub.add_response("describe_scaling_policies", {"ScalingPolicies": [_policy()]})
    cw_stub.add_response(
        "describe_alarm_history",
        {
            "AlarmHistoryItems": [
                {
                    "AlarmName": "TargetTracking-AlarmHigh",
                    "Timestamp": T0 + timedelta(seconds=50),
                    "HistoryItemType": "StateUpdate",
                    "HistorySummary": "Alarm updated from OK to ALARM",
                }
            ]
        },
    )
    aas_stub.add_response(
        "describe_scaling_activities",
        {"ScalingActivities": activities if activities is not None else [_activity(at_s=55)]},
    )
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
    return cw_client, aas_client, logs_client


class TestCollectTimeline:
    def _event(self) -> ScaleEvent:
        return ScaleEvent(
            from_instances=1,
            to_instances=2,
            desired_changed_at=T0 + timedelta(seconds=60),
            in_service_at=T0 + timedelta(seconds=240),
        )

    def _call(self, cw: Any, aas: Any, lg: Any, **overrides: Any) -> TTotalReport:
        kwargs: dict[str, Any] = {
            "cloudwatch": cw,
            "appscaling": aas,
            "logs": lg,
            "endpoint": ENDPOINT,
            "variant": VARIANT,
            "model_name": "kokoro-82m",
            "run_id": "run123",
            "trigger": TRIGGER_DRIVE_LOAD,
            "scaling_target": 0.713,
            "ttfab_budget_ms": 300.0,
            "load_applied_at": T0,
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
            "load_events": [
                *_events(T0, 60, ttfab_ms=1200.0, every_s=2.0),
                *_events(T0 + timedelta(seconds=250), 40, ttfab_ms=120.0),
            ],
        }
        kwargs.update(overrides)
        return collect_timeline(**kwargs)

    def test_assembles_every_stage_from_its_own_api(
        self, cloudwatch: Any, appscaling: Any, logs: Any
    ) -> None:
        cw, aas, lg = _stub_collect(cloudwatch, appscaling, logs, log_lines=LOG_LINES)

        report = self._call(cw, aas, lg)

        assert report.at(TimelineStage.METRIC_PUBLISHED) == T0 + timedelta(seconds=20)
        assert report.at(TimelineStage.ALARM_FIRED) == T0 + timedelta(seconds=50)
        assert report.at(TimelineStage.ACTIVITY_STARTED) == T0 + timedelta(seconds=55)
        assert report.at(TimelineStage.INSTANCE_LOGGING) == T0 + timedelta(seconds=180)
        assert report.at(TimelineStage.READY) is not None
        assert report.at(TimelineStage.IN_SERVICE) == T0 + timedelta(seconds=240)
        assert report.at(TimelineStage.TRAFFIC_RECOVERED) == T0 + timedelta(seconds=250)
        assert report.t_total_s == pytest.approx(250.0)

    def test_names_the_new_instance_by_log_stream_diff(
        self, cloudwatch: Any, appscaling: Any, logs: Any
    ) -> None:
        # SageMaker does not report which instance it added; the set difference over
        # stream names is the only way to identify it.
        cw, aas, lg = _stub_collect(cloudwatch, appscaling, logs, log_lines=LOG_LINES)

        report = self._call(cw, aas, lg)

        assert report.instance_id == "i-0abc123def4567890"

    def test_records_the_degradation_being_recovered_from(
        self, cloudwatch: Any, appscaling: Any, logs: Any
    ) -> None:
        cw, aas, lg = _stub_collect(cloudwatch, appscaling, logs, log_lines=LOG_LINES)

        report = self._call(cw, aas, lg)

        assert report.p95_before_ms == pytest.approx(1200.0)
        assert report.p95_after_ms == pytest.approx(120.0)
        assert report.requests_before > 0 and report.requests_after > 0

    def test_a_stream_with_no_markers_is_a_gap_not_a_failure(
        self, cloudwatch: Any, appscaling: Any, logs: Any
    ) -> None:
        # An image built before the markers existed, or one whose startup lines aged
        # out. The AWS half is still fully attributed.
        cw, aas, lg = _stub_collect(cloudwatch, appscaling, logs, log_lines=LOG_LINES[:7])

        report = self._call(cw, aas, lg)

        assert str(TimelineStage.READY) in report.missing_stages
        assert any("no STAGE markers" in n for n in report.notes)
        assert report.at(TimelineStage.ALARM_FIRED) is not None
        assert report.t_total_s == pytest.approx(250.0)

    def test_no_new_stream_is_a_gap_not_a_failure(
        self, cloudwatch: Any, appscaling: Any, logs: Any
    ) -> None:
        # Logs lag the endpoint by a minute or two, so this is a routine outcome.
        cw, aas, lg = _stub_collect(
            cloudwatch, appscaling, logs, log_lines=LOG_LINES, new_stream=False
        )

        report = self._call(cw, aas, lg)

        assert report.instance_id is None
        assert str(TimelineStage.INSTANCE_LOGGING) in report.missing_stages
        assert any("no new log stream" in n for n in report.notes)

    def test_says_so_when_concurrency_never_reached_the_target(
        self, cloudwatch: Any, appscaling: Any, logs: Any
    ) -> None:
        # Capacity changed but the metric never showed the crossing: detection lag is
        # unattributed, and the report must not imply otherwise.
        cw, aas, lg = _stub_collect(
            cloudwatch, appscaling, logs, log_lines=LOG_LINES, concurrency=0.1
        )

        report = self._call(cw, aas, lg)

        assert str(TimelineStage.METRIC_PUBLISHED) in report.missing_stages
        assert report.t_total_from_metric_s is None
        assert any("never reached the scaling target" in n for n in report.notes)

    def test_force_desired_skips_the_policy_reads_and_says_why(
        self, cloudwatch: Any, appscaling: Any, logs: Any
    ) -> None:
        # No metric, no alarm — and the note has to be explicit that this is not a full
        # T_total, or a container-only figure ends up sized against a real surge.
        _, aas_stub = appscaling
        _, logs_stub = logs
        aas_stub.add_response("describe_scaling_activities", {"ScalingActivities": []})
        logs_stub.add_response("describe_log_streams", {"logStreams": []})

        report = self._call(cloudwatch[0], appscaling[0], logs[0], trigger=TRIGGER_FORCE_DESIRED)

        assert str(TimelineStage.METRIC_PUBLISHED) in report.missing_stages
        assert str(TimelineStage.ALARM_FIRED) in report.missing_stages
        assert any("must not be fed to the planner" in n for n in report.notes)
        cloudwatch[1].assert_no_pending_responses()

    def test_an_unrecovered_run_is_flagged_as_a_floor(
        self, cloudwatch: Any, appscaling: Any, logs: Any
    ) -> None:
        cw, aas, lg = _stub_collect(cloudwatch, appscaling, logs, log_lines=LOG_LINES)

        report = self._call(cw, aas, lg, load_events=_events(T0, 200, ttfab_ms=2000.0, every_s=2.0))

        assert report.t_total_bounded is True
        assert str(TimelineStage.TRAFFIC_RECOVERED) in report.missing_stages
        assert any("floor" in n for n in report.notes)
        # Still reports a number, ending at in_service.
        assert report.t_total_s == pytest.approx(240.0)

    def test_times_the_activity_that_worked_not_the_ones_refused(
        self, cloudwatch: Any, appscaling: Any, logs: Any
    ) -> None:
        # A quota-blocked policy retries every ten seconds, so the *first* activity in
        # the window is a rejection. Timing it would attribute the whole AWS half to an
        # attempt that changed nothing — here, 55s early.
        cw, aas, lg = _stub_collect(
            cloudwatch,
            appscaling,
            logs,
            log_lines=LOG_LINES,
            activities=[
                _activity(at_s=0, status="Failed", activity_id="f1", status_message=QUOTA_REFUSAL),
                _activity(at_s=10, status="Failed", activity_id="f2", status_message=QUOTA_REFUSAL),
                _activity(at_s=55, activity_id="ok"),
            ],
        )

        report = self._call(cw, aas, lg)

        assert report.at(TimelineStage.ACTIVITY_STARTED) == T0 + timedelta(seconds=55)

    def test_failed_activities_are_reported_in_awss_own_words(
        self, cloudwatch: Any, appscaling: Any, logs: Any
    ) -> None:
        # The StatusMessage is the only place the reason appears anywhere in AWS, so it
        # is quoted rather than summarized — the set of reasons is AWS's to extend.
        cw, aas, lg = _stub_collect(
            cloudwatch,
            appscaling,
            logs,
            log_lines=LOG_LINES,
            activities=[
                _activity(at_s=0, status="Failed", activity_id="f1", status_message=QUOTA_REFUSAL),
                _activity(at_s=55, activity_id="ok"),
            ],
        )

        report = self._call(cw, aas, lg)

        note = next(n for n in report.notes if "FAILED" in n)
        assert "1 of 2" in note
        assert "ml.g5.xlarge for endpoint usage" in note
        assert "InService at its old count" in note

    def test_an_all_failed_window_still_reports_a_timeline(
        self, cloudwatch: Any, appscaling: Any, logs: Any
    ) -> None:
        # Reachable: capacity came from somewhere else — a manual bump, or a scale-in
        # reversing — while every policy attempt was refused. The report says so instead
        # of implying the policy delivered the instance.
        cw, aas, lg = _stub_collect(
            cloudwatch,
            appscaling,
            logs,
            log_lines=LOG_LINES,
            activities=[
                _activity(at_s=0, status="Failed", activity_id="f1", status_message=QUOTA_REFUSAL)
            ],
        )

        report = self._call(cw, aas, lg)

        entry = next(e for e in report.timeline if e.stage == str(TimelineStage.ACTIVITY_STARTED))
        assert entry.note is not None and "no activity succeeded" in entry.note
        assert report.at(TimelineStage.IN_SERVICE) == T0 + timedelta(seconds=240)


class TestExplainNoScaleOut:
    """The error text after a timeout.

    A run that reaches this has already spent ``max_wait_s`` of real load. Two causes
    are indistinguishable from the endpoint — which stays ``InService`` at its old count
    either way — so the message has to name which one it was, or the operator pays for
    another run to find out.
    """

    def _explain(self, client: Any, **overrides: Any) -> str:
        kwargs: dict[str, Any] = {
            "endpoint": ENDPOINT,
            "variant": VARIANT,
            "from_instances": 1,
            "max_wait_s": 1500.0,
            "scaling_target": 0.713,
            "window_start": T0,
            "window_end": T0 + timedelta(seconds=1500),
        }
        kwargs.update(overrides)
        return _explain_no_scale_out(client, **kwargs)

    def test_a_refusal_is_quoted_and_named_as_one(self, appscaling: Any) -> None:
        client, stub = appscaling
        stub.add_response(
            "describe_scaling_activities",
            {
                "ScalingActivities": [
                    _activity(
                        at_s=100,
                        status="Failed",
                        activity_id="f1",
                        description="Setting desired instance count to 4.",
                        status_message=QUOTA_REFUSAL,
                    )
                ]
            },
        )

        message = self._explain(client)

        assert "The policy DID act" in message
        assert "not a detection problem" in message
        assert "ml.g5.xlarge for endpoint usage" in message

    def test_a_slow_provision_is_told_apart_from_a_refusal(self, appscaling: Any) -> None:
        # AWS accepted "set desired to 4" and is still pulling images. Reporting that as a
        # failure sent one live run's reader after the quota, which had already been fixed.
        # Opposite fix: wait longer.
        client, stub = appscaling
        stub.add_response(
            "describe_scaling_activities",
            {"ScalingActivities": [_activity(at_s=50, status="InProgress", activity_id="live")]},
        )

        message = self._explain(client, window_end=T0 + timedelta(seconds=500))

        assert "ACCEPTED" in message
        assert "not refusing" in message
        assert "--max-wait" in message
        assert "lower max_capacity" in message
        # The elapsed figure, so "still provisioning" can be judged against how long.
        assert "450s" in message
        # And emphatically NOT the word that sent the last read astray.
        assert "failed" not in message.lower()

    def test_a_very_long_in_flight_reads_as_capacity_not_patience(self, appscaling: Any) -> None:
        """The live outcome, and the one a longer ``--max-wait`` cannot fix.

        AWS accepted "set desired instance count to 4", ``AWS/Usage`` for
        ``endpoint/ml.g5.xlarge`` went to 4 — the quota *was* reserved — and then no
        instance ever arrived. 34 minutes later the endpoint returned to ``InService`` at
        its old count with no ``FailureReason`` and the activity never left ``InProgress``.
        Advising a longer wait there burns another run for the same nothing.
        """
        client, stub = appscaling
        stub.add_response(
            "describe_scaling_activities",
            {
                "ScalingActivities": [
                    _activity(
                        at_s=50,
                        status="InProgress",
                        activity_id="live",
                        description="Setting desired instance count to 4.",
                        status_message=(
                            "Successfully set desired instance count to 4. Waiting for "
                            "change to be fulfilled by sagemaker."
                        ),
                    )
                ]
            },
        )

        message = self._explain(client)

        assert "ACCEPTED" in message
        assert "capacity being unavailable" in message
        assert "24min" in message
        # The advice has to invert, or the reader pays for the same 25 minutes again.
        assert "another instance type or region rather than a longer --max-wait" in message
        # It may say AWS did not *report* a failure; it must not claim one occurred.
        assert "scaling activities failed" not in message

    def test_a_refusal_wins_over_a_later_retry_in_flight(self, appscaling: Any) -> None:
        # A window can hold both. The refusal is the more expensive finding, so it leads.
        client, stub = appscaling
        stub.add_response(
            "describe_scaling_activities",
            {
                "ScalingActivities": [
                    _activity(
                        at_s=10, status="Failed", activity_id="f1", status_message=QUOTA_REFUSAL
                    ),
                    _activity(at_s=60, status="InProgress", activity_id="live"),
                ]
            },
        )

        assert "The policy DID act" in self._explain(client)

    def test_an_all_terminal_window_that_still_did_not_arrive(self, appscaling: Any) -> None:
        # Succeeded, nothing in flight, and yet the count never rose. Nothing to blame,
        # so it says where to look rather than inventing a cause.
        client, stub = appscaling
        stub.add_response(
            "describe_scaling_activities",
            {"ScalingActivities": [_activity(at_s=100, activity_id="ok")]},
        )

        message = self._explain(client)

        assert "none failed and none is still in flight" in message
        assert "describe-scaling-activities" in message

    def test_no_activity_at_all_points_at_the_offered_load(self, appscaling: Any) -> None:
        # The policy never decided, so the fault is upstream: too little load, or an
        # alarm still in INSUFFICIENT_DATA.
        client, stub = appscaling
        stub.add_response("describe_scaling_activities", {"ScalingActivities": []})

        message = self._explain(client)

        assert "the policy never acted" in message
        assert "C_target=0.713" in message
        assert "tts-bench drift" in message

    def test_every_message_states_the_count_it_never_passed(self, appscaling: Any) -> None:
        client, stub = appscaling
        stub.add_response("describe_scaling_activities", {"ScalingActivities": []})

        message = self._explain(client, from_instances=2, max_wait_s=600.0)

        assert "more than 2 instance(s) within 600s" in message

    def test_an_unreadable_activity_log_does_not_mask_the_timeout(self, appscaling: Any) -> None:
        # scaling_activities() degrades to [] on AccessDenied, so this reads as "never
        # acted". The timeout itself still has to survive being reported.
        client, stub = appscaling
        stub.add_client_error("describe_scaling_activities", service_error_code="AccessDenied")

        assert "did not reach more than 1 instance(s)" in self._explain(client)
