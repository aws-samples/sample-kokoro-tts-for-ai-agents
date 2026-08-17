# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Tests for the container startup stage-marker format contract."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from shared.stages import (
    Stage,
    StageMarker,
    format_stage_marker,
    parse_stage_marker,
    parse_stage_markers,
)


class TestParseStageMarker:
    def test_parses_a_well_formed_line(self) -> None:
        marker = parse_stage_marker(
            "=== STAGE weights_ready t=2026-07-29T18:04:05.123Z elapsed_s=42.500 ==="
        )
        assert marker is not None
        assert marker.name == Stage.WEIGHTS_READY
        assert marker.at == datetime(2026, 7, 29, 18, 4, 5, 123000, tzinfo=UTC)
        assert marker.elapsed_s == 42.5
        assert marker.known

    def test_returns_none_for_a_non_marker_line(self) -> None:
        # The containers interleave markers with diagnostics worth keeping, so the
        # overwhelming majority of lines are legitimately not markers.
        assert parse_stage_marker("INFO: Model download complete. Size: 4.2G") is None

    def test_returns_none_for_the_legacy_container_start_banners(self) -> None:
        # These predate the contract and are what 3a normalizes; they must not be
        # mistaken for markers, or every stream would report a bogus stage.
        assert parse_stage_marker("=== CONTAINER START: 2026-07-29T18:04:05Z | host=x ===") is None
        assert (
            parse_stage_marker("=== CHATTERBOX CONTAINER START: 2026-07-29T18:04:05Z ===") is None
        )

    def test_tolerates_a_cloudwatch_line_prefix(self) -> None:
        # CloudWatch Logs prepends its own text, so anchoring the pattern to the
        # start of the line would find nothing in real data.
        marker = parse_stage_marker(
            "2026-07-29T18:04:05.900Z abc123 "
            "=== STAGE ready t=2026-07-29T18:04:05.888Z elapsed_s=61.001 ==="
        )
        assert marker is not None
        assert marker.name == Stage.READY

    def test_accepts_an_unknown_stage_name_as_data(self) -> None:
        # A deployed image may be ahead of or behind this module; an unrecognized
        # name is information, not a parse failure.
        marker = parse_stage_marker(
            "=== STAGE some_future_stage t=2026-07-29T18:04:05.000Z elapsed_s=1.000 ==="
        )
        assert marker is not None
        assert marker.name == "some_future_stage"
        assert not marker.known

    def test_rejects_elapsed_without_a_value(self) -> None:
        assert (
            parse_stage_marker("=== STAGE ready t=2026-07-29T18:04:05.000Z elapsed_s= ===") is None
        )

    @pytest.mark.parametrize(
        "timestamp",
        ["2026-07-29T18:04:05", "2026-07-29T18:04:05.12", "2026-07-29 18:04:05.123"],
        ids=["no-millis", "two-digit-millis", "space-separated"],
    )
    def test_rejects_malformed_timestamps(self, timestamp: str) -> None:
        # Fixed-width millis is what lets `strptime` parse without a fallback ladder.
        assert parse_stage_marker(f"=== STAGE ready t={timestamp}Z elapsed_s=1.000 ===") is None


class TestFormatStageMarker:
    def test_round_trips_through_the_parser(self) -> None:
        at = datetime(2026, 7, 29, 18, 4, 5, 123456, tzinfo=UTC)
        marker = parse_stage_marker(format_stage_marker(Stage.WARMUP_DONE, at, 12.25))
        assert marker is not None
        assert marker.name == Stage.WARMUP_DONE
        assert marker.elapsed_s == 12.25
        # Microseconds are truncated to millis by the format, not carried through.
        assert marker.at == at.replace(microsecond=123000)

    def test_renders_millis_zero_padded(self) -> None:
        line = format_stage_marker(
            Stage.READY, datetime(2026, 1, 2, 3, 4, 5, 7000, tzinfo=UTC), 1.0
        )
        assert "t=2026-01-02T03:04:05.007Z" in line

    def test_converts_a_non_utc_input_to_utc(self) -> None:
        from datetime import timedelta, timezone

        at = datetime(2026, 7, 29, 20, 4, 5, tzinfo=timezone(timedelta(hours=2)))
        assert "t=2026-07-29T18:04:05.000Z" in format_stage_marker(Stage.READY, at, 0.0)

    @pytest.mark.parametrize("stage", list(Stage), ids=lambda s: s.value)
    def test_every_known_stage_round_trips(self, stage: Stage) -> None:
        at = datetime(2026, 7, 29, 18, 4, 5, tzinfo=UTC)
        marker = parse_stage_marker(format_stage_marker(stage, at, 1.0))
        assert marker is not None
        assert marker.name == stage
        assert marker.known


class TestParseStageMarkers:
    def test_preserves_emission_order_and_drops_noise(self) -> None:
        lines = [
            "=== STAGE container_start t=2026-07-29T18:03:00.000Z elapsed_s=0.001 ===",
            "--- GPU ---",
            "=== STAGE weights_fetched t=2026-07-29T18:03:30.000Z elapsed_s=30.000 ===",
            "INFO: Starting vLLM with args: --port 8000",
            "=== STAGE ready t=2026-07-29T18:04:05.000Z elapsed_s=65.000 ===",
        ]
        markers = parse_stage_markers(lines)
        assert [m.name for m in markers] == [
            Stage.CONTAINER_START,
            Stage.WEIGHTS_FETCHED,
            Stage.READY,
        ]
        assert [m.elapsed_s for m in markers] == [0.001, 30.0, 65.0]

    def test_returns_empty_for_a_stream_with_no_markers(self) -> None:
        # A pre-3a image, which `ttotal` must degrade on rather than raise.
        assert parse_stage_markers(["INFO: Model loaded", "=== CONTAINER START: x ==="]) == []


class TestStageMarker:
    def test_is_frozen(self) -> None:
        marker = StageMarker(name="ready", at=datetime.now(UTC), elapsed_s=1.0)
        with pytest.raises((AttributeError, TypeError)):
            marker.elapsed_s = 2.0  # type: ignore[misc]
