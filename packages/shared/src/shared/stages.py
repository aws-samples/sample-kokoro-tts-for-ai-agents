# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Container startup stage markers: the format contract between emitter and parser.

`T_total` — the lag from a scaling metric crossing its threshold to a new instance
serving traffic — is the number the whole capacity plan is most sensitive to, and it
is only actionable when attributed to a stage. Attributing it needs the containers to
say where their startup time went.

The emitters are **duplicated** across the four container images. That is not an
oversight: each container is its own Docker build context
(`DockerImageAsset(directory=container_dir)`), so they cannot import this module, and
two of them emit from shell rather than Python. This module is therefore the single
place the *format* is defined, so a divergent emitter is caught by a test rather than
by a silently empty `ttotal` report.

The line format, byte-identical everywhere::

    === STAGE <name> t=<iso8601>.<ms>Z elapsed_s=<float> ===

`elapsed_s` runs from container start, so a log stream whose earlier lines aged out of
CloudWatch is still partially usable — the absolute timestamp anchors it, and the
elapsed value survives losing the `container_start` line entirely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum


class Stage(StrEnum):
    """Startup stages a container may report.

    Not every container emits every stage, and the order differs by design — the two
    S3-syncing images fetch weights in the shell *before* the framework starts, while
    the kokoro images bake weights into the image and load them inside the Python
    lifespan. Consumers should read the sequence as emitted rather than assume one.
    """

    #: Process entry. Bounds image pull only from outside, via the log stream's
    #: `firstEventTimestamp` — a container cannot observe its own pull.
    CONTAINER_START = "container_start"

    #: Model bytes are on local disk. Emitted only by the containers that sync from
    #: S3; for the kokoro images it would always be zero.
    WEIGHTS_FETCHED = "weights_fetched"

    #: Runtime is up and configured, before weights are resident.
    FRAMEWORK_INIT = "framework_init"

    #: Weights are loaded and resident on the accelerator.
    WEIGHTS_READY = "weights_ready"

    #: One discarded inference has run. Emitted even when the warm-up *failed*, so
    #: the sequence stays complete and the cost stays visible either way.
    WARMUP_DONE = "warmup_done"

    #: Accepting traffic.
    READY = "ready"


#: Matches one emitted marker. Tolerant of surrounding text because CloudWatch log
#: lines arrive with their own prefixes, and of an unknown stage name because a
#: container may be ahead of this module — an unrecognized name is data, not an error.
STAGE_MARKER_RE = re.compile(
    r"===\s+STAGE\s+(?P<name>[a-z_]+)\s+"
    r"t=(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})Z\s+"
    r"elapsed_s=(?P<elapsed_s>\d+(?:\.\d+)?)\s+==="
)


@dataclass(frozen=True, slots=True)
class StageMarker:
    """One parsed startup-stage marker."""

    #: The stage name as emitted. A plain `str`, not `Stage`: a container running an
    #: image built before a stage was renamed must still parse.
    name: str

    #: When the stage completed, UTC.
    at: datetime

    #: Seconds from container start, as the container measured it. Preferred over
    #: differencing timestamps, which have only millisecond resolution and depend on
    #: the `container_start` line still being in the stream.
    elapsed_s: float

    @property
    def known(self) -> bool:
        """Whether `name` is a stage this module knows about."""
        return self.name in _KNOWN_STAGE_VALUES


_KNOWN_STAGE_VALUES = frozenset(s.value for s in Stage)


def format_stage_marker(name: str, at: datetime, elapsed_s: float) -> str:
    """Render a marker line exactly as the containers emit it.

    Exists for tests: it is what lets a fixture assert an emitter matches the parser
    without either side hard-coding the other's string.
    """
    stamp = at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S")
    return f"=== STAGE {name} t={stamp}.{at.microsecond // 1000:03d}Z elapsed_s={elapsed_s:.3f} ==="


def parse_stage_marker(line: str) -> StageMarker | None:
    """Parse one log line, or return None if it carries no marker.

    Returning None rather than raising is deliberate: the containers interleave these
    with human-readable diagnostics that are worth keeping, so the overwhelming
    majority of lines are legitimately not markers.
    """
    match = STAGE_MARKER_RE.search(line)
    if match is None:
        return None
    return StageMarker(
        name=match.group("name"),
        at=datetime.strptime(match.group("timestamp"), "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=UTC),
        elapsed_s=float(match.group("elapsed_s")),
    )


def parse_stage_markers(lines: list[str]) -> list[StageMarker]:
    """Parse every marker in a log stream, in the order emitted."""
    return [m for line in lines if (m := parse_stage_marker(line)) is not None]
