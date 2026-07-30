"""Guards the startup-stage markers the four container images emit.

`tts-bench ttotal` attributes scaling lag by parsing these lines out of CloudWatch,
so an emitter that drifts from `shared.stages` does not fail loudly — it produces an
empty report that looks like a container with nothing to say. The format is duplicated
across four images (each is its own Docker build context, and two emit from shell), so
these tests are the only thing holding the copies together.

The container runtime deps are not installed in the lean infra dev env, so the
emitters cannot be imported here. Instead the Python helpers are executed in a
subprocess with the surrounding module stripped away, and the shell helpers are
sourced directly — in both cases running the *real* code from the *real* file, so a
change to an emitter is caught rather than a change to a copy of it.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shared.stages import Stage, parse_stage_marker

_CONTAINERS = Path(__file__).resolve().parents[1] / "containers"

#: Every image that emits markers, and the file each emits them from.
_PYTHON_EMITTERS = {
    "kokoro": _CONTAINERS / "kokoro" / "serve.py",
    "kokoro-cpu": _CONTAINERS / "kokoro-cpu" / "serve.py",
    "chatterbox": _CONTAINERS / "chatterbox" / "streaming_proxy.py",
}
_SHELL_EMITTERS = {
    "chatterbox": _CONTAINERS / "chatterbox" / "entrypoint.sh",
    "vllm": _CONTAINERS / "vllm" / "entrypoint.sh",
}

#: The shell `stage` function, from its opening line to its closing brace.
_SH_STAGE_RE = re.compile(r"^stage\(\) \{\n(?:.*?\n)*?^\}\n", re.MULTILINE)


def _fail_missing(what: str, path: Path) -> None:
    pytest.fail(
        f"Could not find the {what} in {path}. If it was renamed or restructured, "
        f"update the extraction in this test — do not delete the assertion, it is what "
        f"keeps `tts-bench ttotal` able to read this container's logs."
    )


def _extract(pattern: re.Pattern[str], path: Path, what: str) -> str:
    match = pattern.search(path.read_text())
    if match is None:
        _fail_missing(what, path)
    return match.group(0)  # type: ignore[union-attr]


def _extract_py_stage(path: Path) -> str:
    """Lift `_STAGE_EPOCH` and `def _stage` out of a container's serve module.

    Via `ast` rather than a regex: the module cannot be imported here (its deps are
    absent) but it can always be parsed, and `get_source_segment` returns the real
    source rather than a pattern's guess at where the helper ends.
    """
    source = path.read_text()
    tree = ast.parse(source)

    epoch = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_STAGE_EPOCH" for t in node.targets)
        ),
        None,
    )
    helper = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_stage"),
        None,
    )
    if epoch is None:
        _fail_missing("_STAGE_EPOCH assignment", path)
    if helper is None:
        _fail_missing("_stage helper", path)

    return "\n".join(
        (
            ast.get_source_segment(source, epoch),  # type: ignore[arg-type]
            ast.get_source_segment(source, helper),  # type: ignore[arg-type]
        )
    )


def _run_python_emitter(path: Path, stage_name: str) -> str:
    """Execute a container's real `_stage` helper in isolation and return its output."""
    snippet = _extract_py_stage(path)
    # `os` and `time` are the only names the helper closes over. Running it detached
    # from its module is what lets this test cover images whose deps (torch, kokoro,
    # chatterbox) are absent here. `str()` because a `Stage` member's repr is not a
    # literal the subprocess can parse.
    program = f"import os, time\n{snippet}\n_stage({str(stage_name)!r})\n"
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=30,
        env={"CONTAINER_START_EPOCH": "1000.0", "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, f"{path} _stage helper failed: {result.stderr}"
    return result.stdout.strip()


#: Where the skewing `date` below starts its clock. Arbitrary, but far from any real
#: `elapsed_s` so a mismatch is unmistakable in the failure message.
_FAKE_START_EPOCH = 1800000000.0

#: How far the skewing clock jumps per reading — a day, so no plausible startup
#: duration could account for the gap.
_FAKE_SKEW_S = 86400


def _skewing_date_on_path(tmp_path: Path) -> str:
    """Shell preamble installing a `date` whose clock jumps a day per *reading*.

    Formatting calls (those passing `-d`) pass through untouched, because converting a
    timestamp already in hand is not a clock reading — only calls that ask for the
    current time advance the counter. A helper that reads once therefore stays
    self-consistent; one that reads twice emits a `t=` and an `elapsed_s=` describing
    instants a day apart. Real `date` is invoked by absolute path so this cannot recurse.
    """
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    counter = tmp_path / "date-readings"
    fake = fakebin / "date"
    fake.write_text(
        "#!/bin/bash\n"
        'for arg in "$@"; do\n'
        '    if [ "$arg" = "-d" ]; then exec /usr/bin/date "$@"; fi\n'
        "done\n"
        f'n=$(cat "{counter}" 2>/dev/null || echo 0)\n'
        f'echo $((n + 1)) > "{counter}"\n'
        f'exec /usr/bin/date -d "@$(({int(_FAKE_START_EPOCH)} + n * {_FAKE_SKEW_S}))" "$@"\n'
    )
    fake.chmod(0o755)
    return f'export PATH="{fakebin}:$PATH"\n'


def _run_shell_emitter(path: Path, stage_name: str, *, skew_dir: Path | None = None) -> str:
    """Execute a container's real shell `stage` function and return its output.

    Args:
        skew_dir: When given, a scratch dir in which to build the skewing `date` from
            `_skewing_date_on_path`. The start epoch is moved onto that fake clock so
            `elapsed_s` stays small and the two fields remain comparable.
    """
    snippet = _extract(_SH_STAGE_RE, path, "stage() function")
    if skew_dir is not None:
        preamble = _skewing_date_on_path(skew_dir)
        start_epoch = _FAKE_START_EPOCH
    else:
        preamble = ""
        start_epoch = 1000.0
    program = (
        f'export CONTAINER_START_EPOCH="{start_epoch}"\n{preamble}{snippet}\nstage {stage_name}\n'
    )
    result = subprocess.run(
        ["bash", "-c", program],
        capture_output=True,
        text=True,
        timeout=30,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, f"{path} stage() failed: {result.stderr}"
    return result.stdout.strip()


class TestPythonEmitters:
    @pytest.mark.parametrize("image", sorted(_PYTHON_EMITTERS), ids=sorted(_PYTHON_EMITTERS))
    def test_emits_a_line_the_shared_parser_accepts(self, image: str) -> None:
        line = _run_python_emitter(_PYTHON_EMITTERS[image], Stage.WEIGHTS_READY)
        marker = parse_stage_marker(line)
        assert marker is not None, f"{image} emitted an unparseable marker: {line!r}"
        assert marker.name == Stage.WEIGHTS_READY

    @pytest.mark.parametrize("image", sorted(_PYTHON_EMITTERS), ids=sorted(_PYTHON_EMITTERS))
    def test_measures_elapsed_from_the_inherited_container_start(self, image: str) -> None:
        # CONTAINER_START_EPOCH=1000.0 is decades in the past, so a helper that
        # ignored it and used process start would report ~0 instead.
        line = _run_python_emitter(_PYTHON_EMITTERS[image], Stage.READY)
        marker = parse_stage_marker(line)
        assert marker is not None
        assert marker.elapsed_s > 1_000_000, (
            f"{image} ignored CONTAINER_START_EPOCH; elapsed_s would exclude the shell "
            f"half of startup (got {marker.elapsed_s})"
        )

    def test_all_emitters_agree_byte_for_byte(self) -> None:
        # Four copies of one format. Compare with the varying fields masked out, so
        # this catches a structural divergence without being a clock race.
        shapes = set()
        for path in (*_PYTHON_EMITTERS.values(), *_SHELL_EMITTERS.values()):
            runner = _run_python_emitter if path.suffix == ".py" else _run_shell_emitter
            line = runner(path, Stage.READY)
            shapes.add(re.sub(r"[0-9]", "N", line))
        assert len(shapes) == 1, f"emitters disagree on marker shape: {shapes}"


class TestShellEmitters:
    @pytest.mark.parametrize("image", sorted(_SHELL_EMITTERS), ids=sorted(_SHELL_EMITTERS))
    def test_emits_a_line_the_shared_parser_accepts(self, image: str) -> None:
        line = _run_shell_emitter(_SHELL_EMITTERS[image], Stage.CONTAINER_START)
        marker = parse_stage_marker(line)
        assert marker is not None, f"{image} emitted an unparseable marker: {line!r}"
        assert marker.name == Stage.CONTAINER_START

    @pytest.mark.parametrize("image", sorted(_SHELL_EMITTERS), ids=sorted(_SHELL_EMITTERS))
    def test_exports_the_start_epoch_for_the_python_half(self, image: str) -> None:
        # Without `export`, the Python process would fall back to its own start time
        # and elapsed_s would silently exclude the S3 model sync.
        text = _SHELL_EMITTERS[image].read_text()
        assert re.search(r"^export CONTAINER_START_EPOCH=", text, re.MULTILINE), (
            f"{image}/entrypoint.sh must export CONTAINER_START_EPOCH so elapsed_s is "
            f"continuous across the shell -> Python boundary"
        )

    @pytest.mark.parametrize("image", sorted(_SHELL_EMITTERS), ids=sorted(_SHELL_EMITTERS))
    def test_timestamp_and_elapsed_come_from_one_clock_reading(
        self, image: str, tmp_path: Path
    ) -> None:
        # Two `date` calls can straddle a second boundary, emitting a timestamp that
        # disagrees with its own elapsed_s. Under a clock that jumps a day per read, a
        # single-read helper stays self-consistent and a two-read one cannot.
        line = _run_shell_emitter(_SHELL_EMITTERS[image], Stage.READY, skew_dir=tmp_path)
        marker = parse_stage_marker(line)
        assert marker is not None, f"{image} emitted an unparseable marker: {line!r}"

        # The two fields describe one instant, so `t=` must equal the start epoch plus
        # `elapsed_s=`. A second clock read breaks that by a day under the stand-in.
        expected = datetime.fromtimestamp(_FAKE_START_EPOCH + marker.elapsed_s, tz=UTC)
        assert abs((marker.at - expected).total_seconds()) < 1.0, (
            f"{image}/entrypoint.sh stage() reads the clock more than once: t= says "
            f"{marker.at.isoformat()} but elapsed_s={marker.elapsed_s} implies "
            f"{expected.isoformat()}. Both fields must derive from a single reading."
        )


class TestStageSequences:
    """Each image must emit a coherent sequence, not just a valid line."""

    @staticmethod
    def _emitted_stages(path: Path) -> list[str]:
        """Stage names a file emits, in source order."""
        if path.suffix == ".py":
            return re.findall(r'^\s*_stage\("([a-z_]+)"\)', path.read_text(), re.MULTILINE)
        return re.findall(r"^\s*stage ([a-z_]+)$", path.read_text(), re.MULTILINE)

    @pytest.mark.parametrize(
        ("image", "path"),
        [(k, v) for k, v in sorted({**_PYTHON_EMITTERS, **_SHELL_EMITTERS}.items())],
        ids=lambda v: v if isinstance(v, str) else "",
    )
    def test_only_emits_known_stage_names(self, image: str, path: Path) -> None:
        unknown = set(self._emitted_stages(path)) - {s.value for s in Stage}
        assert not unknown, f"{image} emits stages absent from shared.stages.Stage: {unknown}"

    @pytest.mark.parametrize(
        ("image", "expected"),
        [
            # The kokoro images bake weights into the image and load them in-process,
            # so they have no `weights_fetched` and no shell half.
            ("kokoro", ["framework_init", "weights_ready", "warmup_done", "ready"]),
            ("kokoro-cpu", ["framework_init", "weights_ready", "warmup_done", "ready"]),
            # chatterbox splits across its entrypoint (fetch) and proxy (load).
            ("chatterbox", ["framework_init", "weights_ready", "warmup_done", "ready"]),
        ],
        ids=["kokoro", "kokoro-cpu", "chatterbox"],
    )
    def test_python_half_emits_the_expected_sequence(self, image: str, expected: list[str]) -> None:
        assert self._emitted_stages(_PYTHON_EMITTERS[image]) == expected

    def test_chatterbox_shell_half_covers_start_and_fetch(self) -> None:
        assert self._emitted_stages(_SHELL_EMITTERS["chatterbox"]) == [
            "container_start",
            "weights_fetched",
        ]

    def test_vllm_emits_the_whole_sequence_from_shell(self) -> None:
        # vllm/streaming_proxy.py has no lifespan hook, so its entrypoint carries
        # every stage — the boundaries are its `until curl` health gates.
        assert self._emitted_stages(_SHELL_EMITTERS["vllm"]) == [
            "container_start",
            "weights_fetched",
            "framework_init",
            "weights_ready",
            "warmup_done",
            "ready",
        ]

    @pytest.mark.parametrize(
        ("image", "path"),
        [(k, v) for k, v in sorted({**_PYTHON_EMITTERS, **_SHELL_EMITTERS}.items())],
        ids=lambda v: v if isinstance(v, str) else "",
    )
    def test_no_stage_is_emitted_twice(self, image: str, path: Path) -> None:
        # A duplicate would make one stage's duration unattributable.
        stages = self._emitted_stages(path)
        assert len(stages) == len(set(stages)), f"{image} emits a duplicate stage: {stages}"

    @pytest.mark.parametrize(
        ("image", "path"),
        [(k, v) for k, v in sorted({**_PYTHON_EMITTERS, **_SHELL_EMITTERS}.items())],
        ids=lambda v: v if isinstance(v, str) else "",
    )
    def test_ready_is_emitted_last(self, image: str, path: Path) -> None:
        stages = self._emitted_stages(path)
        if Stage.READY in stages:
            assert stages[-1] == Stage.READY, f"{image} emits stages after `ready`: {stages}"
