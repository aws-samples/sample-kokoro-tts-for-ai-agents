# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Guards the startup-stage markers the container image emits.

`tts-bench ttotal` attributes scaling lag by parsing these lines out of CloudWatch,
so an emitter that drifts from `shared.stages` does not fail loudly — it produces an
empty report that looks like a container with nothing to say.

The container runtime deps are not installed in the lean infra dev env, so the
emitter cannot be imported here. Instead the Python helper is executed in a
subprocess with the surrounding module stripped away, running the *real* code
from the *real* file, so a change to the emitter is caught rather than a change
to a copy of it.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

from shared.stages import Stage, parse_stage_marker

_CONTAINERS = Path(__file__).resolve().parents[1] / "containers"

#: Every image that emits markers, and the file each emits them from.
_PYTHON_EMITTERS = {
    "kokoro": _CONTAINERS / "kokoro" / "serve.py",
}


def _fail_missing(what: str, path: Path) -> None:
    pytest.fail(
        f"Could not find the {what} in {path}. If it was renamed or restructured, "
        f"update the extraction in this test — do not delete the assertion, it is what "
        f"keeps `tts-bench ttotal` able to read this container's logs."
    )


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
    # from its module is what lets this test cover the image's deps (torch, kokoro)
    # being absent here. `str()` because a `Stage` member's repr is not a literal
    # the subprocess can parse.
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
        sorted(_PYTHON_EMITTERS.items()),
        ids=lambda v: v if isinstance(v, str) else "",
    )
    def test_only_emits_known_stage_names(self, image: str, path: Path) -> None:
        unknown = set(self._emitted_stages(path)) - {s.value for s in Stage}
        assert not unknown, f"{image} emits stages absent from shared.stages.Stage: {unknown}"

    @pytest.mark.parametrize(
        ("image", "expected"),
        [
            # kokoro bakes weights into the image and loads them in-process, so it
            # has no `weights_fetched` and no shell half.
            ("kokoro", ["framework_init", "weights_ready", "warmup_done", "ready"]),
        ],
        ids=["kokoro"],
    )
    def test_python_half_emits_the_expected_sequence(self, image: str, expected: list[str]) -> None:
        assert self._emitted_stages(_PYTHON_EMITTERS[image]) == expected

    @pytest.mark.parametrize(
        ("image", "path"),
        sorted(_PYTHON_EMITTERS.items()),
        ids=lambda v: v if isinstance(v, str) else "",
    )
    def test_no_stage_is_emitted_twice(self, image: str, path: Path) -> None:
        # A duplicate would make one stage's duration unattributable.
        stages = self._emitted_stages(path)
        assert len(stages) == len(set(stages)), f"{image} emits a duplicate stage: {stages}"

    @pytest.mark.parametrize(
        ("image", "path"),
        sorted(_PYTHON_EMITTERS.items()),
        ids=lambda v: v if isinstance(v, str) else "",
    )
    def test_ready_is_emitted_last(self, image: str, path: Path) -> None:
        stages = self._emitted_stages(path)
        if Stage.READY in stages:
            assert stages[-1] == Stage.READY, f"{image} emits stages after `ready`: {stages}"
