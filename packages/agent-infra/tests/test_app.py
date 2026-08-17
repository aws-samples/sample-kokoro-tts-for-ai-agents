# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Tests for app.py's module-level wiring.

``REPO_ROOT`` being off by one directory level (an actual bug caught by a
real ``cdk diff``, not by the mocked stack tests -- ``image_uri_override``
in those keeps ``DockerImageAsset`` from ever touching this path) is exactly
the class of error worth a regression test: it fails silently under any test
that stubs the image URI, and only surfaces when CDK actually tries to open
the Dockerfile on disk.
"""

from __future__ import annotations

from agent_infra.app import REPO_ROOT


class TestRepoRoot:
    def test_points_at_the_actual_repo_root(self) -> None:
        assert (REPO_ROOT / "packages" / "agent-infra" / "container" / "Dockerfile").is_file()
        assert (REPO_ROOT / "packages" / "tts-client" / "pyproject.toml").is_file()
        assert (REPO_ROOT / "pyproject.toml").is_file()
