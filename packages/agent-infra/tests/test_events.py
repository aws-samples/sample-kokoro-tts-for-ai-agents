# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Tests for the container's event dict factories.

No skip needed: this module is pure stdlib (``base64``), unlike
``server.py``/``strands_agent.py`` which need ``bedrock_agentcore``/
``strands`` not installed in this lean dev environment.
"""

from __future__ import annotations

import base64

import events


class TestEventFactories:
    def test_agent_start(self) -> None:
        assert events.agent_start("req-1", "bidi") == {
            "type": "agent_start",
            "request_id": "req-1",
            "mode": "bidi",
        }

    def test_text_delta(self) -> None:
        assert events.text_delta("req-1", 3, "hello") == {
            "type": "text_delta",
            "request_id": "req-1",
            "seq": 3,
            "text": "hello",
        }

    def test_audio_stream_start(self) -> None:
        assert events.audio_stream_start("req-1", "af_heart", 24000) == {
            "type": "audio_stream_start",
            "request_id": "req-1",
            "format": "pcm16",
            "voice": "af_heart",
            "sample_rate": 24000,
        }

    def test_audio_chunk_base64_round_trips(self) -> None:
        pcm = b"\x01\x02\x03\x04"
        event = events.audio_chunk("req-1", 0, pcm)
        assert event["type"] == "audio_chunk"
        assert base64.b64decode(event["data"]) == pcm

    def test_audio_stream_end_rounds_duration(self) -> None:
        event = events.audio_stream_end("req-1", 5, 1.23456)
        assert event["duration_s"] == 1.235

    def test_agent_end(self) -> None:
        assert events.agent_end("req-1") == {"type": "agent_end", "request_id": "req-1"}

    def test_error(self) -> None:
        assert events.error("req-1", "boom") == {
            "type": "error",
            "request_id": "req-1",
            "message": "boom",
        }
