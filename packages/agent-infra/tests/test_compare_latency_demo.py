"""Tests for the demo script's event-recording and target-dispatch logic.

``_run_mode_local``/``_run_mode_deployed`` are thin wrappers around httpx's
and boto3's own line-reconstructing stream iterators (well-tested library
behavior); what's worth testing directly is this module's own logic: which
event fields turn into which timing, and which target shape routes to which
transport.
"""

from __future__ import annotations

from agent_infra.scripts.compare_latency_demo import ModeTiming, _record_event, _run_mode


class TestRecordEvent:
    def test_first_text_delta_sets_ttft(self) -> None:
        timing = ModeTiming(mode="bidi")
        _record_event(timing, {"type": "text_delta", "seq": 0, "text": "hi"}, 10.0)
        _record_event(timing, {"type": "text_delta", "seq": 1, "text": "there"}, 20.0)
        assert timing.ttft_ms == 10.0

    def test_first_audio_chunk_sets_ttfa_and_counts_chunks(self) -> None:
        timing = ModeTiming(mode="bidi")
        _record_event(timing, {"type": "audio_chunk", "seq": 0}, 50.0)
        _record_event(timing, {"type": "audio_chunk", "seq": 1}, 60.0)
        assert timing.ttfa_ms == 50.0
        assert timing.chunk_count == 2

    def test_agent_end_sets_total(self) -> None:
        timing = ModeTiming(mode="batch")
        _record_event(timing, {"type": "agent_end"}, 100.0)
        assert timing.total_ms == 100.0

    def test_error_event_records_message(self) -> None:
        timing = ModeTiming(mode="bidi")
        _record_event(timing, {"type": "error", "message": "boom"}, 5.0)
        assert timing.error == "boom"


class TestRunModeDispatch:
    def test_arn_target_dispatches_to_deployed_path(self, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(
            "agent_infra.scripts.compare_latency_demo._run_mode_deployed",
            lambda *args: calls.append(("deployed", args)) or ModeTiming(mode="bidi"),
        )
        _run_mode("arn:aws:bedrock-agentcore:us-east-1:111111111111:runtime/foo", "p", "v", "bidi")
        assert calls[0][0] == "deployed"

    def test_url_target_dispatches_to_local_path(self, monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(
            "agent_infra.scripts.compare_latency_demo._run_mode_local",
            lambda *args: calls.append(("local", args)) or ModeTiming(mode="bidi"),
        )
        _run_mode("http://localhost:8080", "p", "v", "bidi")
        assert calls[0][0] == "local"
