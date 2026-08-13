# 001: AgentCore voice agent with a bidi/batch TTS mode switch

## Context

The blog post "Streaming Low Latency Speech from AI Agents Using Kokoro TTS
on Amazon SageMaker" needed a working agent to demonstrate its core claim:
synthesizing audio sentence-by-sentence as an agent generates text beats
waiting for its full response. `packages/speech-infra` already deploys
Kokoro as a SageMaker bidirectional-streaming endpoint, and `packages/tts
-client` already has both TTS call shapes needed for the comparison
(`synthesize_bidi_stream` for incremental delivery, `synthesize_bidi` for
send-everything-at-once). What was missing was the agent itself, deployed
somewhere that could stream its output back to a caller, with a way to
measure the real latency difference between the two delivery strategies.

## Decision

Added `packages/agent-infra`: a Strands agent deployed to Amazon Bedrock
AgentCore Runtime (`packages/agent-infra/container/`), with CDK stacks
(`packages/agent-infra/src/agent_infra/stacks/`) to deploy it, and a local
demo script (`agent_infra/scripts/compare_latency_demo.py`) that measures
the difference for real.

- **`BedrockAgentCoreApp` + an async-generator `@app.entrypoint`**, not a
  hand-rolled Starlette server. Reading its source
  (`bedrock_agentcore.runtime.app`) confirmed it satisfies AgentCore
  Runtime's HTTP contract (`/invocations`, `/ping`, port 8080) entirely on
  its own, and that yielded dicts are serialized as `data: {json}\n\n` —
  real SSE, but without an `event:` line, unlike the SSE relay this repo
  built once and deleted from the Kokoro container. The event dicts'
  own `"type"` field is the discriminator instead.
- **One explicit mode switch, not two transports.** `mode: "bidi" | "batch"`
  on the request selects between feeding the agent's streaming text deltas
  straight into `TTSClient.synthesize_bidi_stream` (audio starts on
  sentence one) or collecting the full response first and calling
  `TTSClient.synthesize_bidi` once (the "wait for everything" baseline).
  Both paths emit the same event vocabulary
  (`agent_start`/`text_delta`/`audio_stream_start`/`audio_chunk`/
  `audio_stream_end`/`agent_end`/`error`), so the demo script measures a
  true apples-to-apples comparison against the same agent, same TTS
  endpoint, same prompt.
- **Two async/sync boundaries, bridged with worker threads and queues**
  (`container/bidi_bridge.py`): `BidiChunkStream` drives its own private
  event loop via blocking `run_until_complete()` calls, so it runs in a
  dedicated worker thread; the agent's own `stream_async()` output crosses
  back into that thread's synchronous `text_source` iterable via a second
  queue. Errors from either side are relayed as tagged queue items and
  re-raised on the consumer side, rather than left to propagate as
  unhandled task exceptions — an early version of this without tagging hung
  the whole request on any mid-stream failure.
- **`tts-client` is installed into the container straight from source**
  (`COPY packages/tts-client/ ./tts-client/` + `pip install ./tts-client`),
  matching this repo's existing container convention (plain `pip`,
  `requirements.txt`, not `uv`) — it isn't on PyPI, but its `pyproject.toml`
  already has a working PEP 517 build backend, so no separate wheel-build
  stage is needed.

## Consequences

- The container's base image is `python:3.12-slim`, not the repo's
  documented `>=3.11` floor: `tts-client`'s own dependency
  (`aws-sdk-sagemaker-runtime-http2`) only ships wheels for Python 3.12+,
  discovered by a real build failure under 3.11, not assumed in advance.
- `DockerImageAsset`'s build context is the repo root (needed for the
  `COPY packages/tts-client/` line), which means its `exclude` list has to
  actively keep out anything large or self-referential living there —
  `cdk.out` itself (which lives inside `packages/agent-infra/`, inside the
  same tree being copied) recursed into itself until excluded, and a
  harness state directory (`.claude/`) needed excluding for the same
  reason.
- Measured for real against live infrastructure (Bedrock
  `us.anthropic.claude-opus-5`, the deployed `speech-kokoro-82m` SageMaker
  endpoint): bidi mode's time-to-first-audio was 2.4x faster than batch
  mode locally, and 3.5x faster against the deployed AgentCore Runtime
  (the added network hop makes batch's "wait for everything" cost relatively
  worse, not better).
- `container/server.py`'s own error handling (an explicit in-band `"error"`
  event) is necessary, not redundant with `BedrockAgentCoreApp`'s built-in
  fallback: an exception that escapes the entrypoint entirely is still
  caught and streamed by the SDK, but with a different, un-typed shape
  (`error`/`error_type`/`message`, no `"type"` field), which would break
  every consumer expecting this project's event vocabulary.
