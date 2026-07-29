# Kokoro-82M Streaming API

Wire contract for `speech-kokoro-82m`. Written for the AgentCore relay, which sits
between SageMaker and the browser (`SageMaker -> AgentCore -> Browser via SSE`).

Implementation: `packages/speech-infra/containers/kokoro/serve.py`.
Client: `SynthesisClient.synthesize_sse()` in `packages/tts-eval/src/tts_eval/synthesize.py`.

All numbers below are measured against the live endpoint, not estimated.

## Request

`POST /invocations`, `ContentType: application/json`.

| Field | Values | Default | Effect |
|-------|--------|---------|--------|
| `text` | string | required | 400 if empty |
| `voice` | string | `af_heart` | Kokoro voice id |
| `speed` | float | `1.0` | Playback rate |
| `stream` | bool | `true` | `false` returns one complete response. Ignored when `transport` is `sse` |
| `transport` | `binary` \| `sse` | `binary` | Raw chunked bytes, or `text/event-stream` |
| `format` | `wav` \| `mp3` | `wav` | Raw PCM frames, or 48 kbps mono MP3 |
| `request_id` | string | `"unknown"` | Echoed in every SSE event |

Every field defaults to the pre-existing behaviour, so callers that send only
`text`/`voice` are unaffected. That matters: `SynthesisClient.synthesize_stream()`
sends no `transport`/`format` and asserts `RIFF` on the response, and the eval
baseline depends on it. An invalid `transport` or `format` returns 400.

Transport is selected by body field rather than the `Accept` header because
`Accept` passthrough behaviour through SageMaker is unverified, while
`body.get(...)` is already proven on this endpoint.

| `transport` | `format` | Response `ContentType` |
|-------------|----------|------------------------|
| `binary` | `wav` | `audio/wav` (WAV header + PCM) |
| `binary` | `mp3` | `audio/mpeg` (bare MP3 frames) |
| `sse` | `mp3` | `text/event-stream` (base64 MP3 in JSON) |
| `sse` | `wav` | `text/event-stream` (base64 WAV: placeholder header in the first chunk, then PCM) |

On the streaming WAV paths the leading header carries placeholder sizes
(`0xFFFFFFFF`), since the total length is unknown when it is sent. Only the
`stream: false` path emits real RIFF/data sizes. Players tolerate this; anything
computing duration from the header will not.

## SSE Event Contract

Deliberately aligned with the Polly contract in `polly-tts-hld.md`, substituting
`format: "mp3"` for `"ogg_vorbis"`, so a relay can handle both identically.

| Event | Payload |
|-------|---------|
| `audio_stream_start` | `{ request_id, format, voice, sample_rate }` |
| `audio_chunk` | `{ request_id, seq, data: "<base64>" }` |
| `audio_stream_end` | `{ request_id, total_chunks, duration_s }` |
| `error` | `{ request_id, message }` |

`audio_stream_start` is emitted **before** inference begins, so the client can
build its decoder while the model runs. It is not an audio-timing signal —
measure time-to-first-audio on the first `audio_chunk`, or you will report ~15 ms
instead of the real wait.

`error` exists because a mid-stream failure on the binary path just truncates the
chunked body and reaches the caller as an opaque `ModelStreamError`. On SSE the
failure arrives in-band with a message.

## Consumers Must Buffer Across Delivery Units

**SSE frames do not align with SageMaker's `PayloadPart` boundaries.** A consumer
that parses each part as one event will fail on valid traffic.

Measured (`scratch/check_sse_framing.py`), a 4-frame response arrived as 6 parts:

| part | bytes | ends on frame boundary |
|------|-------|------------------------|
| 0 | 119 | yes |
| 1 | 4090 | **no** |
| 2 | 8192 | **no** |
| 3 | 3180 | yes |
| 4 | 1202 | yes |
| 5 | 95 | yes |

Two of six parts ended mid-frame. Accumulate into a buffer, split on the blank
line, and retain the remainder:

```python
buffer = b""
for event in resp["Body"]:
    if "PayloadPart" not in event:
        continue
    buffer += event["PayloadPart"]["Bytes"]
    while b"\n\n" in buffer:
        frame, buffer = buffer.split(b"\n\n", 1)
        handle(frame)
```

`synthesize_sse()` does this; `test_reassembles_frames_split_across_payload_parts`
locks it in, and reverting to per-part parsing fails that test.

## Why MP3

Compared over 24 clips x 4 candidates = 96 AWS Transcribe round-trips
(`scratch/codec_shootout.py`):

| Candidate | Bytes | Base64 | Bitrate | WER |
|-----------|-------|--------|---------|-----|
| Raw PCM | 2,396,400 | 3,195,200 | 384 kbps | 0.0000 |
| MP3 48k | 308,304 | 411,072 | 49.5 kbps | 0.0000 |
| AAC 32k fMP4 | 258,039 | 344,084 | 41.6 kbps | 0.0000 |

**Quality did not decide it — all four scored WER 0.0000.** AAC is 16% smaller
than MP3, but needs fragmented MP4: a 729-byte `ftyp+moov` init segment, then
`moof`/`mdat` pairs, and it only plays via MediaSource Extensions. MP3 is a bare
frame stream — any prefix is decodable, concatenation is trivial, and it works
with both `new Audio()` and MSE. That framing simplicity is worth 16% through a
relay that must forward chunks it does not parse.

Progressive decode is verified: prefixes at 10/25/50/75/100% of the byte stream
decode to 0.206/0.516/1.032/1.548/2.064 s of audio.

## Why lameenc, Not ffmpeg

`lameenc` (in-process LAME bindings) replaced an ffmpeg subprocess pipe.

| | ffmpeg pipe | lameenc |
|---|---|---|
| First output after | 1.5-2.1 s of audio | ~0.5 s |
| Per-call cost | subprocess + pipe | ~3 ms |
| Image dependency | ffmpeg binary | 242 KB wheel |

ffmpeg buffers 1.5-2.1 s of input before emitting a byte, which defeats
progressive playback for short replies. Input-buffering flags only reached ~1.5 s,
and `-fflags nobuffer` **silently discarded 70% of the audio** (16992 B -> 5040 B)
while appearing to improve latency.

Three `lameenc` constraints the server code depends on:

- `flush()` output carries **real audio**, not just padding. For utterances short
  enough that `encode()` never returns anything, it carries *all* the audio. It
  must always be sent before ending the stream.
- `encode()` returning `b""` is normal — LAME is still filling its frame buffer.
- An encoder is single-use: `encode()` after `flush()` raises. One instance per
  request.

## Segmentation

`serve.py` passes no `split_pattern` to `KPipeline`, so KPipeline's default
`r"\n+"` applies. **Sentence-punctuated prose arrives as one segment**, subdivided
further only when KPipeline hits its own internal token limit.

This differs from the sibling containers, which both split on sentence
punctuation: `chatterbox/streaming_proxy.py:41` and `vllm/streaming_proxy.py:88`
share `_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")`.

Measured impact (`scratch/check_segment_granularity.py`,
`scratch/check_split_cost_live.py`):

| 7 sentences | separator `". "` | separator `"\n"` |
|-------------|------------------|------------------|
| segments | 3 | 8 |
| first audio | 503 ms | **143 ms** |

| 4 sentences | current | per-sentence |
|-------------|---------|--------------|
| first audio | 361 ms | **106 ms** (71% faster) |
| duration | 16.43 s | 16.90 s (+2.9%) |
| WER | 0.0233 | 0.0233 (identical transcript) |

So `split_pattern=r"[.!?]+\s+|\n+"` would cut first-audio latency ~71% for +2.9%
duration (inter-sentence pauses) at no intelligibility cost. **Not applied** — it
would move the default path the eval harness benchmarks, making prior TTFAB/RTF
numbers non-comparable. Recorded here as an available optimization.

Until then, a caller wanting per-sentence granularity can simply send
newline-separated text.

## WebSocket

`ws://localhost:8080/invocations-bidirectional-stream` streams **raw PCM only** —
no SSE, no MP3, by design. See the bidirectional contract in
`tts-architecture-reference.md`. Playback here is one-way (no barge-in), so the
SSE path covers the AgentCore use case and the WebSocket path was left alone.

## Verification

| Script | Covers |
|--------|--------|
| `packages/speech-infra/tests/test_kokoro_serve.py` | Server: 13 tests, incl. flush tail, encoder reuse, default-path regression |
| `packages/tts-eval/tests/test_synthesize.py` | Client: split-frame reassembly, ttfab timing, error event |
| `scratch/verify_sse_live.py` | Live: SSE + WER, progressive decode, binary MP3, default WAV |
| `scratch/verify_sse_client.py` | Live: `synthesize_sse()` end-to-end with WER |
| `scratch/check_sse_framing.py` | The `PayloadPart` split-frame measurement |

Live results on 4 verified clips: WER 0.000, 2 chunks each, ttfab 77-78 ms,
default no-flag call still returns 96044 B of `RIFF`.
