# Kokoro-82M Streaming API

Wire contract for `speech-kokoro-82m`.

Implementation: `packages/speech-infra/containers/kokoro/serve.py`.
Client: `TTSClient.synthesize()` in `packages/tts-client/tts_client/client.py`.

All numbers below are measured against the live endpoint, not estimated.

## Request

`POST /invocations`, `ContentType: application/json`.

| Field | Values | Default | Effect |
|-------|--------|---------|--------|
| `text` | string | required | 400 if empty |
| `voice` | one of 20 (see below) | `af_heart` | Kokoro voice id |
| `speed` | float | `1.0` | Playback rate |
| `stream` | bool | `true` | `false` returns one complete response |
| `format` | `wav` \| `mp3` | `wav` | Raw PCM frames, or 48 kbps mono MP3 |
| `sample_rate` | `8000` \| `16000` \| `22050` \| `24000` | `24000` | Output rate; see below |

Every field defaults to the pre-existing behaviour, so callers that send only
`text`/`voice` are unaffected. That matters: `TTSClient.synthesize()`
sends no `format` and asserts `RIFF` on the response, and the eval baseline
depends on it. An invalid `format`, `voice`, or `sample_rate` returns 400.

### Voice

The container loads one `KPipeline`, for `lang_code="a"` (American English)
only — see `_load_pipeline()`. Kokoro's model supports 54 voices across 9
languages, but only the 20 `af_*`/`am_*` voices actually work against this
deployment; the rest were never a validation gap so much as an unenforced
one — a wrong-language voice would either 404 deep in `kokoro`'s own
`hf_hub_download`, or silently load a wrong-language style vector into this
English-only phonemization pipeline with no error at all. `_VALID_VOICES`
enforces the boundary explicitly now. The same 20-voice enum is available
client-side as `tts_eval.synthesize.KokoroVoice`.

### Sample rate

`24000` (Kokoro's native rate) is the ceiling, not one option among several:
producing a higher rate from a 24kHz source would be pure interpolation with
no added fidelity, so nothing above it is offered. Every other value is a
real downsample, via `soxr` — proven in `scratch/soxr_resample_probe/` before
this shipped: the buffered (`stream: false`) path resamples the whole
utterance in one `soxr.resample()` call, and the two streaming paths
(`stream: true`, and the WebSocket path) use `soxr.ResampleStream`, which
carries filter state across segments — proven necessary, not just nicer,
since independent per-segment `soxr.resample()` calls measurably degrade at
segment boundaries.

Requesting a downsample adds real resampling latency over the native rate.
Client-side, `tts_client.types.SampleRate` mirrors `SUPPORTED_SAMPLE_RATES`
as an enum, so an unsupported value is rejected at request-construction
time, before any network call.

| `format` | Response `ContentType` |
|----------|------------------------|
| `wav` | `audio/wav` (WAV header + PCM) |
| `mp3` | `audio/mpeg` (bare MP3 frames) |

On the streaming WAV path the leading header carries placeholder sizes
(`0xFFFFFFFF`), since the total length is unknown when it is sent. Only the
`stream: false` path emits real RIFF/data sizes. Players tolerate this; anything
computing duration from the header will not.

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
with both `new Audio()` and MSE. That framing simplicity is worth 16% for a
consumer that must forward chunks it does not parse.

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
no MP3, by design. Playback here is one-way (no barge-in). Each message
accepts `voice`/`sample_rate` with the same validation and resampling as
`/invocations` — an unrecognized value gets an `error` frame instead of a
400, since there is no HTTP status code on this transport.

The bidirectional-streaming contract itself:

- Implement a WebSocket endpoint at `ws://localhost:8080/invocations-bidirectional-stream`.
- Docker label: `com.amazonaws.sagemaker.capabilities.bidirectional-streaming=true`.
- Client side uses the `InvokeEndpointWithBidirectionalStream` API — see
  `packages/tts-client/tts_client/client.py`'s `synthesize_bidi()`.
- SageMaker does **not** batch across WebSocket sessions — each client gets its own. Batching
  across concurrent sessions, if needed, is the container's own job.

## Verification

| Script | Covers |
|--------|--------|
| `packages/speech-infra/tests/test_kokoro_serve.py` | Server: flush tail, encoder reuse, default-path regression, voice/sample_rate validation, resampled WAV/MP3/WebSocket output |
| `packages/tts-client/tests/test_client.py`, `test_streaming.py` | Client: streaming WAV baseline, sample_rate request/response threading on both transports |
| `packages/tts-eval/tests/test_synthesize.py` | `KokoroVoice` catalog, `validate_kokoro_voice` accept/reject cases |

Live results on 4 verified clips: WER 0.000, default no-flag call still returns
96044 B of `RIFF`.
