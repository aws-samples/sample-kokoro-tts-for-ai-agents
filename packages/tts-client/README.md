# tts-client

Blocking client for the TTS synthesis backends this repo talks to: the two SageMaker wire
transports our own endpoints speak (`TTSClient`), and Amazon Polly (`PollyClient`) — a
different AWS service with its own failure modes, kept as a separate class rather than a mode
on `TTSClient`.

Make as many instances as you like. Both classes are safe to construct freely — no pool-size
knob, no shared-instance requirement.

## `TTSClient.synthesize()` — response-stream

One call, one complete answer. Chunked binary WAV/MP3 over `InvokeEndpointWithResponseStream`.

```python
from tts_client.client import TTSClient
from tts_client.types import SynthesisRequest

client = TTSClient(region="us-east-1")
request = SynthesisRequest(text="Hello there.", voice="af_heart")
result = client.synthesize("speech-kokoro-82m", request)

result.audio_bytes   # bytes
result.duration_s     # float
result.latency_ms     # float
```

## `TTSClient.synthesize_bidi()` — bidirectional, one full text

Same call shape, but over SageMaker's bidirectional-streaming (HTTP/2) transport instead of
response-stream. Always raw PCM — no format choice. This is a separate method rather than a
flag on `synthesize()` because the two are different wire protocols end to end (HTTP/1.1
chunked body vs. HTTP/2 bidirectional stream), not two configurations of the same call.

```python
result = client.synthesize_bidi("speech-kokoro-82m", request)
```

## `TTSClient.synthesize_bidi_stream()` — bidirectional, incremental text in

The bidi transport's other real use case: sending text to the model **as it becomes
available**, rather than waiting for the whole utterance to exist first. Useful in a
multi-model pipeline where an upstream LLM is generating a paragraph sentence-by-sentence —
each sentence can be sent to the TTS model the moment it's produced, instead of buffering the
full response first.

```python
with client.synthesize_bidi_stream("speech-kokoro-82m", "af_heart", text_source) as stream:
    for chunk in stream:
        chunk.audio_bytes  # raw PCM for this sentence
```

`text_source` can be a complete string (split into sentences and sent one at a time) or an
iterable of fragments arriving over time — e.g. an upstream LLM's own token stream, fed in as
each piece arrives.

One real constraint: this does not overlap synthesis across chunks. Every bidi container's
handler is strictly sequential (`receive -> synthesize & send audio -> synthesis_complete ->
receive`), so chunk N+1 isn't acted on until chunk N's audio has fully arrived. What it buys
is not waiting for the *entire* text to be assembled before sending anything — chunk 1 starts
synthesizing as soon as it exists, rather than after the last chunk does. That's still a real
latency win for a producer/consumer pipeline, just not concurrent inference on multiple
chunks at once.

## `PollyClient.synthesize()` — Amazon Polly

A different AWS service, not a SageMaker endpoint — no instance to size, no queue to
saturate, so none of `TTSClient`'s pooling/retry/error-taxonomy design applies here. Kept as
its own class for that reason. Requires the `tts-client[polly]` extra (pulls in `librosa`,
used only to measure the duration of the MP3 Polly returns, since Polly's response carries no
duration field of its own) — a caller who only uses `TTSClient` never pays for it.

```python
from tts_client.polly import PollyClient

polly = PollyClient(region="us-east-1")
result = polly.synthesize(voice_id="Joanna", engine="neural", text="Hello there.")
```

## Errors

Every exception raised by `TTSClient` is a `TTSClientError` subclass (`tts_client.errors`) —
`QueueSaturatedError`, `RequestStaleError`, `ThrottledError`, `ServerError`, `ModelError`,
`TTSTimeoutError` — each carrying `http_status`, the container's real status once SageMaker's
own wrapping is unwrapped. Catch the base class for "something went wrong," or a specific
subclass to react to one failure mode (e.g. retry on `QueueSaturatedError`, not on
`RequestStaleError`).

## What this doesn't cover

- The wire contract `TTSClient.synthesize()` actually speaks to (`speech-kokoro-82m`'s
  request/response fields, why MP3, segmentation behavior) — see
  [`docs/models/kokoro-streaming-api.md`](../../docs/models/kokoro-streaming-api.md).
- Why pooling/threading works the way it does (one boto3 client is safe to share; the bidi
  transport builds a fresh HTTP/2 client per call) — see the docstring at the top of
  [`tts_client/client.py`](tts_client/client.py).
