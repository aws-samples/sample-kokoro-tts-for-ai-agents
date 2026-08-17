# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Kokoro-82M TTS serving (PyTorch GPU).

Provides:
- GET /ping: health check
- POST /invocations: streaming TTS inference (per-sentence chunks)
- WS /invocations-bidirectional-stream: streaming TTS over WebSocket

Uses the PyTorch `kokoro` package with KPipeline for native CUDA
inference. Deployed today on A10G (ml.g5, sm_86) off the pinned cu126 wheel
(see the Dockerfile for why cu126, not cu124) — this logs the GPU it
actually found rather than assuming one. Single model instance with
asyncio.Lock serialization; the model is fast enough (0.12s/inference on
A10G) that multi-session adds negligible benefit.

/invocations selects its wire shape from body fields, defaulting to today's
behaviour so existing callers are unaffected:
- format: "wav" (raw PCM frames) | "mp3" (48 kbps mono)
- sample_rate: one of SUPPORTED_SAMPLE_RATES (default: SAMPLE_RATE, no resampling)
- voice: one of _VALID_VOICES (default: DEFAULT_VOICE)

Every prefix of an MP3 frame stream is independently decodable, which is what
lets the client start playing before synthesis finishes.

sample_rate is a downsample-only knob: SAMPLE_RATE (24000, Kokoro's native
rate) is the ceiling, not one option among several -- producing 32000+ from a
24000 source would be pure interpolation with no added fidelity, so nothing
above SAMPLE_RATE is offered. The buffered (non-streaming) path resamples the
whole utterance in one soxr.resample() call; the two streaming paths use
soxr.ResampleStream (stateful across segments) rather than independent
per-segment calls -- proven in scratch/soxr_resample_probe/ that independent
calls measurably degrade at segment boundaries versus the stateful streaming
resampler, which reproduces the whole-buffer result exactly.

voice is validated against _VALID_VOICES because the pipeline below is loaded
once, for lang_code="a" (American English) only: a voice from any other
language would either 404 deep in kokoro's own hf_hub_download, or -- worse
-- silently load a wrong-language style vector into this English-only
phonemization pipeline with no error at all.
"""

import asyncio
import json
import logging
import os
import struct
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import lameenc
import numpy as np
import soxr
import torch
import uvicorn
from kokoro import KPipeline
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

MAX_REQUEST_AGE_S = float(os.environ.get("MAX_REQUEST_AGE_S", "56"))
MAX_QUEUE_DEPTH = int(os.environ.get("MAX_QUEUE_DEPTH", "0"))
SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"
WARMUP_TEXT = os.environ.get("WARMUP_TEXT", "Warming up.")
MP3_BITRATE_KBPS = 48
MP3_QUALITY = 2

FORMAT_WAV = "wav"
FORMAT_MP3 = "mp3"

_MEDIA_TYPES = {FORMAT_WAV: "audio/wav", FORMAT_MP3: "audio/mpeg"}

#: SAMPLE_RATE (native) is the ceiling, not a peer option -- see the module
#: docstring for why upsampling isn't offered.
SUPPORTED_SAMPLE_RATES = frozenset({8000, 16000, 22050, SAMPLE_RATE})

#: The 20 voices that actually work against the single lang_code="a" pipeline
#: loaded below. Duplicated rather than imported from tts_eval.synthesize.
#: KokoroVoice -- this container has no access to sibling monorepo packages
#: at Docker build time (see tts_client/streaming.py's _SENTENCE_RE comment
#: for the same convention elsewhere in this codebase). Keep the two lists in
#: sync by hand if either changes.
_VALID_VOICES = frozenset(
    {
        "af_heart",
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
        "am_santa",
    }
)

_inflight: int = 0

_logger = logging.getLogger("kokoro_serve")

_pipeline: KPipeline | None = None
_inference_lock: asyncio.Lock | None = None

#: Container start, for `elapsed_s` on the stage markers below. Taken from the
#: entrypoint via `CONTAINER_START_EPOCH` where one exists; this container is
#: started directly by `CMD`, so process start is the earliest point observable
#: from inside. Image pull is bounded externally by the log stream's first event.
_STAGE_EPOCH = float(os.environ.get("CONTAINER_START_EPOCH") or time.time())


def _stage(name: str) -> None:
    """Emit a startup-stage marker, parsed by `tts-bench ttotal`.

    T_total is the scaling lag the whole capacity plan is most sensitive to, and
    it is only actionable when attributed to a stage. The format is
    byte-identical across all four containers so a single parser reads them all;
    `time.strftime` rather than `datetime.UTC` because this image and
    chatterbox's are ubuntu22.04-based (Python 3.10, no `datetime.UTC`), and
    the emitters must not diverge between containers.

    `elapsed_s` runs from container start, so a log stream whose earlier lines
    aged out is still partially usable.
    """
    now = time.time()
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now))
    print(
        f"=== STAGE {name} t={stamp}.{int(now % 1 * 1000):03d}Z "
        f"elapsed_s={now - _STAGE_EPOCH:.3f} ===",
        flush=True,
    )


def _load_pipeline() -> KPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(
            f"[kokoro] Pipeline loaded, device={device}, CUDA={torch.cuda.is_available()}",
            flush=True,
        )
        if torch.cuda.is_available():
            print(f"[kokoro] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    return _pipeline


def _synthesize_full(text: str, voice: str, speed: float) -> np.ndarray:
    pipeline = _load_pipeline()
    chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        if audio is not None:
            chunks.append(audio)
    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks)


def _samples_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    pcm = (samples * 32767).astype(np.int16).tobytes()
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        data_size,
    )
    return header + pcm


def _warmup() -> int:
    """Run one discarded synthesis so the first real request does not pay for JIT.

    The pipeline being loaded is not the same as it being ready: CUDA kernel
    autotune happens on first inference, so without this the first caller after a
    scale-out absorbs it. T_total should measure time-to-serving-*good*-traffic,
    which makes paying that cost here — before uvicorn accepts connections — the
    right trade.

    Returns:
        Samples generated, for the log line. Zero is not fatal: a container that
        cannot warm up can still serve, and failing startup over it would turn a
        latency problem into an outage.
    """
    samples = _synthesize_full(WARMUP_TEXT, DEFAULT_VOICE, 1.0)
    return int(samples.size)


@asynccontextmanager
async def _lifespan(app: Starlette) -> AsyncGenerator[None, None]:
    global _inference_lock
    _inference_lock = asyncio.Lock()

    _stage("framework_init")
    _load_pipeline()
    _stage("weights_ready")

    try:
        count = _warmup()
        print(f"[kokoro] Warm-up complete, {count} samples discarded", flush=True)
    except Exception:
        # Logged and swallowed: see _warmup. The marker is still emitted so
        # ttotal's stage sequence stays complete and the warm-up cost is visible
        # even when the warm-up itself failed.
        _logger.exception("Warm-up inference failed; serving anyway")
    _stage("warmup_done")

    _stage("ready")
    yield


async def ping(request: Request) -> Response:
    return Response(content="OK", status_code=200)


def _wav_header_placeholder(sample_rate: int = SAMPLE_RATE) -> bytes:
    """44-byte WAV header with 0xFFFFFFFF data size for streaming."""
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        0xFFFFFFFF,
    )


def _to_numpy(audio: object) -> np.ndarray:
    """Convert a KPipeline audio segment (torch.Tensor) to a float32 numpy array."""
    if hasattr(audio, "detach"):
        return audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32)


def _samples_to_pcm(samples: np.ndarray) -> bytes:
    return (samples * 32767).astype(np.int16).tobytes()


def _resample_buffer(samples: np.ndarray, target_rate: int) -> np.ndarray:
    """Resample one complete buffer. No-op at the native rate.

    For the buffered (non-streaming) path only -- one call over the whole
    utterance, no segment-boundary concern. See _SegmentResampler for the
    streaming paths, which see one short segment at a time.
    """
    if target_rate == SAMPLE_RATE:
        return samples
    return soxr.resample(samples, SAMPLE_RATE, target_rate)


class _SegmentResampler:
    """Resamples a sequence of independent segments as one continuous signal.

    Wraps soxr.ResampleStream, which carries filter state across
    resample_chunk() calls. Proven in scratch/soxr_resample_probe/: resampling
    each KPipeline segment with an independent, stateless soxr.resample()
    call measurably degrades at segment boundaries (max sample-to-sample jump
    0.0904 vs. a clean 0.0867 baseline); this class reproduces the
    whole-buffer result exactly (0.0867), because the filter never resets
    between segments.

    A no-op passthrough at the native rate, so the default (no resampling
    requested) path allocates nothing extra and takes the exact code path it
    did before this class existed.

    The segment-fetch loops in this file only learn a segment was the last
    one after already yielding it (_next_segment returns None on the
    following call, not a flag on the current one). Restructuring those
    loops to look ahead by one segment would be more invasive than needed:
    every real segment goes through with last=False, and one trailing
    resample_chunk(<empty array>, last=True) call after the loop ends flushes
    the tail. Verified in scratch/soxr_resample_probe/: this produces output
    bit-for-bit identical to resampling the whole buffer in one call.
    """

    def __init__(self, target_rate: int) -> None:
        self._passthrough = target_rate == SAMPLE_RATE
        self._stream = (
            None
            if self._passthrough
            else soxr.ResampleStream(SAMPLE_RATE, target_rate, num_channels=1, dtype="float32")
        )

    def push(self, samples: np.ndarray) -> np.ndarray:
        """Resample one segment. Call flush() once, after the last push()."""
        if self._passthrough:
            return samples
        assert self._stream is not None
        return self._stream.resample_chunk(samples, last=False)

    def flush(self) -> np.ndarray:
        """Drain the resampler's remaining buffered output. Call exactly once."""
        if self._passthrough:
            return np.array([], dtype=np.float32)
        assert self._stream is not None
        return self._stream.resample_chunk(np.array([], dtype=np.float32), last=True)


class Mp3StreamEncoder:
    """Incremental MP3 encoder for one request.

    lameenc is used instead of an ffmpeg subprocess because ffmpeg buffers
    ~1.5-2.1s of audio before emitting its first byte, which would defeat
    progressive playback for short replies. This emits after ~0.5s of audio at
    roughly 3ms per call.

    A single instance is valid for exactly one utterance: lameenc raises
    "Encoder not initialised" if encode() is called after flush().
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self._encoder = lameenc.Encoder()
        self._encoder.set_bit_rate(MP3_BITRATE_KBPS)
        self._encoder.set_in_sample_rate(sample_rate)
        self._encoder.set_channels(1)
        self._encoder.set_quality(MP3_QUALITY)
        self._encoder.silence()
        self._closed = False

    def encode(self, samples: np.ndarray) -> bytes:
        """Encode one segment. Returns b"" when LAME is still filling its frame buffer."""
        if self._closed:
            raise RuntimeError("Mp3StreamEncoder already flushed")
        return bytes(self._encoder.encode(_samples_to_pcm(samples)))

    def flush(self) -> bytes:
        """Emit LAME's remaining frames.

        This tail carries real audio, not just padding, and for utterances short
        enough that encode() never returned anything it carries *all* of the
        audio. Callers must always send it before ending the stream.
        """
        if self._closed:
            return b""
        self._closed = True
        return bytes(self._encoder.flush())


def _generate_sentences(
    text: str, voice: str, speed: float, sample_rate: int = SAMPLE_RATE
) -> list[np.ndarray]:
    """Generate all sentence audio chunks, resampled to sample_rate (runs in executor).

    Resampling happens here, inside the executor call, rather than in
    bidirectional_stream's async send loop -- this function already exists
    specifically to keep CPU-bound work off the event loop, and soxr's work
    is exactly that. One _SegmentResampler for the whole call, same as the
    HTTP streaming path, so the sequence of segments resamples as one
    continuous signal rather than independently per segment.
    """
    pipeline = _load_pipeline()
    resampler = _SegmentResampler(sample_rate)
    results = []
    for _, _, audio in pipeline(text, voice=voice, speed=speed):
        if audio is not None:
            resampled = resampler.push(_to_numpy(audio))
            if len(resampled) > 0:
                results.append(resampled)
    tail = resampler.flush()
    if len(tail) > 0:
        results.append(tail)
    return results


def _next_segment(segments: object) -> np.ndarray | None:
    """Advance the KPipeline generator to the next audio segment (runs in executor).

    Returns the segment as float32 numpy, or None once the generator is exhausted.
    The passed iterator resumes where it left off, so repeated calls walk the
    segments one at a time - enabling true incremental streaming.
    """
    for _, _, audio in segments:
        if audio is not None:
            return _to_numpy(audio)
    return None


async def _stream_sentences_generator(
    text: str,
    voice: str,
    speed: float,
    audio_format: str = FORMAT_WAV,
    sample_rate: int = SAMPLE_RATE,
    stats: dict | None = None,
) -> AsyncGenerator[bytes, None]:
    """Yield encoded audio chunks per KPipeline segment as they are produced.

    For WAV this is a placeholder header followed by raw PCM; for MP3 it is a
    bare frame stream, so any prefix the client has received is playable.

    Segment size is KPipeline's, not ours: no split_pattern is passed, so its
    default r"\\n+" applies and ". "-separated prose arrives as ONE segment. A
    measured 7-sentence paragraph produced first audio at 503ms in 3 segments;
    the same sentences newline-separated gave 143ms in 8. Passing a
    sentence-aware split_pattern would cut first-audio latency ~70% at +3%
    duration from inter-sentence pauses and no WER change, but it moves the
    default path the eval harness measures, so it is deliberately not done here.

    The inference lock is held for the whole stream: KPipeline is a single stateful
    instance, so two interleaved segment generators would corrupt each other. The
    model is fast enough that serializing whole requests costs nothing meaningful.

    Each segment is resampled through one _SegmentResampler for the whole
    request, not an independent call per segment -- see that class's
    docstring for why (measurable degradation at segment boundaries
    otherwise). sample_count and header_sent are tracked on the resampled
    (output) stream, since that's what's actually sent over the wire.
    """
    header_sent = False
    encoder = Mp3StreamEncoder(sample_rate=sample_rate) if audio_format == FORMAT_MP3 else None
    resampler = _SegmentResampler(sample_rate)
    sample_count = 0

    def _encode(audio: np.ndarray) -> bytes | None:
        nonlocal header_sent
        if encoder is None:
            pcm = _samples_to_pcm(audio)
            if not header_sent:
                header_sent = True
                return _wav_header_placeholder(sample_rate) + pcm
            return pcm
        # Empty output is normal: LAME is still filling its frame buffer.
        chunk = encoder.encode(audio)
        return chunk or None

    loop = asyncio.get_event_loop()
    pipeline = _load_pipeline()

    async with _inference_lock:
        segments = iter(pipeline(text, voice=voice, speed=speed))
        while True:
            audio = await loop.run_in_executor(None, _next_segment, segments)
            if audio is None:
                break
            audio = resampler.push(audio)
            sample_count += len(audio)
            if len(audio) > 0:
                chunk = _encode(audio)
                if chunk:
                    yield chunk

        tail_samples = resampler.flush()
        if len(tail_samples) > 0:
            sample_count += len(tail_samples)
            chunk = _encode(tail_samples)
            if chunk:
                yield chunk

        if encoder is not None:
            tail = encoder.flush()
            if tail:
                yield tail

    if stats is not None:
        stats["samples"] = sample_count


def _samples_to_mp3(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    encoder = Mp3StreamEncoder(sample_rate=sample_rate)
    return encoder.encode(samples) + encoder.flush()


async def _invocations_sync(
    text: str,
    voice: str,
    speed: float,
    audio_format: str = FORMAT_WAV,
    sample_rate: int = SAMPLE_RATE,
) -> Response:
    """Original synchronous path: complete audio file in one response."""
    t0 = time.perf_counter()

    async with _inference_lock:
        samples = await asyncio.get_event_loop().run_in_executor(
            None, _synthesize_full, text, voice, speed
        )

    samples = _resample_buffer(samples, sample_rate)

    if audio_format == FORMAT_MP3:
        content = _samples_to_mp3(samples, sample_rate)
    else:
        content = _samples_to_wav(samples, sample_rate)
    elapsed = time.perf_counter() - t0
    audio_duration = len(samples) / sample_rate

    return Response(
        content=content,
        media_type=_MEDIA_TYPES[audio_format],
        headers={
            "X-Audio-Duration": f"{audio_duration:.3f}",
            "X-Inference-Time-Ms": str(int(elapsed * 1000)),
            "X-RTF": f"{elapsed / audio_duration:.3f}" if audio_duration > 0 else "0",
            "X-Characters-Processed": str(len(text)),
        },
    )


async def invocations(request: Request) -> Response:
    """TTS: accept text, return audio (chunked WAV by default)."""
    global _inflight

    # Admission gate: checked and incremented before any await so the asyncio
    # event loop cannot interleave another request between the check and the
    # increment. Streaming responses decrement via _inflight_wrap; the sync
    # path decrements in a try/finally.
    if MAX_QUEUE_DEPTH > 0 and _inflight >= MAX_QUEUE_DEPTH:
        return JSONResponse(
            status_code=503,
            content={
                "error": "queue_saturated",
                "queue_depth": _inflight,
                "max_queue_depth": MAX_QUEUE_DEPTH,
            },
        )
    _inflight += 1

    body = json.loads(await request.body())
    text = body.get("text", "")
    voice = body.get("voice", DEFAULT_VOICE)
    speed = body.get("speed", 1.0)
    use_stream = body.get("stream", True)
    audio_format = body.get("format", FORMAT_WAV)
    sample_rate = body.get("sample_rate", SAMPLE_RATE)

    request_ts = body.get("request_timestamp")
    if request_ts is not None and time.time() - request_ts > MAX_REQUEST_AGE_S:
        _inflight -= 1
        return JSONResponse(status_code=408, content={"error": "request_stale"})

    if not text:
        _inflight -= 1
        return JSONResponse(status_code=400, content={"error": "text is required"})

    if audio_format not in _MEDIA_TYPES:
        _inflight -= 1
        return JSONResponse(
            status_code=400,
            content={"error": f"format must be one of {sorted(_MEDIA_TYPES)}"},
        )

    if voice not in _VALID_VOICES:
        _inflight -= 1
        return JSONResponse(
            status_code=400,
            content={"error": f"voice must be one of {sorted(_VALID_VOICES)}"},
        )

    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        _inflight -= 1
        return JSONResponse(
            status_code=400,
            content={"error": f"sample_rate must be one of {sorted(SUPPORTED_SAMPLE_RATES)}"},
        )

    async def _inflight_wrap(gen: AsyncGenerator[bytes, None]) -> AsyncGenerator[bytes, None]:
        global _inflight
        try:
            async for chunk in gen:
                yield chunk
        finally:
            _inflight -= 1

    if not use_stream:
        try:
            return await _invocations_sync(text, voice, speed, audio_format, sample_rate)
        finally:
            _inflight -= 1

    return StreamingResponse(
        _inflight_wrap(_stream_sentences_generator(text, voice, speed, audio_format, sample_rate)),
        media_type=_MEDIA_TYPES[audio_format],
    )


async def _receive_message(websocket: WebSocket) -> str:
    """Read one client frame as text, whatever frame type it arrived as.

    SageMaker's bidirectional transport forwards ``RequestPayloadPart`` as a
    *binary* WebSocket frame, so ``receive_text()`` raises ``KeyError: 'text'``
    on every request from ``invoke_endpoint_with_bidirectional_stream`` — the
    only way this endpoint is invoked in production. Browsers and the local
    test client send text frames. Accept both rather than picking one.
    """
    message = await websocket.receive()
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", 1000))
    payload = message.get("text")
    if payload is None:
        payload = message.get("bytes", b"").decode("utf-8")
    return payload


async def bidirectional_stream(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        while True:
            raw = await _receive_message(websocket)
            msg = json.loads(raw)

            if msg.get("type") == "close":
                break

            text = msg.get("text", "")
            voice = msg.get("voice", DEFAULT_VOICE)
            speed = msg.get("speed", 1.0)
            request_id = msg.get("request_id", "unknown")
            sample_rate = msg.get("sample_rate", SAMPLE_RATE)

            if not text:
                await websocket.send_text(
                    json.dumps(
                        {"type": "error", "request_id": request_id, "message": "text required"}
                    )
                )
                continue

            if voice not in _VALID_VOICES:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "request_id": request_id,
                            "message": f"voice must be one of {sorted(_VALID_VOICES)}",
                        }
                    )
                )
                continue

            if sample_rate not in SUPPORTED_SAMPLE_RATES:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "request_id": request_id,
                            "message": (
                                f"sample_rate must be one of {sorted(SUPPORTED_SAMPLE_RATES)}"
                            ),
                        }
                    )
                )
                continue

            await websocket.send_text(
                json.dumps({"type": "synthesis_start", "request_id": request_id})
            )

            t0 = time.monotonic()
            loop = asyncio.get_event_loop()

            async with _inference_lock:
                sentence_audios = await loop.run_in_executor(
                    None, _generate_sentences, text, voice, speed, sample_rate
                )

            cumulative_duration = 0.0
            for audio in sentence_audios:
                pcm = (audio * 32767).astype(np.int16).tobytes()
                await websocket.send_bytes(pcm)
                segment_duration = len(audio) / sample_rate
                cumulative_duration += segment_duration

            elapsed = time.monotonic() - t0

            await websocket.send_text(
                json.dumps(
                    {
                        "type": "synthesis_complete",
                        "request_id": request_id,
                        "total_duration_s": round(cumulative_duration, 3),
                        "elapsed_s": round(elapsed, 3),
                    }
                )
            )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        _logger.exception("WebSocket error")
        try:
            await websocket.send_text(
                json.dumps({"type": "error", "request_id": "unknown", "message": str(e)})
            )
        except Exception:
            pass


app = Starlette(
    routes=[
        Route("/ping", ping, methods=["GET"]),
        Route("/invocations", invocations, methods=["POST"]),
        WebSocketRoute("/invocations-bidirectional-stream", bidirectional_stream),
    ],
    lifespan=_lifespan,
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
