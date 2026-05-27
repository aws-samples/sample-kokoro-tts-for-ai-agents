# TTS Architecture Reference

Architectural details, serving strategies, and deployment guidance for the four TTS models in this evaluation suite.

## Overview

| Model | Params | Type | Stages | Codec/Vocoder | Serving Engine | Streaming |
|-------|--------|------|--------|---------------|----------------|-----------|
| Kokoro | 82M | Feed-forward | Single-stage acoustic + vocoder | Built-in | Custom PyTorch / Triton | Sentence-chunked |
| Orpheus | 3B | Autoregressive | LLM -> SNAC decoder | SNAC 24kHz | vLLM / TRT-LLM | Token-chunked (7-token groups) |
| Maya/Veena | ~3B | Autoregressive | CausalLM -> SNAC decoder | SNAC 24kHz | transformers (current), vLLM (target) | Token-chunked |
| Chatterbox-Turbo | 350M | Hybrid (AR + FF) | T3 (GPT-2) -> S3Gen (flow-match) + HiFiGAN | S3Gen + HiFiGAN | Custom PyTorch (current) | Token-chunked via T3 |

## Model Architectures

### Kokoro 82M

**Classification:** Non-autoregressive, feed-forward.

**Pipeline:** Custom KPipeline -- acoustic model produces mel spectrogram, vocoder converts to waveform. No KV cache, no sequential token dependency. Behaves like the audio decoder/vocoder at the end of LLM-TTS pipelines.

**Streaming:** Sentence-chunked. The pipeline yields audio per sentence/segment. First chunk emits immediately after that segment's forward pass completes.

**Serving engine:** Skip vLLM/SGLang/TRT-LLM entirely -- their value (paged KV, continuous batching, speculative decoding) is meaningless for feed-forward models. Two options:

1. **Custom PyTorch server** (FastAPI/Starlette or gRPC) with own micro-batcher
2. **NVIDIA Triton** with PyTorch backend -- `dynamic_batching` config with `max_queue_delay_microseconds` (~15ms timeout)

**torch.compile:** Cache the compiled artifact (compilation takes minutes, dominates cold start). Pin CUDA/cuDNN/torch versions exactly. Bucket inputs by (batch_size, seq_len) to avoid recompilation on new shapes. Mark dynamic dims explicitly or pad to a fixed bucket set.

**Hardware:** 82M params is tiny -- bake weights into the Docker image (no S3/HF fetch on cold start). L4/A10G/g4dn is plenty. Scale horizontally (each GPU is own replica). Pack with MIG slices to drive per-stream cost down.

---

### Orpheus 3B

**Classification:** Autoregressive (Llama-3B backbone). Canonical LLM-backbone TTS template.

**Pipeline -- two stages:**

| Stage | Component | Type | Serving Strategy |
|-------|-----------|------|------------------|
| 1 | Llama-3B | Autoregressive | vLLM (current, v0.7.3) or TRT-LLM (production) |
| 2 | SNAC decoder (`hubertsiuzdak/snac_24khz`) | Feed-forward | torch.compile + shape bucketing |

**Stage 1** generates SNAC audio tokens autoregressively. Continuous batching via the LLM engine. Prefix caching applies for repeated system prompts and speaker tokens.

**Stage 2** converts SNAC token codes to waveform. This stage SHOULD be torch.compiled -- same compilation strategy as Kokoro (cache artifact, pin versions, bucket shapes). It is NOT "N/A" just because the overall model is AR.

**Speaker identity:** Via special tokens prepended to the prompt (not reference audio clips).

**Context window:** 2048 tokens sufficient for most utterances.

**vLLM vs TRT-LLM tradeoff:**

- **vLLM** -- Python-native, fast iteration, good enough for eval/dev. Provides continuous batching + paged KV cache.
- **TRT-LLM** -- Higher throughput on Hopper+ GPUs with native FP8 quantization. Worth it at production scale. Same batching semantics.

**TTFB:** ~150ms on H100 (Stage 1 first token generation + SNAC decode of first 7-token chunk).

**Streaming:** As Stage 1 emits groups of 7 SNAC tokens, Stage 2 decodes and yields audio bytes immediately. Each 7-token group produces one audio frame.

**Hardware:** A10G/L4 for eval, H100 for production throughput. Single GPU per replica.

---

### Maya/Veena

**Classification:** Autoregressive (CausalLM). Architecturally identical to Orpheus.

**Pipeline:** Same two-stage template as Orpheus -- CausalLM (AR) generating SNAC tokens -> SNAC vocoder (feed-forward). Speaker identity via tokens (same as Orpheus, not reference clips). Uses the same SNAC codec: `hubertsiuzdak/snac_24khz`.

**Current state (NOT production-ready):**

- Ships on `transformers .generate()` + BitsAndBytes 4-bit (nf4) quantization
- No continuous batching, no paged KV cache, no prefix caching
- Single-request throughput only
- Adequate for evaluation but unacceptable for production latency/throughput

**Migration path to production:**

1. **Move Stage 1 to vLLM** (same configuration as Orpheus) -- gains continuous batching, paged KV, prefix caching
2. **Drop BitsAndBytes nf4 for FP8 quantization** -- native in vLLM/TRT-LLM, better accuracy-per-bit than nf4
3. **Reuse the same compiled SNAC decoder** as Orpheus -- identical codec, one compiled artifact serves both models

**Key insight:** After migration, Maya/Veena's serving infrastructure is identical to Orpheus. One compiled SNAC decoder shared across both models. The only difference is the Stage 1 model weights.

---

### Chatterbox-Turbo 350M

**Classification:** Hybrid -- autoregressive first stage, feed-forward second stage. Each stage wants opposite serving strategies.

**Pipeline -- two stages with opposing characteristics:**

| Stage | Component | Type | Details |
|-------|-----------|------|---------|
| T3 | GPT-2 medium (AR) | Autoregressive | Has KV cache, decodes token-by-token, supports continuous batching |
| S3Gen + HiFiGAN | Flow-matching decoder (1-step distilled) + vocoder | Feed-forward | Turbo distilled from 10 steps to single-pass |

**Critical Turbo insight:** The decoder bottleneck is solved by distillation. Latency is now dominated by T3's autoregressive generation, not S3Gen. The S3Gen step is effectively free compared to T3 token generation.

**Voice cloning:** Mandatory -- requires reference audio for every synthesis. Cache speaker conditioning tensors by voice ID to avoid recomputing on repeated requests.

**Batch size = 1 enforced** in the library code. Two paths forward:

1. **Don't fight it:** Scale horizontally (one replica per GPU, pack onto MIG slices). Right choice for voice-agent latency where p99 matters more than throughput.
2. **Re-engineer:** Front T3 with an LLM engine for continuous batching + dynamic batcher for S3Gen. Defer until economics demand it.

**Serving architecture:**

- **T3:** LLM-engine territory. Candidate for TensorRT-LLM or vLLM. Can quantize to FP8.
- **S3Gen:** Serves like Kokoro's pipeline -- torch.compile + dynamic batching (~15ms timeout). No in-flight batching possible (single forward pass).
- **Co-locate T3 + S3Gen in one container** by default. Split into separate services only at scale -- the inter-service hop adds latency that negates the throughput gain.

**PerTh watermarking:** Always on. Small post-processing step on every request (negligible latency).

**TTFB:** Sub-200ms realistic. Gated by T3's first tokens, not the decoder. Stream audio chunks as T3 emits tokens -> single-step S3Gen -> emit.

**Hardware:** L4 (g6) or A10G (g5) is plenty at 350M params. Bake weights into image.

---

## Serving Engine Matrix

| Engine | Applicable Models | Why | Key Features |
|--------|-------------------|-----|--------------|
| vLLM | Orpheus Stage 1, Maya Stage 1, Chatterbox T3 | AR models with KV cache | Continuous batching, paged KV, prefix caching, Python-native |
| TRT-LLM | Orpheus Stage 1, Maya Stage 1, Chatterbox T3 | Production throughput on Hopper+ | FP8 native, higher tokens/s, same batching as vLLM |
| Custom PyTorch + torch.compile | Kokoro (whole model), SNAC decoder (Orpheus/Maya), S3Gen + HiFiGAN (Chatterbox) | Feed-forward stages | Dynamic batching with shape bucketing, ~15ms queue timeout |
| NVIDIA Triton | Kokoro, SNAC decoder, S3Gen | Alternative to custom server for FF stages | Built-in dynamic_batching config, model ensemble pipelines |

**Models that should NOT use LLM engines:** Kokoro (entirely feed-forward), SNAC decoder stage, S3Gen stage. The value proposition of LLM engines (paged KV, continuous batching, speculative decoding) is meaningless for feed-forward inference.

---

## Dynamic Batching

### Feed-Forward Stages (Kokoro, SNAC, S3Gen)

Strategy: Accumulate requests, fire when batch fills OR ~10-15ms timer expires.

Implementation guidance:

- **Shape bucketing:** Group requests by (batch_size, seq_len) to prevent torch.compile recompilation on novel input shapes. Define a fixed set of bucket sizes and pad inputs to the nearest bucket.
- **Dynamic dims:** Alternatively, mark sequence length as a dynamic dimension in torch.compile. Trades slight overhead for flexibility.
- **Queue timeout:** 10-15ms balances latency (waiting for batch to fill) against throughput (larger batches amortize overhead). Tune based on traffic pattern.
- **Triton config example:**
  ```
  dynamic_batching {
    max_queue_delay_microseconds: 15000
    preferred_batch_size: [4, 8, 16]
  }
  ```

### Autoregressive Stages (Orpheus/Maya Stage 1, Chatterbox T3)

Strategy: Continuous (in-flight) batching via LLM engine.

- New requests join the running batch without waiting for existing requests to finish
- Each request independently generates tokens at its own pace
- vLLM and TRT-LLM handle this transparently -- no application-level batching code needed
- **Prefix caching:** Enable for repeated speaker tokens / system prompts (amortizes KV computation across requests with shared prefixes)

### Chatterbox Batch=1 Constraint

The library enforces batch_size=1. Until re-engineered:

- Scale horizontally: one replica per GPU, pack replicas onto MIG slices
- Each replica handles one concurrent synthesis
- Autoscaler target = number of concurrent real-time streams needed

---

## Streaming and Latency

### TTFB by Model

| Model | Expected TTFB | Gated By | Streaming Granularity |
|-------|---------------|----------|----------------------|
| Kokoro | <50ms | Single forward pass for first sentence | Per-sentence |
| Orpheus | ~150ms (H100) | Stage 1 first 7 tokens + SNAC decode | Per 7-token group (~21ms audio) |
| Maya/Veena | ~150ms (after vLLM migration) | Same as Orpheus | Per 7-token group |
| Chatterbox-Turbo | <200ms | T3 first tokens -> 1-step S3Gen | Per T3 token group |

### Streaming Implementation

**Orpheus/Maya (SNAC-based):** Stage 1 emits tokens continuously. Every 7 tokens form one SNAC frame. Stage 2 decodes that frame immediately and yields audio bytes. Client receives audio chunks as they're produced -- no need to wait for full generation.

**Kokoro:** KPipeline yields audio per sentence/segment. Each segment is a complete forward pass. First segment streams immediately; subsequent segments may benefit from pre-computation overlap.

**Chatterbox-Turbo:** T3 emits semantic tokens. Groups of tokens feed into single-step S3Gen which produces audio. Stream pattern: T3 tokens -> S3Gen forward -> emit audio chunk -> repeat.

---

## SageMaker Deployment

### Bidirectional Streaming Contract

**Container requirements:**

- Implement WebSocket endpoint at `ws://localhost:8080/invocations-bidirectional-stream`
- Docker label: `com.amazonaws.sagemaker.capabilities.bidirectional-streaming=true`
- Client uses `InvokeEndpointWithBidirectionalStream` API

**Batching behavior:** SageMaker does NOT batch across WebSocket sessions. Each client gets its own session. For cross-request batching, the container needs an internal scheduler that pulls pending work from all live sessions and batches it.

### When to Use Bidirectional Streaming

Pure TTS is mostly server->client stream. Bidirectional earns its keep when:

- TTS is a leg of a voice agent (ASR -> LLM -> TTS) and text is fed incrementally as the upstream LLM produces it
- Support for barge-in / interruption over one persistent connection

For standalone TTS, a one-way response stream is sufficient; bidirectional is just a convenient persistent WebSocket.

### SageMaker vs ECS/EKS

| Dimension | SageMaker | ECS/EKS |
|-----------|-----------|---------|
| Ops burden | Least (managed) | More (self-managed) |
| Autoscaling | Built-in, target-based | KEDA-driven, true scale-to-zero |
| Batching control | Container-internal only | Full control over batcher |
| Best for | One or few models, moderate traffic | Many models, high traffic, need fine control |

Reach for ECS/EKS only when running enough models/traffic that the control pays for the ops burden.

### Autoscaling Guidance

Set concurrency target to the number of simultaneous real-time streams a single replica can sustain within TTFB budget. Make internal batch size match that number for utilization/latency alignment.

---

## Optimization Roadmap

### torch.compile

| Target | Priority | Notes |
|--------|----------|-------|
| Kokoro (full model) | High | Entire pipeline benefits. Cache compiled artifact -- compilation takes minutes, dominates cold start. |
| SNAC decoder (shared by Orpheus + Maya) | High | One compiled artifact serves both models. Same bucketing strategy as Kokoro. |
| Chatterbox S3Gen | Medium | Already 1-step distilled. Compilation reduces the already-small constant further. |

**Compilation best practices:**

- Pin CUDA, cuDNN, and torch versions exactly (artifact is version-specific)
- Pre-compile during Docker build, not at runtime
- Bucket shapes to avoid recompilation: define fixed (batch_size, seq_len) pairs
- Mark dynamic dims if shape diversity is high

### Quantization

| Model | Current | Target | Benefit |
|-------|---------|--------|---------|
| Orpheus Stage 1 | FP16 (vLLM default) | FP8 (TRT-LLM or vLLM on Hopper+) | ~2x throughput, minimal quality loss |
| Maya Stage 1 | BitsAndBytes nf4 | FP8 (via vLLM/TRT-LLM) | Better accuracy-per-bit than nf4, production-grade |
| Chatterbox T3 | FP16 | FP8 | Modest gain at 350M params |
| Kokoro | FP32/FP16 | FP16 (sufficient) | 82M params, quantization overhead not worth it |

### KV-Cache Tuning

Applies to: Orpheus Stage 1, Maya Stage 1, Chatterbox T3.

- **Context length:** 2048 tokens sufficient for Orpheus/Maya. Chatterbox T3 context depends on utterance length.
- **Prefix caching:** Enable for speaker tokens and system prompts shared across requests.
- **Paged attention:** vLLM handles automatically. TRT-LLM requires explicit configuration.
- **Memory budget:** Set `gpu_memory_utilization` to leave headroom for SNAC/S3Gen decoder co-located on same GPU.

### Shared Infrastructure Opportunities

1. **One SNAC decoder** compiled once, shared by Orpheus and Maya/Veena
2. **One dynamic batching framework** for all feed-forward stages (Kokoro, SNAC, S3Gen)
3. **One container template** for AR+FF two-stage models (Orpheus, Maya, Chatterbox)
4. **Speaker conditioning cache** shared across Chatterbox replicas (Redis/memcached by voice ID)
