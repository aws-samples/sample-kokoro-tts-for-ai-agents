# TTS Deployment Guide for SageMaker

A reference for deploying four candidate TTS models on SageMaker. Focused on which serving engines work for each, what trade-offs you're accepting, and the right approach per model — not step-by-step setup.

Models covered:
- **Kokoro-82M** (CPU and GPU variants)
- **Chatterbox-Turbo** (350M)
- **Maya1** (3B)
- **Step-Audio-EditX** (3B)

---

## Features & Trade-offs

| Model | Voice Cloning | Voice Design | Emotion Control | Streaming | Languages | Watermark | Quality Tier |
|---|---|---|---|---|---|---|---|
| **Kokoro-82M** | ❌ | ❌ Pre-set voices only | ❌ | Sentence-chunked | EN, JA, ZH, ES, FR, HI, IT, PT | None | Good |
| **Chatterbox-Turbo** | ✅ Zero-shot from reference clip | ❌ | ✅ Exaggeration knob + paralinguistic tags ([laugh], [cough]) | Intra-utterance, sub-200ms TTFA | English only | ✅ PerTh (always on) | Very Good |
| **Maya1** | ❌ | ✅ Text-described voices | ✅ 20+ inline tags (`<laugh>`, `<sigh>`, etc.) | Intra-utterance via vLLM, ~100ms TTFA | English (multi-accent) | None | Very Good |
| **Step-Audio-EditX** | ✅ Zero-shot from reference clip | ❌ | ✅ Iterative editing across passes | Intra-utterance via vLLM | Chinese + English (some Cantonese/Sichuanese) | None | Excellent (Elo 1101) |

### Trade-offs at a glance

| Trade-off | Winner | Why |
|---|---|---|
| **Lowest cost-per-character** | Kokoro | Smallest model, CPU-viable, no autoregressive loop |
| **Simplest to deploy** | Kokoro | No vLLM, no neural codec, standard PyTorch |
| **Voice cloning quality** | Step-Audio-EditX | Best Apache 2.0 cloning, plus iterative editing |
| **Voice cloning + MIT license** | Chatterbox-Turbo | Permissive license, mature ecosystem |
| **Voice design without reference audio** | Maya1 | Only one of the four that does text-described voices |
| **Best quality (Apache 2.0)** | Step-Audio-EditX | #2 open-weights model on Artificial Analysis leaderboard |
| **Fastest first-audio latency** | Kokoro (CPU) or Maya1 (vLLM) | Both target ~100-300ms TTFA |
| **Best for character voices / games** | Maya1 | Voice design + emotion tags = unique capability |
| **Best Chinese-language quality** | Step-Audio-EditX | Trained primarily on Chinese + English |
| **No watermarking required** | Kokoro, Maya1, Step-Audio-EditX | All output is clean; Chatterbox always watermarks |

---

## Serving Engine Compatibility

| Model | Triton | vLLM | FastAPI Custom | Serverless (CPU) | Notes |
|---|---|---|---|---|---|
| **Kokoro-82M** | ✅ Native | ❌ Not applicable | ✅ Easy (community wrapper) | ✅ Only CPU model | Non-autoregressive, no benefit from vLLM |
| **Chatterbox-Turbo** | ✅ Native | ⚠️ Community port for original Chatterbox, **not Turbo** | ✅ Easy (community wrapper) | ❌ Needs GPU | Turbo's 1-step decoder makes vLLM unjustified |
| **Maya1** | ⚠️ Possible but loses key features | ✅ **Official**, with prefix caching | ⚠️ Slow without batching | ❌ Needs GPU | vLLM is the intended path |
| **Step-Audio-EditX** | ⚠️ Possible but loses key features | ✅ **Official**, ships with `app.py` | ⚠️ Slow without batching | ❌ Needs GPU | vLLM is the intended path; CosyVoice decoder runs alongside |

**Why vLLM only matters for some:**

vLLM optimizes autoregressive LLM-style inference — continuous batching, paged KV cache, prefix caching. It's only valuable for models that spend most of their compute in an autoregressive token-generation loop. Maya1 and Step-Audio-EditX are exactly that pattern, so vLLM transforms their serving economics. Kokoro is non-autoregressive (one forward pass) and Chatterbox-Turbo has a distilled 1-step decoder — both have nothing for vLLM to optimize.

---

## Three Deployment Approaches

You only need to be familiar with three patterns:

### 1. Lightweight PyTorch (Kokoro)

**When:** Models without an autoregressive loop, or small enough that any serving stack works.

**Stack:** Standard PyTorch container, FastAPI or Triton in front, dynamic batching enabled.

**SageMaker fit:**
- **CPU**: ml.c6i.2xlarge with Serverless Inference for sporadic traffic, or real-time endpoint for steady traffic
- **GPU**: ml.g4dn.xlarge or ml.g5.xlarge for higher throughput / lower latency

**Base image:** `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`

**Optimization options:** `torch.compile`, ONNX export, TensorRT compilation for 1.5–2× speedup.

### 2. Triton + PyTorch (Chatterbox-Turbo)

**When:** Mid-size models with custom output heads where vLLM doesn't fit, but you need batching and production-grade serving.

**Stack:** Triton Inference Server with the PyTorch backend, dynamic batching configured in `config.pbtxt`.

**SageMaker fit:** ml.g5.xlarge or ml.g6.xlarge with a real-time endpoint. Response streaming via `InvokeEndpointWithResponseStream`.

**Base image:** Custom Dockerfile based on `nvcr.io/nvidia/tritonserver` or `pytorch/pytorch`. Community option: `devnen/Chatterbox-TTS-Server` is a working FastAPI implementation you can adapt.

**Why not vLLM here:** Chatterbox-Turbo's distilled 1-step decoder means there are very few autoregressive steps per utterance. vLLM's batching benefits depend on long autoregressive sequences. Triton's batching is sufficient.

### 3. vLLM + Codec Decoder (Maya1, Step-Audio-EditX)

**When:** LLM-style autoregressive models emitting neural codec tokens. Long autoregressive sequences mean vLLM's continuous batching and prefix caching deliver large throughput gains.

**Stack:** vLLM as the LLM backbone, codec decoder (SNAC for Maya1, CosyVoice decoder for EditX) as a post-processing step in-process.

**SageMaker fit:** ml.g5.xlarge for both. ml.g5.2xlarge/4xlarge if you want headroom for larger batch sizes or concurrent voices. Response streaming via `InvokeEndpointWithResponseStream`.

**Base image:** Custom Dockerfile based on `vllm/vllm-openai:latest`. Add codec decoder dependencies. Wrap with SageMaker `/ping` + `/invocations` handlers that proxy to the internal vLLM OpenAI-compatible server and run the decoder on the output tokens.

**Why this matters:** Prefix caching gives you significant throughput when reference audio or voice descriptions are reused across many requests. This is the case for most production workloads. Without vLLM, you're leaving 3–10× throughput on the table.

---

## Per-Model Quick Reference

### Kokoro-82M

**Best deployment:** SageMaker BYOC with FastAPI or Triton. Use Serverless Inference if traffic is spiky and CPU is acceptable; real-time endpoint with ml.g4dn.xlarge if you need consistent low latency.

**Key gotcha:** Requires `espeak-ng` installed in the container. Missing it is the #1 cause of silent failures.

**Quick start library:** `pip install kokoro` — single Python class (`KPipeline`), yields audio chunks via generator.

---

### Chatterbox-Turbo

**Best deployment:** SageMaker BYOC with Triton or FastAPI on ml.g5.xlarge. Compile with `torch.compile` or export to TensorRT for additional speedup. Skip vLLM — the community port exists but targets original Chatterbox and adds complexity without payoff for Turbo.

**Key gotcha:** PerTh watermark is always on. Every output has an inaudible watermark embedded. Important if your downstream pipeline does voice fingerprinting or identity verification.

**Quick start library:** `pip install chatterbox-tts` — single class, supports voice cloning via `audio_prompt_path`.

---

### Maya1

**Best deployment:** SageMaker BYOC based on `vllm/vllm-openai` + SNAC decoder. The official repo ships a `vllm_streaming_inference.py` reference script.

**Key gotcha:** SNAC token unpacking (7 tokens per frame, hierarchical) is fragile — use the official reference implementation, don't write it from scratch.

**Quick start path:** Hugging Face repo includes a working inference script. Voice control via `<description="...">` natural-language prompt + inline `<laugh>` / `<sigh>` tags.

---

### Step-Audio-EditX

**Best deployment:** SageMaker BYOC based on the official Docker image (which uses stock vLLM + CosyVoice decoder). The repo ships a Gradio app that you swap for a SageMaker `/invocations` handler.

**Key gotcha:** Audio longer than 30 seconds per inference degrades quality. Split long text at sentence boundaries.

**Quick start path:** `git clone` the repo, download weights from HF, run `python app.py` with `--model-source local`. Includes optional AWQ quantization to fit ~6–8 GB VRAM.

---

## Going from Local to SageMaker

The path is the same for all four:

1. **Validate quality locally** with a single Python script against your actual content
2. **Wrap in the serving layer** (FastAPI, Triton, or vLLM per the table above)
3. **Containerize** with a SageMaker-compatible `/ping` + `/invocations` interface (port 8080)
4. **Push to ECR** and create a SageMaker model + endpoint config + endpoint
5. **Enable autoscaling** on `InvocationsPerInstance`
6. **Benchmark with SageMaker Inference Recommender** to dial in instance type and batch settings

For real-time audio output, use `InvokeEndpointWithResponseStream` instead of standard invoke (supports up to 1 GB streamed response).

---

## Decision Quick Reference

| If your priority is... | Pick |
|---|---|
| Lowest cost, English-only narration/IVR | **Kokoro-82M** on ml.c6i.2xlarge (Serverless or real-time) |
| Voice cloning + emotion + MIT license + mature | **Chatterbox-Turbo** on ml.g5.xlarge (Triton) |
| Voice design from text + character/game expressiveness | **Maya1** on ml.g5.xlarge (vLLM) |
| Top quality + voice cloning + Apache 2.0 | **Step-Audio-EditX** on ml.g5.xlarge (vLLM) |

**Recommended evaluation order:** Kokoro first (fastest to validate baseline) → Chatterbox-Turbo (most likely production winner) → Maya1 and Step-Audio-EditX in parallel (top-quality candidates). Run all four on the same 30-minute test corpus and pick from blind A/B results.
