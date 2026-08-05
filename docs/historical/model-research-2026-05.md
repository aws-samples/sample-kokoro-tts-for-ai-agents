# Historical: Pre-Deployment Model Research (2026-05)

Merged from `docs/tts-deployment-guide.md` and `docs/tts-architecture-reference.md`, written
before any of the four TTS models were actually deployed. Reality has since diverged in named
ways:

- These docs recommend Triton for feed-forward models; the actual deployed containers
  (`packages/speech-infra/containers/`) are custom PyTorch/FastAPI, not Triton.
- `Step-Audio-EditX` was evaluated here but never deployed.
- The "Maya1" writeup below doesn't match the actually-deployed `maya-veena` model.
- Today's real, deployed model config — instance types, container types, autoscaling — lives
  in `packages/speech-infra/src/speech_infra/config.py`, not here.

Kept for the parts still useful to a future 5th/6th model evaluation: the serving-engine
decision logic and the SageMaker bidirectional-streaming container contract. Per-model
recommendations, effort estimates, and the model-vs-model trade-off tables from the original
docs are cut — they answered a decision that's already been made.

## Serving engine choice: why vLLM only matters for autoregressive stages

vLLM optimizes autoregressive LLM-style inference — continuous batching, paged KV cache,
prefix caching. That's only valuable for models spending most of their compute in an
autoregressive token-generation loop. Feed-forward or 1-step-distilled stages (Kokoro's whole
pipeline, S3Gen after Chatterbox-Turbo's distillation) have nothing for vLLM to optimize —
using it there just adds complexity.

Three deployment patterns cover nearly everything:

1. **Lightweight PyTorch** — for feed-forward or tiny models. Standard PyTorch container,
   FastAPI or Triton in front, dynamic batching if the model can actually use it (see
   `docs/models/kokoro-batching-limitation.md` for a case where it can't).
2. **Triton + PyTorch** — for mid-size models with custom output heads where vLLM doesn't
   fit but production-grade batching still matters.
3. **vLLM + codec decoder** — for LLM-backbone models emitting neural codec tokens (SNAC,
   CosyVoice, etc.) where long autoregressive sequences make vLLM's continuous batching and
   prefix caching pay off. The codec decoder itself is a feed-forward post-processing step
   and should be `torch.compile`d separately, not run through vLLM.

## Streaming and TTFB by architecture

| Architecture | Streaming unit | What gates first audio |
|---|---|---|
| Feed-forward, single pass (Kokoro) | Per segment | One forward pass over the first segment |
| LLM backbone -> codec decoder (Orpheus, Maya/Veena-style) | Per N-token codec frame | Stage-1 first N tokens + one codec-decoder pass |
| Hybrid AR + distilled feed-forward (Chatterbox-Turbo-style) | Per AR token group | AR stage's first tokens; the distilled decoder step is cheap enough not to gate |

Segmentation granularity is a real, measurable lever independent of architecture — see
`docs/models/kokoro-streaming-api.md` for a concrete before/after on sentence- vs.
newline-segmented input.

## SageMaker bidirectional-streaming container contract

For a model whose serving strategy needs a persistent connection (voice-agent barge-in,
incremental text as an upstream LLM produces it):

- Implement a WebSocket endpoint at `ws://localhost:8080/invocations-bidirectional-stream`.
- Docker label: `com.amazonaws.sagemaker.capabilities.bidirectional-streaming=true`.
- Client side uses the `InvokeEndpointWithBidirectionalStream` API — see
  `packages/tts-client/tts_client/client.py`'s `synthesize_bidi()` for this repo's client.
- SageMaker does **not** batch across WebSocket sessions — each client gets its own. Batching
  across concurrent sessions, if needed, is the container's own job (an internal scheduler
  pulling from all live sessions), not something the platform provides.

Pure one-way TTS doesn't need this — a response-stream (`InvokeEndpointWithResponseStream`)
is sufficient unless the model is one leg of a larger bidirectional voice agent, or you need
barge-in/interruption over one persistent connection.
