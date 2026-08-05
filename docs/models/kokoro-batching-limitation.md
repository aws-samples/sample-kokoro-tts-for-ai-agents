# Kokoro-82M: Why Batch Inference Is Not Possible

## Summary

The Kokoro-82M TTS model cannot process multiple samples in a single forward pass. This applies to both the ONNX export (`kokoro-onnx`) and the PyTorch source (`kokoro` v0.9.4+). The model is architecturally limited to batch_size=1.

## Root Cause: Variable-Length Duration Expansion

Classic ML models (BERT, ResNet, GPT) produce fixed-shape outputs per sample — pad inputs to the same length, run the batch through identical tensor operations, get `[batch, seq_len, hidden]` or `[batch, classes]` back.

Kokoro uses a **duration predictor** that breaks this property:

1. The duration predictor outputs per-phoneme frame counts that vary per input:
   - "hello" might predict `[12, 8, 15, 9, 6]` = 50 total mel frames
   - "hi" might predict `[14, 11]` = 25 total mel frames

2. `repeat_interleave` expands hidden states by these durations — after this step, sample A is 50 frames and sample B is 25 frames.

3. The iSTFTNet decoder runs on this variable-length expanded representation to produce audio.

After the duration expansion, samples have **incompatible tensor shapes** that cannot be stacked into a single `[batch, frames, hidden]` tensor without padding.

## Evidence

**ONNX model inputs** (confirmed via `onnx.load`):
- `tokens`: shape `[1, sequence_length]` — hardcoded batch dim of 1
- `style`: shape `[1, 256]`
- `speed`: shape `[1]`

**PyTorch `KModel.forward_with_tokens()`** (kokoro v0.9.4):
- `pred_dur.squeeze()` collapses the batch dimension
- `torch.repeat_interleave` on `pred_dur` only works for a 1D tensor (single sample)
- Alignment matrix `pred_aln_trg` is constructed for a single sequence

## What True Batching Would Require

1. Pad all expanded sequences to the longest in the batch
2. Add masking to prevent padded positions from affecting computation
3. Rewrite the alignment construction logic (`pred_aln_trg`)
4. Modify the iSTFTNet decoder to handle masked variable-length inputs
5. Re-export the ONNX model with a dynamic batch dimension

This is not a drop-in change — it requires architectural rework of the model internals.

## ONNX and GPU Acceleration

ONNX Runtime does support CUDA and TensorRT execution providers. The Kokoro container already uses TensorRT for GPU-accelerated inference. The limitation is **not** GPU vs CPU — it's that the model processes one sample per forward pass even on GPU.

## Practical Implications

- The "dynamic batching" queue (`BATCH_MAX_WAIT_MS=100`, `BATCH_MAX_SIZE=8`) that was previously in `serve.py` added 100ms latency per request with zero throughput benefit — it collected requests but processed them sequentially in a for-loop.
- Kokoro throughput on a single GPU is capped by sequential processing time.
- To increase aggregate throughput, **scale horizontally** (multiple instances/endpoints) rather than attempting to parallelize inference on a single GPU.
- Optimize per-sample latency via TensorRT execution provider and eliminating unnecessary queue delays.

## Other Models for Comparison

- **Orpheus-3B / Chatterbox-Turbo** (vLLM-based): Use continuous batching via the vLLM engine, which handles concurrent requests internally with true parallelism.
- **FastSpeech2, StyleTTS**: Same duration-based architecture as Kokoro — same batch_size=1 limitation.
- **VITS, Bark**: Autoregressive or flow-based — different architecture, different constraints.
