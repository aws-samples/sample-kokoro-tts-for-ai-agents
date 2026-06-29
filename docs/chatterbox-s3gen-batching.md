# Chatterbox-Turbo: Batching S3Gen to Reduce Cost

## Summary

Chatterbox-turbo costs $36.98/M chars on ml.g5.xlarge — 60x more than kokoro-82m ($0.62/M). The bottleneck is S3Gen (diffusion-based vocoder) processing utterances sequentially in a for-loop. The batch=1 restriction is artificial; the underlying neural networks support batched inference. Removing this constraint should yield 3-4x throughput improvement, reducing cost to ~$10-12/M chars.

## Problem

The chatterbox-vllm `generate_with_conds()` pipeline:

```
1. T3 (vLLM autoregressive) — generates speech tokens, batches efficiently via continuous batching
2. S3Gen (diffusion) — converts tokens to audio, runs SEQUENTIALLY per utterance
```

After T3 completes a batch of N utterances, S3Gen processes them one at a time in a for-loop (`tts.py:346-364`):

```python
for i, batch_result in enumerate(batch_results):
    for output in batch_result.outputs:
        speech_tokens = ...
        wav, _ = self.s3gen.inference(speech_tokens=speech_tokens, ref_dict=s3gen_ref, n_timesteps=10)
        results.append(wav.cpu())
```

This means at saturation concurrency (C=4), T3 batches 4 sequences efficiently but S3Gen then holds the GPU for 4 sequential diffusion passes.

## Root Cause: Artificial Batch=1 Constraints

Two explicit restrictions prevent batching:

1. **`CausalMaskedDiffWithXvec.inference`** (`flow.py:256`):
   ```python
   assert token.shape[0] == 1
   ```

2. **`CausalConditionalCFM`** (`flow_matching.py:201`):
   ```python
   self.rand_noise = torch.randn([1, 80, 50 * 300])  # hardcoded batch=1
   ```

## Why Batching Is Feasible (Unlike Kokoro)

Unlike Kokoro-82M (see `docs/kokoro-batching-limitation.md`), where variable-length duration expansion makes batching architecturally impossible, Chatterbox's S3Gen uses standard transformer/U-Net components that natively support batched tensors:

| Component | Batch Support | Evidence |
|-----------|:---:|---|
| UpsampleConformerEncoder | Yes | Uses `token_len` masking, `make_pad_mask` for variable lengths |
| CausalConditionalCFM (diffusion) | Yes | `solve_euler` already runs batch=2 for CFG (conditional + unconditional) |
| ConditionalDecoder (U-Net) | Yes | All ops use `(batch_size, n_feats, mel_timesteps)` shape |
| HiFiGAN | Yes | Fully convolutional, operates on arbitrary batch dimension |

The `solve_euler` method is particularly telling — it already constructs `x_in = [2, 80, mel_len]` for classifier-free guidance, proving the decoder handles batch>1.

## S3Gen GPU Cost

S3Gen is NOT lightweight. It runs:
- **UpsampleConformerEncoder**: 6-block conformer with 8 attention heads, 2048 FFN, upsampling
- **CausalConditionalCFM**: 10 Euler steps, each step running the full ConditionalDecoder
  - ConditionalDecoder: 1 down-block (4 attention layers) + 12 mid-blocks (48 attention layers) + 1 up-block (4 attention layers) = 56 attention layers per step
  - With 10 diffusion steps = **560 attention forward passes per utterance**
  - CFG doubles this: 1120 attention passes total
- **HiFiGAN**: Single feedforward pass, upsample [8, 5, 3] — fast, ~5% of total S3Gen time

## Implementation Approach

### Dynamic Batching in the Async Handler

Add a request accumulation layer in `streaming_proxy.py`:
- Incoming requests queue for up to 50ms (or until N requests arrive)
- T3 processes the batch (already works)
- S3Gen processes the batch in a single forward pass (new)
- Results are returned to individual callers

### Patching S3Gen Inference

Since `chatterbox-vllm` is pip-installed in the container, monkey-patch at runtime:

1. Remove `assert token.shape[0] == 1` in the flow inference path
2. Replace `self.rand_noise[:, :, :mu.size(2)]` with `torch.randn(batch_size, 80, mel_len)`
3. Pad variable-length speech tokens to max length with corresponding `token_len` tensor
4. After diffusion, slice each output back to its actual mel length before HiFiGAN

### Handling Variable-Length Tokens

Different T3 outputs produce different token counts. For batched S3Gen:
```python
# Pad speech tokens to max length in batch
max_len = max(t.shape[0] for t in speech_token_list)
padded = torch.zeros(batch_size, max_len, device='cuda', dtype=torch.long)
token_lens = torch.zeros(batch_size, dtype=torch.long)
for i, tokens in enumerate(speech_token_list):
    padded[i, :tokens.shape[0]] = tokens
    token_lens[i] = tokens.shape[0]

# Run batched through encoder + diffusion
# Slice output mels back to actual lengths before HiFiGAN
```

## Expected Impact

| Metric | Before (serial) | After (batch=4) | Improvement |
|--------|:---:|:---:|:---:|
| Throughput at C=4 | 11-12 chars/s | ~35-45 chars/s | 3-4x |
| $/M chars | $36.98 | ~$10-12 | 3x cheaper |
| Single-request latency | 3.7s P50 | 3.7s P50 | Unchanged |
| Saturation concurrency | 4 | 8-16 | Higher |

## Memory Budget

GPU memory (A10G = 24GB):
- T3 vLLM: ~70% = 16.8GB (KV cache + model weights)
- S3Gen + HiFiGAN: ~30% = 7.2GB
- Per-utterance mel tensor at batch=4: `4 * 80 * 400 * 4 bytes` = 512KB
- Diffusion intermediates at batch=4: ~50MB total

Memory is not a constraint for batch sizes up to 8.

## Additional Optimizations (Stackable)

These can be combined with batching for further gains:

1. **Reduce diffusion steps 10 -> 5**: Nearly halves S3Gen time. Code comments note "5 is often enough."
2. **Enable FP16 for S3Gen** (`s3gen_use_fp16=True`): Currently disabled. Halves memory bandwidth.
3. **Pipeline T3 and S3Gen**: Start S3Gen on utterance 1 while T3 still generates utterance 2+. Reduces wall-clock at load.

Combined with batch=4: potential 6-8x throughput improvement, bringing cost to ~$5-7/M chars.

## Risks

- **Padding waste**: If one utterance is 3x longer than others in the batch, 2/3 of compute on shorter items is wasted. Mitigate by sorting requests by estimated length or capping the max padding ratio.
- **T3 memory pressure**: If `GPU_MEMORY_UTILIZATION` for vLLM is too high, S3Gen batching may OOM. Monitor and reduce vLLM allocation if needed.
- **Quality regression**: Verify via UTMOS scoring that batched output matches serial output. The math should be identical (same weights, same noise), but numerical precision with padding could theoretically differ at boundaries.
