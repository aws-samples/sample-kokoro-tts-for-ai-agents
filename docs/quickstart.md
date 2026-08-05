# Quickstart

Setup, deploy a model, make a synthesis call, tear down. `list`, the `describe-endpoint`
check, and the synthesis call were run for real against this repo's live AWS account while
writing this doc (against the already-deployed `speech-kokoro-82m`); `deploy`/`destroy`
follow the same pattern `speech-infra` uses everywhere else in this repo.

## 1. Setup

```bash
uv sync
```

## 2. See what's deployable

```bash
uv run speech-infra list
```

```
Model                Endpoint                     Instance         Container
--------------------------------------------------------------------------------
whisper-large-v3     speech-whisper-large-v3      ml.g5.2xlarge    pytorch
qwen3-asr            speech-qwen3-asr             ml.g5.xlarge     pytorch
kokoro-82m           speech-kokoro-82m            ml.g5.xlarge     pytorch
kokoro-82m-cpu       speech-kokoro-82m-cpu        ml.c5.2xlarge    pytorch
maya-veena           speech-maya-veena            ml.g5.xlarge     vllm
chatterbox-turbo     speech-chatterbox-turbo      ml.g5.xlarge     pytorch
orpheus-3b           speech-orpheus-3b            ml.g5.xlarge     vllm
```

`kokoro-82m-cpu` is the cheapest one to try first — a CPU instance (`ml.c5.2xlarge`), so it
doesn't compete for this account's limited GPU quota (`ml.g5.xlarge` is capped at 4,
account-wide).

## 3. Deploy it

```bash
uv run speech-infra deploy kokoro-82m-cpu
```

This runs `cdk deploy` for the endpoint's stack. Takes a few minutes.

Check it landed:

```bash
aws sagemaker describe-endpoint --endpoint-name speech-kokoro-82m-cpu \
  --query "{Status:EndpointStatus,Instances:ProductionVariants[0].CurrentInstanceCount}"
```

Wait for `"Status": "InService"`. (`speech-infra status` exists as a CLI command but doesn't
actually check anything yet — `describe-endpoint` is the real answer today.)

## 4. First synthesis call

```python
from tts_client.client import TTSClient
from tts_client.types import SynthesisRequest

client = TTSClient(region="us-east-1")
request = SynthesisRequest(text="Hello there.", voice="af_heart")
result = client.synthesize("speech-kokoro-82m-cpu", request)  # or your endpoint's name

print(result.audio_bytes[:4])   # b'RIFF' -- a WAV header
print(result.duration_s, result.latency_ms)
```

Run for real against the already-deployed `speech-kokoro-82m` while writing this doc (same
call shape, different endpoint name):

```
audio_bytes: 73244
audio_format: wav
duration_s: 1.525
latency_ms: 158.2
```

See [`packages/tts-client/README.md`](../packages/tts-client/README.md) for the rest of what
`tts-client` can do (bidirectional streaming, incremental text-in, Amazon Polly).

## 5. Now that it's deployed: autoscaling

A freshly deployed model has no autoscaling tuned to it yet — the default config is a
placeholder. Turning a real measurement into deployed autoscaling values is its own
workflow: see [`docs/tts-bench-and-autoscaling.md`](tts-bench-and-autoscaling.md).

## 6. Teardown

```bash
uv run speech-infra destroy kokoro-82m-cpu
```
