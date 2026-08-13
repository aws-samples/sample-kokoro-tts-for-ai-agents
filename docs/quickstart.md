# Quickstart

Setup, deploy the model, make a synthesis call, tear down. Every command below was run for
real against this repo's live AWS account while writing this doc.

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
kokoro-82m           speech-kokoro-82m            ml.g5.xlarge     pytorch
```

One model: Kokoro-82M, on a single `ml.g5.xlarge` GPU instance.

## 3. Deploy it

```bash
uv run speech-infra deploy kokoro-82m
```

This runs `cdk deploy` for the endpoint's stack. Takes a few minutes.

Check it landed:

```bash
aws sagemaker describe-endpoint --endpoint-name speech-kokoro-82m \
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
result = client.synthesize("speech-kokoro-82m", request)

print(result.audio_bytes[:4])   # b'RIFF' -- a WAV header
print(result.duration_s, result.latency_ms)
```

Run for real against the deployed endpoint while writing this doc:

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
uv run speech-infra destroy kokoro-82m
```
