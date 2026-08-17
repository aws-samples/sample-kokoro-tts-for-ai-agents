# STT-TTS-Model-Eval

Deploying and autoscaling the Kokoro-82M text-to-speech model on Amazon SageMaker: a client
library for the deployed endpoint, benchmarking tools that turn a live endpoint's measured
behavior into an autoscaling config, and the CDK stack that deploys it.

## Layout

| Package | What it's for |
|---|---|
| `tts-client` | Blocking client for the deployed Kokoro endpoint, plus Amazon Polly |
| `tts-eval` | Quality evaluation — UTMOS/WER |
| `tts-bench` | Performance benchmarking — latency, scalability, cost, and the `qmax`/`ttotal`/`plan` autoscaling chain |
| `speech-infra` | CDK stack and the `speech-infra` CLI that deploy/manage the SageMaker endpoint |
| `tts-inference` | Model catalog types and local/SageMaker inference backends |
| `shared` | Shared types and data loaders used across the above |
| `models/tts-kokoro` | Isolated venv for local Kokoro inference outside the shared workspace |

## Setup

```bash
uv sync
uv run pytest -q
uv run pre-commit run --all-files
```

## Where to go next

- New to this repo and want to deploy a model and make a synthesis call?
  [`docs/quickstart.md`](docs/quickstart.md)
- Calling a deployed TTS endpoint from Python? [`packages/tts-client/README.md`](packages/tts-client/README.md)
- Figuring out autoscaling values for a deployed model?
  [`docs/tts-bench-and-autoscaling.md`](docs/tts-bench-and-autoscaling.md) (overview + a real
  worked example), or go straight to the operator recipe in
  [`packages/tts-bench/README.md`](packages/tts-bench/README.md) and the math in
  [`docs/autoscaling-capacity-model.md`](docs/autoscaling-capacity-model.md)
- Per-model technical notes (wire contracts, batching limitations) live in
  [`docs/models/`](docs/models/)

## Security Disclaimer

This solution is provided as a proof-of-value and is not intended as a
production-ready implementation. You are responsible for evaluating how the AWS
Shared Responsibility Model applies to your specific use case and for
implementing the necessary security controls to meet your organization's
requirements. AWS provides a broad set of security tools and configurations to
support your security objectives, and it is your responsibility as the developer
to ensure all aspects of your application are appropriately secured for
production use.
