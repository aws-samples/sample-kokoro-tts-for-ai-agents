# Benchmark Artifacts

Machine-generated outputs from `tts-bench` runs against live SageMaker
endpoints. These are committed as the initial reference set so that examples,
tests, and capacity-planning tooling can leverage real data.

## File naming

| Pattern | Source command | Purpose |
|---------|---------------|---------|
| `plan-<model>.json` | `tts-bench plan` | Full autoscaling plan (instances, cooldowns, cost) |
| `qmax-<model>-<mode>-<instance>-<image>.json` | `tts-bench qmax` | Q_max measurement (max safe queue depth) |
| `ttotal-<model>-<trigger>-<instance>-<image>.json` | `tts-bench ttotal` | T_total measurement (scale-out wall-clock time) |
| `validate-<model>.json` | `tts-bench validate` | SLO validation checks against a live fleet |

## Regenerating

Any of these can be reproduced by re-running the corresponding `tts-bench`
sub-command against the deployed endpoint. The image digest in the filename
pins the exact container version that was measured.
