# Benchmarking and Autoscaling: Overview and a Worked Example

`tts-bench` turns two measurements of a deployed endpoint into a deployable autoscaling
config. Four commands in order: `drift` (preflight) -> `qmax` (find the concurrency ceiling)
-> `ttotal` (time a scale-out) -> `plan` (arithmetic on both, prints a config block to paste).

For the exact flags, what a good run looks like, and troubleshooting, see
[`packages/tts-bench/README.md`](../packages/tts-bench/README.md) — that's the operator
recipe and isn't repeated here. For why the numbers relate to each other the way they do —
the full variable set, the CloudWatch unit-conversion problem, the equations — see
[`autoscaling-capacity-model.md`](autoscaling-capacity-model.md). This doc is the front door:
what problem this solves, the two numbers that matter, and a real worked example.

## The two numbers that matter

- **`Q_max`** — the largest concurrency (queued + executing, on one instance) at which p95
  time-to-first-byte still meets the latency SLO. Everything about *when* to scale out or in
  is a fraction of this one number.
- **`T_total`** — how long it actually takes, wall-clock, from asking for one more instance to
  that instance serving traffic. This sets how much headroom the scale-out threshold needs to
  reserve, and how long to wait before scaling in again.

Once both are measured, `plan` is pure arithmetic — no further judgment calls, which is the
point: change the instance type or the model, and both numbers get re-measured by the same
two commands rather than re-derived by hand.

## Worked example: kokoro-82m on ml.g5.xlarge

This is real output, not illustrative. `packages/speech-infra/artifacts/` has committed
`tts-bench` output from a real run against the live `speech-kokoro-82m` endpoint — the
`qmax` and `ttotal` artifacts below are that run's actual measurements, and `tts-bench plan`
was re-run against them (read-only, no AWS calls) to reproduce this output live.

### `qmax` — the concurrency ladder

Held at each rung for 240s, only the trailing 60s measured, frozen and pinned at one
instance:

| concurrency | p95 TTFAB | meets 3000ms SLO? |
|---|---|---|
| 1 | 65ms | yes |
| 5 | 286ms | yes |
| 10 | 568ms | yes |
| 20 | 1129ms | yes |
| 30 | 1690ms | yes |
| 40 | 2245ms | yes |
| 50 | 2806ms | yes |
| 60 | 3375ms | **no** |

`Q_max = 50` — the highest rung that passed, bracketed by rung 60 actually failing (not a
lower bound). `S` (uncontended service time, at concurrency 1) came out to 61ms mean / 66ms
p95, client round-trip included.

### `ttotal` — timing one scale-out

Triggered by directly raising `DesiredInstanceCount` by one (bypasses the deployed policy's
own detection lag, which is bounded separately rather than measured):

| Stage | Duration | Share |
|---|---|---|
| capacity request -> instance visible in logs | 181s | 69% (EC2 provision + image pull) |
| container start -> weights ready | 6.5s | 2% |
| weights ready -> warmup done | ~1s | <1% |
| warmup done -> in service | 21s | 8% |
| in service -> traffic actually recovered | 47s | 18% |

`T_total = 260s` measured, **320s** planned against — the extra 60s is the deployed policy's
own detection lag (alarm period × evaluation periods + cooldown), bounded from configuration
rather than measured, because this run bypassed the live policy on purpose.

### `plan` — the arithmetic

```
uv run tts-bench plan \
  --qmax   artifacts/qmax-kokoro-82m-response-stream-g5xlarge-8b1b96bc.json \
  --ttotal artifacts/ttotal-kokoro-82m-force-desired-g5xlarge-8b1b96bc.json \
  --peak-rps 5000 --trough-rps 10 --max-scaling-per-t-total 1.25
```

```
  derived (none of these can be set independently):
    C_scale_max      37.50 concurrent — scale out here (0.75 x Q_max)
    C_scale_min      25.00 concurrent — scale in here (0.50 x Q_max)
    W_max            2.93s queueing budget = 3.0s SLO - 0.066s p95 service
    utilization      97.4% at C_scale_max
    in CW units      37.50 = C_scale_max x 1.00 — THIS is what deploys
```

Two `WARN` findings came back on this real plan — worth knowing these exist, not bugs in the
run:

- **`surge_survival`** — 97.4% utilization at the scale-out point means very little drain-rate
  headroom is actually left, despite the 25% surge ratio nominally reserved.
- **`scale_in_safety`** — scale-in is only flap-free from 3 instances up; this plan's floor of
  1 makes a 2->1 scale-in land the survivor right back at `Q_max`.

Both are exactly what's deployed today in `packages/speech-infra/src/speech_infra/config.py`
(`kokoro-82m`'s `scaling_target_value=37.5`, `scale_in_threshold=25.0`) — the warnings are
known, accepted trade-offs, not defects `plan` failed to catch.

## The three standalone comparison commands

Not part of the scaling chain above — these compare models against each other rather than
producing a deployable config for one:

| Command | Measures |
|---|---|
| `tts-bench latency` | p50/p90/p99 time-to-first-audio-byte per model, uncontended |
| `tts-bench scalability` | throughput and latency at a fixed set of concurrency levels |
| `tts-bench cost` | cost per million characters synthesized |
