# tts-bench

Turns measurements of a deployed endpoint into an autoscaling configuration.

The requirement this package exists to serve: **if we deploy a different instance type,
reconfiguring is a rerun of these commands, not a re-derivation by hand.** Before it, the
path from a measurement to a deployed policy ran through hand-arithmetic in a comment
block in `config.py` — which is how four scaling numbers ended up deployed that nobody
could re-derive.

This README is the operator recipe for that rerun. For *why* the numbers relate the way
they do — the variables, which are chosen and which are measured, and the equations
between them — read
[`docs/autoscaling-capacity-model.md`](../../docs/autoscaling-capacity-model.md) instead;
it is not repeated here.

## The recipe

Four commands, in this order. Each one prints a `Next:` line naming the one after it.

| Step | Command | Touches AWS | Wall clock | Produces |
|---|---|---|---|---|
| 1 | `tts-bench drift` | read-only | seconds | findings; exits non-zero on an ERROR |
| 2 | `tts-bench qmax` | **mutates** — suspends scaling, pins capacity | ~36 min + ~2 min settle | `artifacts/qmax-<model>-<transport>-<slug>.json` |
| 3 | `tts-bench ttotal` | **mutates** — suspends scaling, pins, then **+1 instance** | up to ~28 min | `artifacts/ttotal-<model>-<trigger>-<slug>.json` |
| 4 | `tts-bench plan` | none — reads artifacts | seconds | a `ModelEndpointConfig` block to paste |

Then `cdk diff`, one `cdk deploy`, and `tts-bench drift` again.

Run everything from `packages/speech-infra/`, so artifacts land in
`packages/speech-infra/artifacts/` — the directory CDK reads at synth time via
`speech_infra.measurements`. Running from the repo root writes artifacts to
`<root>/artifacts/` (a different directory that CDK never reads), which would cause
`cdk synth` to silently omit all autoscaling infrastructure.

`speech_infra.config` is also resolvable from `packages/speech-infra/` — `drift` and
`plan` both require it.

---

## 1. `drift` — preflight, before anything that costs money

```bash
uv run tts-bench drift
```

Read-only reconciliation of deployed autoscaling against `config.py`: orphaned targets and
policies, inert policies, capacity mismatches, and leftover suspensions from an aborted
benchmark.

**Run it first, every time.** An orphaned scaling policy elsewhere in the account can move
capacity mid-run — and a `Q_max` ladder whose fleet grew silently reports `N x Q_max` with
nothing in the number marking it. That is why an orphan is an ERROR and not a warning:
`cdk deploy` will never remove it.

Good run: `No drift: deployed autoscaling matches config.`, followed by the `Next:` hint.
Anything else prints `[ERROR]`/`[WARN]` findings with a `fix:` line each. `--fail-on-error`
is the default, so this gates a deploy in CI; `--output findings.json` writes them.

## 2. `qmax` — the concurrency ladder

```bash
uv run tts-bench qmax --model kokoro-82m --transport response-stream --cloudwatch --require-frozen
```

Measures `Q_max`: the largest concurrency (queued + executing) whose p95 first-byte time
still met `--slo-ms`. Closed-loop and held exactly — each of N workers issues its next
request only when its previous one returns — so the answer is a concurrency that was
actually run, not one inferred from a rate. Runs frozen and pinned at one instance
(`--pin-to 1`), so the number is per-instance.

`--dry-run` prints the schedule and the wall-clock estimate without touching AWS. Use it
before committing 36 minutes and an endpoint freeze.

### The `--concurrency` list is not arbitrary

Default `1,5,10,20,30,40,50,60`. Three of those rungs are consumed downstream, and
**dropping one breaks a later command**:

| Rung | Consumer | What breaks without it |
|---|---|---|
| `1` | `plan` → the `FirstChunkLatencyP95` alarm threshold | `plan` prints `NO THRESHOLD`; the alarm has no measured service time to fire on, and the SLO cannot stand in for it |
| `5` and `10` | `ttotal`'s recovery test | `ttotal` **refuses to run**: recovery is the probe's p95 at 10 halving into its value at 5, and both sides must be measured |

`qmax` warns at startup when any of the three is missing, because by report time the
40 minutes are already spent.

The ladder must also **bracket** its answer: at least one rung above `Q_max` has to be
measured and actually miss the SLO. Otherwise the result is a lower bound (see
troubleshooting).

### Good run

Read these, in the order the command prints them:

- The ladder table — `slo` reads `ok` up to `Q_max` and `OVER` above it, and the `note`
  column is empty. `over SLO` is a real crossing; anything else in `note` means the rung
  was excluded and did not inform the answer.
- The `cw_max` column is populated, not `n/a`. That is the CloudWatch join landing.
- `Q_max: N concurrent per instance` with **no** `NOTE: LOWER BOUND` under it.
- `p95 at c=1: Xms (FirstChunkLatencyP95 alarm threshold)`.
- `Recovery pair for ttotal: c=5 -> Xms, c=10 -> Yms` — not `incomplete`.
- No `WARNING:` about the container's queue bound or about `not safe to read as
  per-instance`.

## 3. `ttotal` — force one scale-out and time it

```bash
uv run tts-bench ttotal --model kokoro-82m --transport response-stream \
  --qmax artifacts/qmax-kokoro-82m-response-stream-g5xlarge-<slug>.json
```

Holds a saturating probe at `--probe-concurrency` (10), raises `DesiredInstanceCount` by
one, and attributes the lag stage by stage — provisioning, container startup, in-service,
traffic recovery — each from the API that timestamps it.

**It runs after `qmax` because both of its latency references come off that ladder.**
`--qmax` is required: recovery is judged by the probe's p95 halving from the ladder's `c=10`
value into its `c=5` value, and there is no other measured level to compare against.
`--transport` must match the ladder's, or the run is refused — the containers serialize
differently per transport, so a p95 from one is not the level a probe on the other reaches.

`--trigger force-desired` is the only mode. Driving load past the deployed policy instead
let the policy add instances *during* the measurement. What that omits — the policy's own
detection lag — is bounded arithmetically from the alarm's periods and cooldown and
reported beside the measurement rather than folded into it.

### Good run

- `Xs from capacity requested to traffic recovered` — not `not measurable from the stages
  observed`.
- No `FLOOR: recovery was never bounded`. A floor stops at `in_service`, which is earlier
  than the new instance serving, so it under-reports the lag.
- `probe p95 TTFAB <before> -> <after>`, checked against the `ladder expected ...` line
  under it: before should sit near the `N=10` rung, after at or under the target.
- A stage breakdown with a `dominant stage:` line — that is the stage to attack to shrink
  `T_total`. `~` marks a stage bounded by inference rather than timestamped.

## 4. `plan` — derive the thresholds

```bash
uv run tts-bench plan \
  --qmax   artifacts/qmax-kokoro-82m-response-stream-g5xlarge-<slug>.json \
  --ttotal artifacts/ttotal-kokoro-82m-force-desired-g5xlarge-<slug>.json \
  --peak-rps 40 --trough-rps 4 --max-scaling-per-t-total 1.25
```

Reads artifacts, touches no AWS, cheap to repeat. Use `--peak-rps` for response-stream
traffic — requests are short-lived, so the `lambda × S` Little's-law conversion applies.
`--peak-streams` is the alternative for long-lived bidi sessions, where a stream is the
unit and no conversion is needed. One of the two is required, because the fleet size is what
a plan is for.

`--ttfab-slo-ms` defaults to whatever the ladder was measured against. A different value is
refused rather than converted — `Q_max` is *defined* by the SLO.

### Good run

- `Findings:` contains no `[STOP]`. `plan` exits non-zero on an infeasible plan, so it can
  gate a deploy.
- The `in CW units` line reads `X.XX = C_scale_max x Y.YY ... — THIS is what deploys`, not
  `UNAVAILABLE`.
- The config block has a real `scaling_target_value=` and `scale_in_threshold=`, not the
  commented-out `# scaling_target_value=?` form.
- `FirstChunkLatencyP95 alarm: Xms`, not `NO THRESHOLD`.
- Any `WARNING:` after the block naming a deployed value that disagrees is the point of the
  run, not a failure — the block above it carries the right number.

Then paste the block over the model's entry in
[`packages/speech-infra/src/speech_infra/config.py`](../speech-infra/src/speech_infra/config.py)
(`TTS_MODEL_CONFIGS`) and deploy it.

```bash
uv run speech-infra diff   kokoro-82m    # confirm only the scaling fields moved
uv run speech-infra deploy kokoro-82m
```

```bash
uv run tts-bench drift          # the policy that is live is the policy computed
```

---

## Both measurement commands mutate AWS

`qmax` and `ttotal` suspend Application Auto Scaling and pin `DesiredInstanceCount`;
`ttotal` then raises it by one. Restoration runs from a `finally` **and** from the freeze
context manager's `__exit__`, which is written as a class precisely so it fires for
`KeyboardInterrupt` and `SystemExit`. Ctrl-C is safe.

> **Do not wrap these commands in an outer `timeout`.** A signal that kills the process
> before its own cleanup runs leaves the endpoint with scaling suspended and, after a
> `ttotal`, billing at the raised instance count. Let the command finish, or Ctrl-C it.

After a hard kill:

```bash
uv run tts-bench thaw --endpoint speech-kokoro-82m
```

Idempotent, so running it on a healthy endpoint is a no-op. Capacity is **not** restored by
default — after a hard kill nobody knows what the desired count was before the freeze, and
guessing 1 could shrink a fleet that was legitimately larger. Pass
`--restore-capacity --desired N` to set it explicitly.

## `--cloudwatch` on `qmax` is effectively mandatory

It is on by default. Keep it on.

The client's occupancy and the statistic the deployed alarm reads
(`ConcurrentRequestsPerModel` / `Maximum` / 10 s) are **different quantities whose ratio
moves with load** — across one kokoro ladder they ran from 9.8x apart to 1.35x apart. The
ladder measures both and records the conversion per rung, which is the only reason `plan`
can emit a threshold in the units that actually deploy. Deploying the unconverted client
figure is how a target value no traffic can satisfy reached a live endpoint.

**10 s high-resolution datapoints retain only 3 hours**, so the conversion has to be
captured during the run. It cannot be reconstructed afterwards. Without it, `plan` comments
`scaling_target_value` out rather than emitting an unconverted number — a config that fails
to parse is a better outcome than one that deploys a number nothing can satisfy.

## `Q_max` must be measured against an unbounded queue

`--require-unbounded-queue` (on by default) machine-checks the live container env for
`MAX_QUEUE_DEPTH` and `MAX_PENDING_REQUESTS`. A container with a depth bound stops
accepting work *before* p95 crosses the SLO, so the ladder would find that bound and report
it as a capacity number — and it looks entirely normal, since rejections land in the outcome
counts rather than in the latency percentiles the pass/fail line reads.

The order matters: **measure unbounded, then deploy the bound.** Enforcing `Q_max` at the
instance is the point of measuring it.

## Every artifact carries a configuration fingerprint

Instance type, image digest, and container env, read off the endpoint at run time and
stamped onto the artifact. It also names the file — `...-g5xlarge-139b9068.json` — so
measuring a second configuration cannot overwrite the first by forgetting `--output`.

`ttotal` refuses a `--qmax` artifact whose fingerprint disagrees with the live endpoint, and
`plan` refuses to pair two artifacts whose fingerprints disagree with each other. A ladder
measured on one instance type therefore cannot be paired silently with a lag measured on
another. **This is the mechanism that makes "just rerun it" trustworthy.**

`--allow-config-mismatch` on `ttotal` and `plan` downgrades that refusal to a warning. An
artifact carrying *no* fingerprint counts as a mismatch, not a pass — accepting those
silently is the hole this closes.

## Quota and cost

The account quota `ml.g5.xlarge for endpoint usage` (`L-1928E07B`) is **4**, account-wide
and per-region, counting every endpoint including other teams'. That is why kokoro's
`max_instances` is 4 today, and why **a `ttotal` run needs one free slot** — it raises the
count by one. With the quota full, SageMaker rejects the change with
`ResourceLimitExceeded`, the endpoint stays `InService` at its old count, and nothing
surfaces there: a scale-out that cannot happen looks exactly like one that has not happened
yet. Check free slots before starting, or `ttotal` burns most of `--max-wait` finding out.

These runs hold a GPU endpoint under sustained load for tens of minutes — a default `qmax`
ladder is ~36 minutes plus a ~2 minute CloudWatch settle, and `ttotal` briefly **doubles
the fleet**. At `ml.g5.xlarge` on-demand that is real money per attempt. `--dry-run` on
`qmax` costs nothing; use it to check the command first.

## Troubleshooting

**No scale-out within `--max-wait`.** If the endpoint is still `Updating`, SageMaker
accepted the change and cannot place the instance. With quota free that points at EC2
instance capacity for the type, not at anything in the scaling config — the fix is another
instance type or another region, **not a longer `--max-wait`**. Check
`aws sagemaker describe-endpoint` for a `FailureReason`. If it is back to `InService` at the
old count, SageMaker abandoned the change without recording a failure anywhere.

**`Q_max` ladder ended while still passing.** `NOTE: LOWER BOUND — no rung above it was
measured to actually miss the SLO`. That is a lower bound, not a value: both thresholds are
fractions of `Q_max`, so the plan scales out earlier than necessary. Re-run with
`--concurrency` extended past the reported number.

**Probe never reached its expected pre-scale p95.** The note reads *the probe's pre-scale
p95 was already at or under the recovered level*. This is a **bad probe, not a failed
recovery** — the halving had nothing to detect, so `traffic_recovered` collapses onto
`in_service` and the run is a tautology. The fix is the opposite of the one above: raise
`--probe-concurrency` and add the matching rungs (N and N/2) to the `qmax` ladder. Do not
read that `T_total`.

**`plan` commented `scaling_target_value` out.** The ladder recorded no
`ConcurrentRequestsPerModel` / `Maximum`, so there is no conversion to apply. Re-run `qmax`
**with** `--cloudwatch`; the 3-hour retention means it cannot be backfilled.

**`plan` refuses to pair the artifacts.** The fingerprints differ — usually one artifact
predates a redeploy. Re-measure the stale half, or pass `--allow-config-mismatch` if the
difference is genuinely irrelevant to what you are planning.

**Endpoint left suspended after a kill.** `tts-bench drift` reports a `suspended` finding
with `fix: tts-bench thaw --endpoint <endpoint>`. Run that. If it still warns that scale-out
is suspended afterwards, check IAM permissions.

## Other commands

Not part of the scaling chain; they compare models rather than measuring one configuration.

| Command | Purpose |
|---|---|
| `tts-bench latency` | p50/p90/p99 TTFAB per model, uncontended |
| `tts-bench scalability` | throughput and latency at fixed concurrency levels |
| `tts-bench cost` | cost per million characters |
