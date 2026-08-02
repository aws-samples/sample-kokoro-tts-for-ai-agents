# Autoscaling Capacity Model

How `tts-bench` turns three measurements and a stated load into an autoscaling config.
Written to be rerun: change the instance type, the transport, or the model, and every
number below is re-measured rather than re-reasoned.

## The variables

### Stated (assumptions — you supply them)

| Variable | Flag | Why it matters |
|---|---|---|
| **SLO** | `--ttfab-slo-ms` (3000) | The **whole** first-byte promise: queue wait *plus* service. Everything about queueing follows from it. |
| **k** | `--growth-factor-k` (2.0), `--sweep-k` | Growth *rate*: the factor traffic may multiply by within one `T_total`. Sets how much headroom stands idle. Not measurable without production history — sweep it. |
| **peak / trough** | `--peak-rps`, `--trough-rps`, `--peak-streams`, `--trough-streams` | Growth *level*. Peak sets `max_instances`, trough sets `min_instances` — the floor reserved capacity pays for around the clock. |
| **derate** | `--derate` (0.875) | Fraction of the measured limit we target, for jitter margin. |

`k` and `peak` are different questions: *how fast* traffic grows vs. *how high*.

### Measured

| Variable | Source | Why it matters |
|---|---|---|
| **`C_max`** | `cmax` | Concurrent requests **one instance** serves at its limit. The divisor in every fleet size. |
| **`S`**, **`S_p95`** | `cmax`, free | Mean and p95 uncontended service time. The converter between concurrency and rps, and what the SLO subtracts. |
| **`T_total`** | `ttotal` | Seconds from scale-out decision to an instance serving at recovered p95, **by stage**. Sets cooldowns and how much headroom must stand ready. |

### Derived (outputs — no flag sets them)

| Variable | Equation | Why it matters |
|---|---|---|
| **`W_max`** | `SLO − S_p95`, clamped at 0 | The queueing budget. **Derived, never stated.** 0 means the model's own tail already misses the SLO — infeasible, and no fleet size fixes it. |
| **`Λ_cap`** | `C_max / S` | One instance's max sustainable arrival rate. |
| **`Q_max`** | `Λ_cap × W_max` | Per-instance queue depth: how many may wait and still be served on time. |
| **`C_target`** | `min(C_max × derate / k, W_max/S + 1)` | What the scaling policy tracks. Float — kokoro's is sub-1. |
| **`W_absorbed`** | `W_max / (k−1)` | Scaling lag the queue hides from clients. `∞` at k=1. |
| **`headroom_lag`** | `max(T_total − W_absorbed, 10s)` | The part of the lag standing headroom must cover. |

## Why `W_max` is derived and not an input

It used to be its own flag, `--max-added-wait`, sitting beside the budget with nothing
relating them. That is how kokoro shipped a config promising 300 ms while allowing a 20 s
queue — a request could take 20.2 s to first byte while the config claimed 0.3 s. One
derived field cannot contradict the promise; two independent ones always can. The flag is
retired, not aliased: `plan` rejects it.

## Two things the SLO does *not* do

**It does not move `C_target`.** `c_slo_cap(2.83s, S=0.110) = 26.7` concurrent — orders of magnitude
above kokoro's `C_target` of 0.71. Surge headroom binds, not the wait budget. **The SLO
moves `Q_max`**: at 3 s it is 41; at 4 s, 56; at 10 s, 145.

**It is not the budget `C_max` is read at.** `--ttfab-budget-ms` (300) is a *column
selector* over a ladder that sweeps 300/500/1000/3000 in one run. The tight column is the
conservative number to size a fleet from even when the client promise is looser; reading
the knee at 3000 ms reports a `C_max` inflated by accumulated backlog.

## Two kinds of `C_max` — plan on the lower

- **Latency knee** — p95 TTFAB crossed the budget.
- **Throughput ceiling** — the server stopped keeping up: `max_sustained_rps × S`.

When the ceiling binds, the knee's higher concurrency was **queue, not capacity**, and
planning on it sizes a fleet for capacity the instance does not have. `binding_c_max()`
takes the lower and records which; the report names it. A ladder that ran out while still
passing yields a **lower bound**, which over-sizes the fleet — safe, but wrong, and
flagged.

Beware the unit: the knee's concurrency is *observed* in-flight (queue residence
included); the ceiling's is *useful* concurrency (`max_sustained_rps × S`). On kokoro at
target 2.0 these read 2.93 vs. 1.38 — a 2.1× gap that is pure backlog, widening to 6× at
target 3.0. **The deployed policy tracks the observed figure** — the AWS-managed alarm on
`ConcurrentRequestsPerModel` uses the `Maximum` statistic over 3 × 10 s periods (the
dashboard metric in `observability.py` uses `Average`; the alarm does not) — so the two
must not be interchanged.

## The deliverable

```
C_target             = min(C_max × derate / k, c_slo_cap)   ← whichever binds
scaling_target_value = C_target                             ← C_max, k
queue_max_depth      = Λ_cap × W_max                        ← C_max, S, SLO
peak_n               = ⌈peak × S / C_target⌉                ← peak
trough_n             = ⌈trough × S / C_target⌉              ← trough
min_instances        = max(--min-floor, trough_n)           ← what reserved capacity pays for
max_instances        = max(min_instances, peak_n)           ← below it, peak misses the SLO
scale_out_cooldown_s = clamp(T_total / 2, 10, 30)           ← T_total
scale_in_cooldown_s  = max(300, 3 × T_total)                ← T_total
```

`C_target` takes the **lower** of two ceilings and the plan records which bound:
`surge_headroom` (`C_max × derate / k`) or `slo_wait_budget` (`c_slo_cap = W_max / S + 1`,
since at concurrency `C` on a one-at-a-time server the last arrival waits `(C−1) × S`). On
kokoro the surge term binds by a wide margin — which is the point of the section above.

With `--peak-streams` instead of `--peak-rps`, the `× S` conversion is bypassed: a
long-lived bidi session is not a request, so streams divide straight into `C_target`.

Cooldowns are asymmetric on purpose. Scale-out is short and deliberately **not** `T_total`:
target tracking adds one instance at a time, so a cooldown as long as the lag makes a surge
needing three instances take three lags to answer. Scale-in is long: capacity removed takes
a full `T_total` to get back, so scaling in early trades a saved dollar for a missed SLO.

### Worked example — kokoro-82M, bidi, `ml.g5.xlarge`

Measured: `C_max` 1.63, `S` 0.110 s, `S_p95` 0.174 s. Stated: SLO 3 s, k 2, derate 0.875.

```
Λ_cap    = 1.63 / 0.110          = 14.84 rps per instance
W_max    = 3.000 − 0.174         =  2.83 s
Q_max    = 14.84 × 2.83          = 41 requests per instance
C_target = 1.63 × 0.875 / 2      =  0.71 concurrent
W_absorbed = 2.83 / (2 − 1)      =  2.83 s of lag hidden by the queue
```

`C_target` is sub-1 — one instance per in-flight request, plus reserve. `W_absorbed` is
2.8 s against a `T_total` of minutes, so **standing headroom carries essentially all the
lag**; the queue buys almost none of it. That is the coupling that makes both measurements
feed one plan, and it is why k is worth sweeping.

### Status: `Q_max` is computed but not yet enforced

`queue_max_depth` is stored in `ModelEndpointConfig` (41 for kokoro) and checked for drift
by `plan`, but **no container reads it today** — the bounded deadline queue that would
reject beyond it is not built. Until it is, `Q_max` is a sizing statement, not a control:
the number says how deep a queue the SLO tolerates, and nothing stops a deeper one forming.
The 2.93-vs-1.38 backlog gap above is exactly what an unbounded queue looks like.

## Independent bound: the 60 s platform ceiling

SageMaker hangs up on an invocation at 60 s regardless of the queue. Since the request
deadline *is* the SLO, this caps what you may promise — `plan` refuses an SLO above it
outright rather than emitting a config that cannot hold.

## How the data is gathered

### `C_max` and `S` — `tts-bench cmax`

```
uv run tts-bench cmax --model kokoro-82m --transport bidi \
  --target-concurrency 1,1.25,1.5,1.75,2,2.5,3 --ttfab-budgets 300,500,1000,3000 \
  --hold 240 --runs 3 --max-workers 64 --require-frozen --cloudwatch
```

A step-and-hold ladder at fixed arrival rate per step, Poisson by default.

- **Freezes autoscaling and pins capacity at 1** for the whole run, restored on exit
  including Ctrl-C. `C_max` is *per-instance*: a fleet that grows mid-run silently returns
  `N × C_max`. `--require-frozen` refuses to start otherwise.
- **Measures only the trailing window** of each step, so warm-up and queue drain are
  excluded, with an idle settle between steps.
- **All budgets from one ladder** — choosing an SLO becomes a table lookup, not another
  93-minute run.
- **`--transport bidi`** measures the protocol production uses. Not a refinement of
  `response-stream` but a separate measurement: kokoro holds its inference lock across a
  whole bidi session, so expect a lower `C_max`.
- **`--max-workers`** raises the client pool above the derived size. Needed on bidi, where
  in-flight overshoots the target by the server's backlog; without it a skipped dispatch
  means the *client* ran out, not the server, and the throughput ceiling is a client
  artifact.
- **`--runs 3`** exposes run-to-run spread; a single pass warns as unrepeatable.
- **`--dry-run`** prints the schedule and wall clock without touching AWS.

Writes an artifact stamped with a **configuration fingerprint** (instance type, image,
variant). `plan` refuses to pair artifacts whose fingerprints disagree.

### `T_total` — `tts-bench ttotal`

Scales out and timestamps the whole timeline, each stage from a named source:

```
load_applied → metric_published → alarm_fired → activity_started → instance_logging
  → container_started → weights_fetched → framework_init → weights_ready → warmup_done
  → ready → in_service → traffic_recovered
```

Staged because the stages have different owners: the first four are policy and metric
latency, `instance_logging` bounds EC2 provision plus image pull from outside (a container
cannot time its own pull), the middle ones are the image's own markers, and only
**provision** changes under reserved capacity — which is why `plan` sweeps `--provision-s`
rather than trusting the measured value.

`traffic_recovered` is **bounded, not measured**: SageMaker does not reveal which instance
served a request, so "the new instance took real traffic" is inferred from the client side
as sustained p95 back inside budget. Bounded stages are flagged `bounded=True` with the
reason — a plan may build on a bound but must not mistake one for a measurement.

Two triggers: `--trigger force-desired` (direct capacity change — attributes a timeout to
the endpoint rather than the policy) and `--trigger drive-load` (real policy, real alarm
latency). `T_total` is reported both from `load_applied` and from `metric_published`, since
the first includes detection latency the policy cannot avoid.

Recovery is thresholded at the **measured budget**, not the SLO — the one place these come
apart. Recovery means "p95 came back", so the threshold must sit *between* the overloaded
p95 and the recovered one. Kokoro overloaded reaches 818 ms, already inside 3000 ms:
threshold there and every run reports recovery at the instant the instance came into
service, having measured nothing.

### The plan — `tts-bench plan`

```
uv run tts-bench plan --measured artifacts/cmax-....json --ttotal artifacts/ttotal-....json \
  --peak-rps 20 --trough-rps 2 --sweep-k 1,2,3,5 --provision-s 60,600
```

Reads artifacts, touches no AWS, cheap to repeat. Prints inputs (naming which `C_max` bound
and how firmly), a paste-ready `ModelEndpointConfig` block, and findings. Compares the
*deployed* `queue_max_depth` against what the SLO now implies and warns on drift — it is the
one derived number `config.py` stores rather than computes, so it can go stale silently, and
it did.

### Preflight — `tts-bench drift`

Read-only reconciliation of deployed autoscaling against config: orphaned targets and
policies, inert policies, capacity mismatches, leftover suspensions from an aborted
benchmark. Run before any `cmax`: an orphaned policy can move capacity no template
describes, and an unguarded run then measures a fleet instead of an instance.

## Rerunning on a new configuration

1. `drift` — confirm nothing unmanaged can move capacity.
2. `cmax` — new `C_max`, `S`, fingerprint.
3. `ttotal` — new `T_total` (image size and model load dominate; provision is swept).
4. `plan` — same stated load, new config block.

Nothing in the model is specific to kokoro or to g5. Change the instance type and all three
measurements are re-collected by the same three commands; the equations are unchanged.
