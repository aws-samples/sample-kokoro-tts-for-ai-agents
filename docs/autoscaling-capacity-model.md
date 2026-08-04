# Autoscaling Capacity Model

How `tts-bench` turns two measurements and a stated load into an autoscaling config.
Written to be rerun: change the instance type, the transport, or the model, and every
number below is re-measured rather than re-reasoned.

Six variables. Two are chosen, two are measured, two are derived, and **nothing else is
settable** — which is the whole design. The previous version of this model had a
`C_max`, a `derate`, a `c_slo_cap` and a `W_absorbed`, and none of those were measurable
on this container: kokoro's `serve.py` holds an `asyncio.Lock` across a whole request, so
it serves one at a time and in-flight concurrency has no ceiling to find. The "`C_max`
1.63" every one of those numbers divided by was M/M/1 queue occupancy, `L = ρ/(1−ρ)`, at
whatever utilization one ladder step happened to sit at. What *is* bounded is the **wait**.
So the model bounds the wait, and the layer that divided by a fiction is gone.

## The variables

### Chosen (assumptions — you supply them)

| Variable | Flag | Why it matters |
|---|---|---|
| **SLO** | `--ttfab-slo-ms` (3000) | The **whole** first-byte promise: queue wait *plus* service. It is the pass/fail line the `Q_max` ladder walks, so it does not merely constrain `Q_max` — it *defines* it. |
| **`max_scaling_per_T_total`** | `--max-scaling-per-t-total` (1.25) | Surge ratio: the factor traffic may multiply by within one `T_total`. Both thresholds are fractions of `Q_max` set by it. Not measurable without production history. |
| **peak / trough** | `--peak-streams`, `--trough-streams`, `--peak-rps`, `--trough-rps` | Growth *level*, as against the surge ratio's growth *rate*. Peak sets `max_instances`, trough sets `min_instances` — the floor reserved capacity pays for around the clock. |

The surge ratio and the peak are different questions: *how fast* traffic grows vs. *how
high*. Neither is derivable from the other.

### Measured

| Variable | Source | Why it matters |
|---|---|---|
| **`Q_max`** | `tts-bench qmax` | The largest concurrency — **queued + executing** — at which p95 first byte still lands inside the SLO. Both thresholds and the admission bound are fractions of it. |
| **`S`**, **`S_p95`** | `qmax`, free at the `N=1` rung | Mean and p95 uncontended service time as a client sees it, client round trip included (~34 ms against kokoro). Converts rps into concurrency, and `S_p95` is what the SLO subtracts. |
| **`T_total`** | `tts-bench ttotal` | Seconds from a capacity request to an instance actually serving traffic, **by stage**. Sets both cooldowns, and the surge ratio is a ratio *per `T_total`*, so the whole policy scales with it. |

### Derived (outputs — no flag sets them)

With `h = max_scaling_per_T_total − 1`, the headroom fraction to keep in reserve:

| Variable | Equation | Why it matters |
|---|---|---|
| **`C_scale_max`** | `(1 − h) × Q_max` | Add an instance here. The fleet still holds `h` of queue headroom when the request for capacity goes out. |
| **`C_scale_min`** | `(1 − 2h) × Q_max` | Remove an instance here — one surge of *excess* headroom, so scale-in needs the load to have really gone rather than merely dipped. |
| **`W_max`** | `SLO − S_p95`, clamped at 0 | The queueing budget. **Derived, never stated.** 0 means the model's own tail already misses the SLO — infeasible, and no fleet size fixes it. |

`shared.capacity.scale_thresholds` refuses `h ≥ 0.5`: at 1.5 the scale-in threshold reaches
zero and beyond it goes negative, which deploys as a policy that never scales in. A
plausible-looking number that silently disables half the policy is refused rather than
clamped.

## Why `W_max` is derived and not an input

It used to be its own flag, `--max-added-wait`, sitting beside a second latency field with
nothing relating them. That is how kokoro shipped a config promising 300 ms while allowing
a 20 s queue — a request could take 20.2 s to first byte while the config claimed 0.3 s.
One derived field cannot contradict the promise; two independent ones always can.

For the same reason there is now **one** latency field, `ttfab_slo_ms`. The old
`ttfab_budget_ms` is gone, not aliased. Where a consumer genuinely needs a *tighter*
threshold than the promise — the `FirstChunkLatencyP95` alarm, which watches service time
on an instance already serving and so has spent none of the queue allowance — it reads the
ladder's own `N=1` rung (`ttfab_p95_at_c1_ms`, via `speech_infra.measurements`). Measured
on every rerun, not hand-set.

## Two things the SLO does *not* do

**It does not select a column.** There is no multi-budget curve to read a knee off any
more. One ladder, one SLO, one answer — and `plan` refuses a `Q_max` artifact measured
against a different SLO rather than re-reading it. A different SLO is a `qmax` rerun.

**It is not the `FirstChunkLatencyP95` threshold.** At an SLO-sized 3000 ms that alarm
fires only once the endpoint is roughly 10x past keeping up, which is not an alarm.

## What "concurrency" means, and in whose units

Every number in this model that is called a concurrency means **queued + currently
executing**. That is deliberate, and it is what makes the trigger free: CloudWatch's
`ConcurrentRequestsPerModel` already counts exactly that, so scaling on queue saturation
needs no new metric and no container-published namespace. (The previous policy tracked
`Speech/vLLM:vllm:num_requests_running`, which nothing ever published to, so its alarms
sat in `INSUFFICIENT_DATA` and could never fire.)

The unit that is *not* free is the **statistic**. A client-measured mean in-flight and
`ConcurrentRequestsPerModel` / `Maximum` over 10 s are different quantities, and their
ratio is not a constant. Across one kokoro ladder it collapsed from about **9.8x** at the
bottom to **1.35x** at the top: a lightly loaded endpoint's 10 s peak is many multiples of
its average, while a saturated one's is barely above it. So every number crossing that
boundary has to name its statistic:

| Number | Measured as | Deployed against |
|---|---|---|
| `Q_max` | client-side, closed loop — `N` workers hold `N` outstanding by construction | `queue_max_depth`, a container admission bound (not yet enforced — see below) |
| `S`, `S_p95` | client-observed first byte at the ladder's lowest rung, RTT included | `W_max = SLO − S_p95`; nothing on the server reads it |
| `scaling_target_value` | `C_scale_max` × the ladder's measured ratio | `ConcurrentRequestsPerModel` / **`Maximum`** / 10 s, via the high-resolution predefined metric (`constructs/scaling.py`) |
| `scale_in_threshold` | `C_scale_min` × the same ratio | `ConcurrentRequestsPerModel` / **`Average`** / 60 s, `evaluation_periods=3`, `datapoints_to_alarm=3` |
| `FirstChunkLatencyP95` | the ladder's `N=1` p95, in ms | `FirstChunkLatency` / **`p95`** / 60 s, threshold in µs |
| `T_total` | wall clock from our `UpdateEndpointWeightsAndCapacities` call to the probe's p95 halving | cooldowns, and the surge the thresholds reserve for |

### The defect this table exists to prevent

A **client-measured mean in-flight** compared against `Maximum` is an unsatisfiable target:
target tracking is proportional — `desired = ceil(current × observed / target)` — so a
value drawn from the wrong statistic at lightly loaded conditions can ask for `max_capacity`
instances on the first request. That is the category of defect `qmax --cloudwatch` and the
unit-conversion gate exist to close. The ratio between client mean and server `Maximum`
collapses from roughly **9.8x** at light load to **1.35x** at saturation — so the
unconverted figure is wrong by a variable factor, not a constant one, and no post-hoc
correction can recover it. The conversion has to be measured during the run, per rung.

Hence `qmax --cloudwatch` is on by default and effectively mandatory: it captures the
conversion **during** the run, per rung, and `plan` interpolates on that table rather than
on a fitted constant. With no conversion measured, `plan` comments `scaling_target_value`
out rather than emitting the raw occupancy — a config that fails to parse beats one that
deploys a number no traffic satisfies.

Two smaller consequences of the same table, recorded rather than fixed here. The
`scale_in_threshold` is converted with a ratio measured against `Maximum` but compared
against `Average`, which runs lower — so scale-in fires somewhat more readily than the
occupancy it was derived from implies, in the direction the 600 s cooldown damps. And the
diagnostic `ConcurrencyOverTarget` alarm reads `Average` against a `Maximum`-units
threshold, so it fires later than the policy it is watching; it is diagnostic only and
cannot move capacity.

## The deliverable

```
h                    = max_scaling_per_T_total − 1                ← chosen
C_scale_max          = (1 − h)  × Q_max                           ← Q_max, h
C_scale_min          = (1 − 2h) × Q_max                           ← Q_max, h
scaling_target_value = C_scale_max × cw_ratio(C_scale_max)        ← + the ladder's CW join
scale_in_threshold   = C_scale_min × cw_ratio(C_scale_max)        ← same ratio
W_max                = SLO − S_p95                                ← SLO, S_p95
queue_max_depth      = Q_max                                      ← Q_max
peak_n               = ⌈peak_streams / C_scale_max⌉               ← peak
                     = ⌈peak_rps × S / C_scale_max⌉               ← peak, S (rps form)
trough_n             = same, at the trough
min_instances        = max(--min-floor, trough_n)                  ← what reserved capacity pays for
max_instances        = max(min_instances, peak_n)                  ← below it, peak misses the SLO
scale_out_cooldown_s = clamp(T_total / 2, 10, 30)                  ← T_total
scale_in_cooldown_s  = max(300, 3 × T_total)                       ← T_total
```

The fleet is sized on `C_scale_max`, not on `Q_max`: `Q_max` is where the SLO breaks, and
sizing a fleet to sit there is sizing it to sit at the edge. With `--peak-streams` the
`× S` conversion is bypassed — a long-lived bidi session is not a request, so streams
divide straight into the threshold.

Cooldowns are asymmetric on purpose. Scale-out is short and deliberately **not** `T_total`:
target tracking adds one instance at a time, so a cooldown as long as the lag makes a surge
needing three instances take three lags to answer. It is floored at the 10 s metric period,
since a policy cannot react to data it has not received. Scale-in is long — capacity
removed takes a full `T_total` to get back, so scaling in early trades a saved dollar for a
missed SLO — and it is also what damps the flap `scale_in_safety` warns about below.

The shape this deploys into is two policies on one scalable target: target tracking on the
high-resolution predefined metric with `disable_scale_in=True`, plus a step policy owning
scale-in. Application Auto Scaling takes the *maximum* of scale-out recommendations, so
they compose rather than fight — but only because scale-in belongs to exactly one of them.
`emergency_step_enabled` adds a steeper step-out at 2x/4x the target and stays off until a
measurement justifies it.

### Worked example — kokoro-82M, response-stream, `ml.g5.xlarge` (illustrative)

**No `qmax` or `ttotal` artifact exists yet.** The tools that produce them are new and no
ladder has been run on this configuration. The arithmetic below shows the structure with
placeholder labels; every input is marked with what it is waiting on. Do not paste any of
it into `config.py` — `plan` prints the block that belongs there.

| Input | Value used | Status |
|---|---|---|
| SLO | 3000 ms | chosen; deployed today as `ttfab_slo_ms` |
| surge ratio | 1.25 | chosen; the `--max-scaling-per-t-total` default |
| `S`, `S_p95` | — | **pending `qmax`** |
| `Q_max` | — | **pending `qmax`** |
| `T_total` | — | **pending `ttotal`** |

Once `qmax` and `ttotal` run, the structure is:

```
h            = 1.25 − 1                  =  0.25
C_scale_max  = (1 − h)   × Q_max         → scale out
C_scale_min  = (1 − 2h)  × Q_max         → scale in
W_max        = SLO − S_p95               (queueing budget)
cooldown_out = clamp(T_total/2, 10, 30)
cooldown_in  = max(300, 3 × T_total)
```

`scaling_target_value` and `scale_in_threshold` are **not** derivable here: they are
`C_scale_max × cw_ratio` and `C_scale_min × cw_ratio`, and no ladder has measured
`cw_ratio` on this configuration. That is exactly the gap `plan` refuses to paper over.

Two things fall straight out of the utilization line, and both are findings rather than
footnotes.

## Two known limits of the simple rule

The rule — scale out at `(1−h) × Q_max`, in at `(1−2h) × Q_max` — is arithmetic on one
measured number, which is its virtue. It has two limits, and each is computed on every
`plan` run rather than argued about here.

### `C_scale_max` at 0.75 × `Q_max` is already high utilization — `surge_survival`

Occupancy converts to utilization steeply: `ρ = L/(1+L)`, so occupancy 4 is 80 % utilized
and occupancy 37.5 is **97.4 %**. Three quarters of `Q_max` is therefore not three quarters
of the way to trouble; it is essentially all of it. And the surge the threshold exists to
survive is multiplicative: `1.25 × 0.974 = 1.2175`, a utilization above 1, which is
arithmetically divergent — the queue does not settle at a deeper level, it grows until
something sheds.

The reason the rule can be this wrong while looking right is a units confusion of a
different kind: `C_scale_max` reserves headroom in queue **slots**, which is a stock,
while surviving a surge is a question about drain **rate**, which is a flow. Reserving a
quarter of the stock says nothing about the flow.

So `plan` simulates it. `shared.capacity.shed_probability` runs an M/M/1 continuous-time
chain (competing exponentials, no time discretisation) from `C_scale_max` for `T_total`
seconds and counts the trials that touch `Q_max` — 200 trials on a private `random.Random`
so a caller's own seeding cannot move a published number. At `C_scale_max` of 0.75 × `Q_max`
and measured `T_total` and `S` inputs, this number lands well above the `SHED_PROBABILITY_WARN`
line of 0.1. From occupancy 4 the same simulation returns 0.0. Those are simulation outputs
at stated inputs, so both move when a real `qmax`/`ttotal` pair is measured.

Poisson arrivals are the point, not a convenience. A deterministic fluid model says a queue
below its drain rate never grows, which is precisely how one talks oneself into a threshold
at 97 % utilization. The variance is the entire risk.

The finding is a `WARN`, not a `STOP`, and its recommendation is the honest triple: scale
out at a smaller share of `Q_max`, shorten `T_total`, or hold standing headroom in
instances. What it does not do is quietly pick one.

### `C_scale_min` at 0.5 × `Q_max` flaps below three instances — `scale_in_safety`

Scale-in redistributes load rather than removing it. Dropping one of `N` instances
multiplies every survivor's concurrency by `N/(N−1)`, so the move is safe only while that
factor stays under `r = C_scale_max / C_scale_min`. At `h = 0.25`, `r = 37.5/25.0 = 1.5`,
so `N ≥ r/(r−1) = 3`:

```
3 → 2  survivors at 25.0 × 3/2 = 37.5  = C_scale_max  → scales straight back out
2 → 1  survivors at 25.0 × 2/1 = 50.0  = Q_max        → SLO breached on the way down
```

`ScaleThresholds.min_safe_instances` reports the 3. And kokoro runs `min_instances=1` — a
SageMaker real-time variant cannot scale to zero, so 1 is the floor that deploys — which
makes 2→1 the *common* case rather than the corner one. The long asymmetric scale-in
cooldown (`max(300, 3 × T_total)`) damps the resulting flap; it does not remove it. `plan` emits the smallest safe `N` and a `WARN` when
the planned floor is below it, so a fragile plan is visibly fragile.

## `max_instances` is decoupled from the thresholds

Both thresholds are per-instance occupancy fractions of `Q_max`. Neither equation contains
the fleet ceiling, and `peak_n = ⌈peak / C_scale_max⌉` runs the dependency the other way, so
`max_instances` and the thresholds do not interact at all. A sweep over the stated peak
moved `max_instances` from **2 to 421** while `C_scale_max` and `C_scale_min` stayed exactly
where they were — recorded here because the sweep code that demonstrated it has since been
deleted, so it is an observation rather than something this document re-derives. The
mechanism is re-derivable at any peak: at the illustrative `C_scale_max` of 37.5, a
2000-stream peak needs `⌈2000/37.5⌉ = 54` instances and both thresholds are still 37.5 and
25.0.

Practically: raising the ceiling for a bigger peak is **one flag**, and nothing else in the
plan moves. What does *not* follow is that the flag is enough. `max_instances=4` today sits
exactly at the account quota `ml.g5.xlarge for endpoint usage` (`L-1928E07B`), which is 4,
account-wide and per-region, counting every endpoint including other teams'. A
quota-blocked scale-out is invisible from the endpoint — the policy fires, SageMaker
refuses the *whole* request rather than granting part of it, the activity is logged
`Failed`, and `DescribeEndpoint` keeps reporting `InService` at the old count — so a quota
increase precedes any higher peak. `fixture.endpoint_quota_headroom` reads the remaining
room, and `require_scalable` turns it into a refusal; that refusal has no caller today,
because `ttotal`'s `force-desired` trigger suppresses the live policy the rest of that check
demands. So a `ttotal` run still needs its free slot checked by hand, or it spends most of
`--max-wait` finding out.

The cost of the headroom the thresholds reserve is bounded, which is the other half of why
this decoupling is safe. Both fleets are the same demand over a different divisor, so the
ratio cannot exceed `Q_max / C_scale_max = 1/(1−h)`; with `h < 0.5` refused, the continuous
ceiling is under 2x and integer rounding reaches exactly 2x and no further. `plan` reports
it as `fleet_cost` and does not warn, because there is no reachable multiple at which
"shorten `T_total` instead" becomes the cheaper advice. The old `C_max` model could reach
4x, since there the divisor moved with `k` without a bound.

## Status: `Q_max` is computed but not yet enforced

`queue_max_depth` is stored in `ModelEndpointConfig` and checked for drift by `plan`, but
**no container reads it today** — the bounded admission queue that would refuse work beyond
it is not built. Until it is, `Q_max` is a sizing statement rather than a control: the
number says how deep a queue the SLO tolerates, and nothing stops a deeper one forming.

The measurement and the control want opposite things, and the order matters. `Q_max` has to
be measured against an **unbounded** queue: a container that sheds stops accepting work
*before* p95 crosses the SLO, so the ladder would find `MAX_QUEUE_DEPTH` and report it as a
capacity number — and it looks entirely normal, because rejections land in the outcome
counts rather than in the latency percentiles the pass/fail line reads.
`qmax --require-unbounded-queue` (on by default) machine-checks the live container env for
`MAX_QUEUE_DEPTH` and `MAX_PENDING_REQUESTS` and refuses rather than warns. **Measure
unbounded, then deploy the bound.** Enforcing it in the container is the deferred next
step, and it is the point of measuring it.

## Independent bound: the 60 s platform ceiling

SageMaker hangs up on an invocation at 60 s regardless of the queue. Since `W_max` is
`SLO − S_p95`, a request that spends its whole allowance returns at exactly the SLO, so
this caps what may be promised at all: `plan` reports an SLO past the ceiling as
`INFEASIBLE` rather than emitting a config that cannot hold. A queue sized past it admits
requests that wait their entire allowance and then fail anyway, which is worse than
refusing them at admission — the client paid the wait for nothing. `--ceiling-s` moves the
limit for a stricter internal promise, never to make an infeasible plan pass. The kokoro
container's own `MAX_REQUEST_AGE_S=56` is the same bound enforced one layer down.

## How the data is gathered

Four commands, in this order; each prints a `Next:` line naming the one after it. The
operator recipe — flags, what a good run looks like, troubleshooting — lives in
[`packages/tts-bench/README.md`](../packages/tts-bench/README.md) and is not repeated here.

### Preflight — `tts-bench drift`

Read-only reconciliation of deployed autoscaling against config: orphaned targets and
policies, inert policies, capacity mismatches, leftover suspensions from an aborted
benchmark. Run it before anything that costs money. An orphaned policy can move capacity
that no template describes and that `cdk deploy` will never remove, and an unguarded ladder
then measures a fleet instead of an instance — which is why an orphan is an ERROR.

### `Q_max` and `S` — `tts-bench qmax`

```
uv run tts-bench qmax --model kokoro-82m --transport response-stream --cloudwatch --require-frozen
```

A step-and-hold ladder in **concurrency**, default rungs `1,5,10,20,30,40,50,60`.

- **Concurrency is held exactly, not aimed at.** Each rung runs `N` closed-loop workers, so
  queued + executing is `N` by construction. There is no arrival rate anywhere in the
  module — no `λ = C/S`, no offered-versus-achieved comparison. That deletion is the point:
  every units defect on this measurement path existed because a concurrency had to be
  converted into a rate first.
- **The pass condition is one line**: p95 first byte inside `--slo-ms`. `Q_max` is the
  highest rung that passed, and `q_max_bracketed` records whether a rung above it was
  measured and actually failed. Without that, a ladder that simply ran out while passing
  reports a lower bound as if it were an answer — which over-sizes the fleet: safe, wrong,
  and flagged.
- **Step and hold, never ramp** — 240 s per rung with only the trailing 60 s measured, and
  30 s idle between rungs so the previous queue drains. A ramp smears the crossing, because
  the queue built at `N` is still draining at `N+1`.
- **Frozen and pinned at one instance**, restored on exit including Ctrl-C. `Q_max` is
  per-instance; a fleet that grows mid-run silently returns `N × Q_max`. `--require-frozen`
  refuses to start otherwise.
- **Three rungs are consumed downstream.** `1` is the uncontended service-time sample *and*
  the `FirstChunkLatencyP95` threshold; `5` and `10` are the pair `ttotal`'s halving test
  compares against. Drop one and a later command has nothing to read.
- **`--runs 2` or more** gives a run-to-run spread; the cross-run `Q_max` is the
  **minimum**, which is both a real rung and the conservative one, and a spread past 20 %
  warns that the ladder resolved noise.
- **`--cloudwatch`** joins the server statistic per rung — the whole units story above.
- **`--transport response-stream`** measures the protocol production uses. Not a refinement of
  `bidi` but a separate measurement: `Q_max` does not transfer between them.
- **`--dry-run`** prints the schedule and wall clock (~36 min for the default ladder)
  without touching AWS.

### `T_total` — `tts-bench ttotal`

```
uv run tts-bench ttotal --model kokoro-82m --transport response-stream --qmax artifacts/qmax-....json
```

Holds a saturating probe, raises capacity by one, and timestamps the timeline, each stage
from a different API:

```
load_applied → desired_set → instance_logging → container_started → weights_fetched
  → framework_init → weights_ready → warmup_done → ready → in_service → traffic_recovered
```

**The clock starts at `desired_set`**, our own `UpdateEndpointWeightsAndCapacities` call —
not at `load_applied`, because how long the probe warmed up beforehand is a choice this
module makes rather than lag the fleet absorbs. Staged because the stages have different
owners: `instance_logging` bounds EC2 provision plus image pull from outside (a container
cannot time its own pull), the middle ones are the image's own markers, and **provision is
the one stage a reserved-capacity account changes** — so `plan` labels it rather than
folding it in.

**Recovery is a halving, not a threshold crossing.** SageMaker does not reveal which
instance served a request, so "the new instance took real traffic" is inferred: the endpoint
routes `LEAST_OUTSTANDING_REQUESTS`, and against a serial server outstanding *is* queue
depth, so a probe held at 10 splits 5/5 when a second instance takes traffic and its p95
drops toward the ladder's own `c=5` value (within `RECOVERY_TOLERANCE` 1.25, held for 60 s).
Both sides of that comparison are measured rungs. A probe at concurrency 1 cannot do this —
p95 is already healthy, so recovery collapses onto `in_service` and the run is a tautology.
Every inferred stage carries `bounded=True` with its reason: a plan may build on a bound but
must not mistake one for a measurement.

**What `force-desired` deliberately does not measure.** It is the only trigger, and it
bypasses the deployed policy's own detection lag — which is the point, because driving load
past a live policy let that policy add instances *during* the measurement (a misset threshold
is cleared by any probe traffic, and proportional target tracking then asks for `max_capacity`
in one jump). What the bypass omits is bounded from
the deployed policy's own configuration instead: `ttotal.POLICY_LAG_BOUND_S = 60.0` s = a
10 s metric period × 3 evaluation periods + the 30 s scale-out cooldown. It is reported
beside `t_total_s` as `t_total_with_policy_bound_s`, never added into it. **The planner
sizes against the sum** — production scales out through the policy, not through a capacity
call, so the measured half alone under-states what a surge must be absorbed across — and
the two terms stay separate on the artifact so the sum stays decomposable. A bound is never
a measurement, and the `provision_stage` finding says which is which.

### The plan — `tts-bench plan`

```
uv run tts-bench plan --qmax artifacts/qmax-....json --ttotal artifacts/ttotal-....json \
  --peak-rps 40 --trough-rps 4 --max-scaling-per-t-total 1.25
```

Reads artifacts, touches no AWS, cheap to repeat. Prints the four inputs with their
provenance, the derived numbers with the conversion beside them, every finding worst-verdict
first (`OK` / `WARN` / `STOP` / `----` for a check that never ran), a paste-ready
`ModelEndpointConfig` block, and the `FirstChunkLatencyP95` threshold. It also compares the
*deployed* `queue_max_depth` and `scaling_target_value` — in CloudWatch units on both sides
— against what this plan derives, and says so when they disagree. Those are the two derived
numbers `config.py` stores rather than computes, so they can go stale silently, and both
did.

Three refusals are hard: a `Q_max` measured against one SLO cannot be read against another;
a queueing budget past the invocation ceiling is `INFEASIBLE` rather than a warning; and
artifacts whose configuration fingerprints disagree cannot be combined.

## Rerunning on a new configuration

1. `drift` — confirm nothing unmanaged can move capacity.
2. `qmax` — new `Q_max`, `S`, `S_p95`, CW conversion, fingerprint.
3. `ttotal` — new `T_total` by stage (image size and model load dominate; provision is
   labelled separately).
4. `plan` — same chosen inputs, new config block.
5. Paste the block into `config.py`, `uv run speech-infra diff kokoro-82m`, `uv run speech-infra deploy kokoro-82m`.
6. `drift` again — the policy that is live is the policy that was computed.

**A different instance type is a rerun, not a re-derivation.** Nothing in the model is
specific to kokoro or to g5: change the type and both measurements are re-collected by the
same two commands, with the equations unchanged. That is the requirement this tooling
exists to serve, and it is enforced rather than trusted — every artifact carries a
`DeployedConfig` fingerprint (instance type, container image digest, container env) which
also names the file, so measuring a second configuration cannot overwrite the first.
`plan` refuses to pair artifacts whose fingerprints disagree, so a g5 ladder cannot be
paired silently with a g6 lag; an artifact carrying *no* fingerprint counts as a mismatch,
not a pass.

Two constraints on the rerun itself, both learned the expensive way. SageMaker refuses an
instance-type change while an Application Auto Scaling scalable target is registered on the
variant — three deploys (deregister, retype, re-register) and a window with no autoscaling;
a move to `ml.g6.xlarge` was attempted 2026-07-30 and rolled back on this rule. And
us-east-1 could not place a second `ml.g5.xlarge` for this account on three attempts, the
decisive one with quota free, no load, one instance requested, still `Updating` at 22 min
with no `FailureReason`. That is EC2 capacity for the type rather than an account limit, and
it bounds the `T_total` measurable here — which is why the provision stage is reported
separately.
