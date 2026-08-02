"""Capacity planning math for autoscaled inference endpoints.

Modality-neutral by design: nothing here knows about TTS, STT, or SageMaker.
Every function is pure so it can be unit-tested without AWS in scope.

The model is Little's Law (``concurrency = arrival_rate x service_time``) plus a
derate for the fact that autoscaling reacts *after* load arrives. If traffic can
grow by a factor ``k`` within one scaling lag ``T_total``, each instance must run
at ``1/k`` of its measured ceiling so the fleet can absorb that growth while new
capacity boots.

Symbols used throughout:

    C_max     per-instance concurrency at the latency knee (measured)
    S         mean service time, seconds (measured)
    T_total   scaling lag: metric publication -> instance serving traffic (measured)
    k         growth factor within one T_total (supplied as a scenario argument)
    SLO       end-to-end first-byte promise, seconds (stated)
    W_max     added wait a queued request may absorb, seconds (derived from the SLO)
    C_target  per-instance concurrency the scaling policy should track (derived)

``W_max`` is derived rather than chosen. An end-to-end SLO — "first byte within N
seconds, queueing included" — fixes it at ``SLO - S_p95``: whatever the promise does
not spend on service is all the queue has left. Holding it as an independent policy
knob is how a 300ms budget came to sit beside a 20s wait budget, a combination that
misses the stated SLO by 7x while every individual field looks defensible.
"""

from __future__ import annotations

import math

#: CloudWatch's finest publication granularity for high-resolution metrics.
#: No reactive policy can respond faster than this, so headroom sized below it is
#: being sized for growth we could never observe in time to act on.
CLOUDWATCH_HIGH_RES_PERIOD_S = 10.0

#: Fraction of the measured knee to actually target. The knee is where latency
#: starts degrading, so sitting exactly on it means any jitter crosses the SLO.
DEFAULT_DERATE = 0.875

#: Hard ceiling on a single SageMaker real-time invocation, seconds. Not a policy
#: choice and not tunable: the runtime closes the connection at 60s regardless of
#: what the container is doing. Everything a request spends — queueing wait plus
#: service — has to fit inside it, so it caps ``W_max`` no matter how generous the
#: SLO is. Containers set ``MAX_REQUEST_AGE_S`` a few seconds below to shed a
#: doomed request rather than have the client see a truncated stream.
SAGEMAKER_INVOCATION_CEILING_S = 60.0


def request_deadline_s(max_added_wait_s: float, s_p95_s: float) -> float:
    """Worst-case wall-clock a queued request occupies, seconds.

    ``W_max + S_p95``: the wait it absorbs in the queue plus the service it then
    receives. Judged against :data:`SAGEMAKER_INVOCATION_CEILING_S` rather than
    against the TTFAB budget — TTFAB is when audio *starts*, while this is when
    the invocation *ends*, and only the latter is what the runtime times out.

    ``s_p95_s``, not ``s_mean_s``: a mean-sized deadline is missed by half the
    requests that reach it, which is not a deadline.

    Raises:
        ValueError: If ``s_p95_s`` is not positive or ``max_added_wait_s`` is
            negative.
    """
    if s_p95_s <= 0:
        raise ValueError(f"s_p95_s must be positive, got {s_p95_s}")
    if max_added_wait_s < 0:
        raise ValueError(f"max_added_wait_s must be non-negative, got {max_added_wait_s}")
    return max_added_wait_s + s_p95_s


def fits_invocation_ceiling(
    max_added_wait_s: float,
    s_p95_s: float,
    ceiling_s: float = SAGEMAKER_INVOCATION_CEILING_S,
) -> tuple[bool, float]:
    """Whether a queued request can finish before the platform hangs up on it.

    Returns:
        ``(fits, deadline_s)``. When ``fits`` is false the design is infeasible
        rather than merely fragile: the queue is admitting requests it cannot
        serve inside the ceiling, so they wait the full ``W_max`` and then fail
        anyway, which is worse than rejecting them at admission.

    Raises:
        ValueError: If ``ceiling_s`` is not positive, or via
            :func:`request_deadline_s`.
    """
    if ceiling_s <= 0:
        raise ValueError(f"ceiling_s must be positive, got {ceiling_s}")
    deadline = request_deadline_s(max_added_wait_s, s_p95_s)
    return deadline <= ceiling_s, deadline


def max_added_wait_under_ceiling(
    s_p95_s: float,
    ceiling_s: float = SAGEMAKER_INVOCATION_CEILING_S,
) -> float:
    """Largest ``W_max`` that still fits inside the invocation ceiling, seconds.

    The inverse of :func:`fits_invocation_ceiling`, so an infeasible plan can be
    told *what would work* instead of only that it does not. Clamped at zero: a
    model whose own p95 exceeds the ceiling cannot be rescued by a shorter queue,
    and a negative budget would read as one.

    Raises:
        ValueError: If ``s_p95_s`` or ``ceiling_s`` is not positive.
    """
    if s_p95_s <= 0:
        raise ValueError(f"s_p95_s must be positive, got {s_p95_s}")
    if ceiling_s <= 0:
        raise ValueError(f"ceiling_s must be positive, got {ceiling_s}")
    return max(0.0, ceiling_s - s_p95_s)


def w_max_for_slo(slo_s: float, s_p95_s: float) -> float:
    """Queueing budget an end-to-end first-byte SLO leaves over, seconds.

    ``SLO - S_p95``. The SLO is the whole promise — queue plus service — so the queue
    gets whatever service does not already spend. This makes ``W_max`` derived rather
    than chosen, which is the point: two independent fields can disagree with the SLO,
    one derived field cannot.

    ``s_p95_s``, not ``s_mean_s``, for the reason :func:`request_deadline_s` gives — a
    mean-sized budget is missed by half the requests that reach it. The result is
    therefore the wait a *tail* request can absorb and still make the promise.

    Clamped at zero, so a model too slow to meet the SLO unqueued reports no budget
    rather than a negative one. Zero does not mean "a queue-free design is fine" — it
    means not even an unqueued request makes the promise. Callers wanting that
    distinction should ask :func:`slo_is_feasible` rather than compare against zero.

    Raises:
        ValueError: If ``slo_s`` or ``s_p95_s`` is not positive.
    """
    if slo_s <= 0:
        raise ValueError(f"slo_s must be positive, got {slo_s}")
    if s_p95_s <= 0:
        raise ValueError(f"s_p95_s must be positive, got {s_p95_s}")
    return max(0.0, slo_s - s_p95_s)


def slo_is_feasible(slo_s: float, s_p95_s: float) -> bool:
    """Whether the SLO is achievable at all on this model's own service time.

    False when ``S_p95 >= SLO``: the tail of an *unqueued* request already misses the
    promise, so no queue depth, instance count, or scaling policy can rescue it. Only a
    faster model, a smaller request, or a looser SLO will. Distinct from a ``W_max`` of
    zero being merely tight — this says the design cannot work.

    Raises:
        ValueError: Via :func:`w_max_for_slo`.
    """
    return w_max_for_slo(slo_s, s_p95_s) > 0


def fits_slo(
    max_added_wait_s: float,
    s_p95_s: float,
    slo_s: float,
) -> tuple[bool, float]:
    """Whether a queued request can reach first byte inside the end-to-end SLO.

    The SLO analogue of :func:`fits_invocation_ceiling`, and the check that catches a
    ``W_max`` set by hand rather than derived. Both bounds apply and neither implies the
    other: the 60s ceiling is the platform hanging up, this is the promise being broken,
    and a config can satisfy the ceiling while missing the SLO by an order of magnitude.

    Returns:
        ``(fits, deadline_s)`` where ``deadline_s`` is :func:`request_deadline_s`. When
        ``fits`` is false the queue is admitting requests it can only serve *late* —
        they succeed, so nothing errors, and the SLO is missed silently. That is worse
        than a ceiling breach, which at least surfaces as a failed invocation.

    Raises:
        ValueError: If ``slo_s`` is not positive, or via :func:`request_deadline_s`.
    """
    if slo_s <= 0:
        raise ValueError(f"slo_s must be positive, got {slo_s}")
    deadline = request_deadline_s(max_added_wait_s, s_p95_s)
    return deadline <= slo_s, deadline


def c_target(c_max: float, k: float, derate: float = DEFAULT_DERATE) -> float:
    """Per-instance concurrency the scaling policy should track.

    ``derate x C_max / k`` — reserve ``1/k`` of each instance's ceiling so the
    running fleet can absorb a ``k``-fold surge while replacements boot.

    Raises:
        ValueError: If ``c_max`` is not positive, ``k`` < 1, or ``derate`` is
            outside ``(0, 1]``. ``k < 1`` would *raise* the target above the
            knee, which is never what the caller means.
    """
    if c_max <= 0:
        raise ValueError(f"c_max must be positive, got {c_max}")
    if k < 1:
        raise ValueError(f"k must be >= 1 (k<1 would target above the knee), got {k}")
    if not 0 < derate <= 1:
        raise ValueError(f"derate must be in (0, 1], got {derate}")
    return derate * c_max / k


def c_slo_cap(max_added_wait_s: float, s_mean_s: float) -> float:
    """Concurrency ceiling implied by the queueing budget alone.

    At concurrency ``C`` on a server that handles one request at a time, the
    last arrival waits roughly ``(C - 1) x S``; bounding that by ``W_max`` gives
    ``C <= W_max / S + 1``. This is an independent bound on :func:`c_target` —
    take whichever is smaller (:func:`effective_c_target`), because a knee
    measured under a generous SLO can still violate a tight wait budget.

    Raises:
        ValueError: If ``s_mean_s`` is not positive or ``max_added_wait_s`` is
            negative.
    """
    if s_mean_s <= 0:
        raise ValueError(f"s_mean_s must be positive, got {s_mean_s}")
    if max_added_wait_s < 0:
        raise ValueError(f"max_added_wait_s must be non-negative, got {max_added_wait_s}")
    return max_added_wait_s / s_mean_s + 1.0


def effective_c_target(
    c_max: float,
    k: float,
    s_mean_s: float,
    max_added_wait_s: float,
    derate: float = DEFAULT_DERATE,
) -> tuple[float, str]:
    """Binding per-instance target and which constraint produced it.

    Returns:
        ``(target, binding)`` where ``binding`` is ``"surge_headroom"`` when the
        ``k``-derated knee binds, or ``"slo_wait_budget"`` when the queueing
        budget does. Reporting *which* bound applied matters: the fix differs —
        a surge-bound target wants a smaller ``k`` or a shorter ``T_total``,
        while an SLO-bound target wants a faster model or a looser wait budget.
    """
    surge = c_target(c_max, k, derate)
    slo = c_slo_cap(max_added_wait_s, s_mean_s)
    if slo < surge:
        return slo, "slo_wait_budget"
    return surge, "surge_headroom"


def n_instances(arrival_rate: float, s_mean_s: float, target_concurrency: float) -> int:
    """Instances needed to serve ``arrival_rate`` at ``target_concurrency``.

    Little's Law for the offered concurrency (``lambda x S``), divided by what
    each instance is allowed to carry. Always at least 1 for any positive load —
    a fractional instance does not exist.

    Raises:
        ValueError: If ``target_concurrency`` or ``s_mean_s`` is not positive, or
            ``arrival_rate`` is negative.
    """
    if arrival_rate < 0:
        raise ValueError(f"arrival_rate must be non-negative, got {arrival_rate}")
    if s_mean_s <= 0:
        raise ValueError(f"s_mean_s must be positive, got {s_mean_s}")
    if target_concurrency <= 0:
        raise ValueError(f"target_concurrency must be positive, got {target_concurrency}")
    if arrival_rate == 0:
        return 0
    return max(1, math.ceil(arrival_rate * s_mean_s / target_concurrency))


def n_instances_from_streams(concurrent_streams: float, target_concurrency: float) -> int:
    """Instances needed for a count of concurrent long-lived streams.

    Bypasses the ``lambda x S`` conversion, which is the honest input for
    bidirectional sessions where one session is not one request and ``S`` is
    "however long the user keeps talking".

    Raises:
        ValueError: If ``target_concurrency`` is not positive or
            ``concurrent_streams`` is negative.
    """
    if concurrent_streams < 0:
        raise ValueError(f"concurrent_streams must be non-negative, got {concurrent_streams}")
    if target_concurrency <= 0:
        raise ValueError(f"target_concurrency must be positive, got {target_concurrency}")
    if concurrent_streams == 0:
        return 0
    return max(1, math.ceil(concurrent_streams / target_concurrency))


def lambda_cap_per_instance(c_max: float, s_mean_s: float) -> float:
    """Maximum sustainable arrival rate for one instance, requests/second.

    Little's Law rearranged: ``C_max / S``. Feeding an instance faster than this
    grows its queue without bound, so it is the denominator for every
    admission-control decision.

    Raises:
        ValueError: If ``c_max`` or ``s_mean_s`` is not positive.
    """
    if c_max <= 0:
        raise ValueError(f"c_max must be positive, got {c_max}")
    if s_mean_s <= 0:
        raise ValueError(f"s_mean_s must be positive, got {s_mean_s}")
    return c_max / s_mean_s


def q_per_instance(c_max: float, s_mean_s: float, max_added_wait_s: float) -> int:
    """Per-instance queue depth, ``Q_max = Lambda_cap x W_max``.

    A queue is a time budget expressed as a depth: admitting more than the
    server can drain within ``W_max`` guarantees the extra requests miss their
    deadline. Rounded *down*, because a queue slot that cannot be served in time
    is worse than a rejection — the client waits, then fails anyway.

    Raises:
        ValueError: If ``c_max`` or ``s_mean_s`` is not positive, or
            ``max_added_wait_s`` is negative.
    """
    if max_added_wait_s < 0:
        raise ValueError(f"max_added_wait_s must be non-negative, got {max_added_wait_s}")
    return int(lambda_cap_per_instance(c_max, s_mean_s) * max_added_wait_s)


def w_absorbed(max_added_wait_s: float, k: float) -> float:
    """Scaling lag the queue can hide from clients, seconds.

    During a ``k``-fold surge, the fleet drains at ``1x`` while work arrives at
    ``kx``, so backlog accumulates at ``(k-1)x`` and the queue's ``W_max`` of
    slack is consumed ``(k-1)`` times faster than wall-clock. At ``k=1`` load is
    flat, there is no backlog to drain, and the queue covers any lag — reported
    as ``inf``.

    Raises:
        ValueError: If ``k`` < 1 or ``max_added_wait_s`` is negative.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if max_added_wait_s < 0:
        raise ValueError(f"max_added_wait_s must be non-negative, got {max_added_wait_s}")
    if k == 1:
        return math.inf
    return max_added_wait_s / (k - 1)


def queue_covers_surge(max_added_wait_s: float, k: float, t_total_s: float) -> bool:
    """Whether the queue alone can carry a ``k``-fold surge for a full ``T_total``.

    True iff ``W_absorbed >= T_total``, i.e. ``W_max >= (k-1) x T_total``. When
    true, a surge is invisible to clients: queued requests drain as new capacity
    arrives. When false, the shortfall must come from standing headroom (a lower
    ``C_target``), a shorter ``T_total``, or shed load.

    Raises:
        ValueError: If ``t_total_s`` is negative, or via :func:`w_absorbed`.
    """
    if t_total_s < 0:
        raise ValueError(f"t_total_s must be non-negative, got {t_total_s}")
    return w_absorbed(max_added_wait_s, k) >= t_total_s


def effective_headroom_lag_s(
    max_added_wait_s: float,
    k: float,
    t_total_s: float,
    high_res_period_s: float = CLOUDWATCH_HIGH_RES_PERIOD_S,
) -> tuple[float, bool]:
    """Lag that standing headroom must cover, after the queue takes its share.

    Returns:
        ``(lag_s, floored)``. ``lag_s`` is the uncovered part of ``T_total``,
        never below ``high_res_period_s``. ``floored`` is True when the metric
        period dominates — a signal that the plan is sized for growth faster
        than we can observe, so the resulting headroom is a guess, not a
        measurement. Callers should surface that rather than bury it.
    """
    absorbed = w_absorbed(max_added_wait_s, k)
    # An infinite absorbed lag (k == 1) subtracts to -inf, which floors to zero:
    # flat traffic leaves nothing for standing headroom to cover.
    uncovered = max(0.0, t_total_s - absorbed)
    if uncovered < high_res_period_s:
        return high_res_period_s, True
    return uncovered, False


def utilization_at_k(k: float, derate: float = DEFAULT_DERATE) -> float:
    """Steady-state utilization implied by reserving headroom for ``k``.

    ``derate / k`` — the price of the surge reserve. At ``k=5`` this is ~17%,
    meaning roughly six instances are paid for to serve one instance of load,
    which is the quantitative argument for attacking ``T_total`` (or using a
    warm pool) instead of buying headroom.

    Raises:
        ValueError: If ``k`` < 1 or ``derate`` is outside ``(0, 1]``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not 0 < derate <= 1:
        raise ValueError(f"derate must be in (0, 1], got {derate}")
    return derate / k


def min_samples_for_k(history_duration_s: float, t_total_s: float) -> int:
    """Independent ``T_total``-wide windows available in a traffic history.

    Measuring ``k`` means asking "how much did load grow across one scaling
    lag", which needs many non-overlapping windows of that width. Too few and
    the observed maximum is an artifact of the sample, not a property of the
    traffic.

    Raises:
        ValueError: If ``t_total_s`` is not positive or ``history_duration_s``
            is negative.
    """
    if t_total_s <= 0:
        raise ValueError(f"t_total_s must be positive, got {t_total_s}")
    if history_duration_s < 0:
        raise ValueError(f"history_duration_s must be non-negative, got {history_duration_s}")
    return int(history_duration_s // t_total_s)
