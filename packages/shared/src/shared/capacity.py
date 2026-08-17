# Copyright (c) 2026 Amazon.com
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

"""Capacity planning math for autoscaled inference endpoints.

Modality-neutral by design: nothing here knows about TTS, STT, or SageMaker.
Every function is pure so it can be unit-tested without AWS in scope.

The model is six variables — two chosen, two measured, two derived:

    SLO                       first byte within N seconds, queueing included (chosen)
    max_scaling_per_T_total   surge ratio to survive within one scaling lag (chosen)
    T_total                   trigger -> new instance serving traffic (measured)
    Q_max                     concurrency still meeting the SLO (measured)
    C_scale_max               concurrency at which to scale out (derived)
    C_scale_min               concurrency at which to scale in (derived)

"Concurrency" means queued **plus** currently executing, everywhere, which is what
CloudWatch ``ConcurrentRequestsPerModel`` publishes.

This replaced a model built on ``C_max``, a supposed per-instance concurrency ceiling,
with ``derate``, ``c_slo_cap``, ``w_absorbed`` and a utilization term layered on it. That
model was wrong for a server that handles one request at a time: in-flight work has no
ceiling there, so the "measured C_max" was only queue occupancy at whatever load the
measurement happened to sit at, and every quantity that divided by it inherited that
arbitrariness. What is really bounded is the *wait*, which is what ``Q_max`` measures.

The deleted layer also converted concurrency into arrival rates (``Lambda_cap = C_max/S``)
in several places, and every units defect found on this endpoint came from one of those
conversions. Measuring ``Q_max`` closed-loop — hold N requests outstanding, count — needs
no such conversion. :func:`n_instances` is the one place a rate still appears, because a
stated peak in requests/second has to reach concurrency somehow; prefer
:func:`n_instances_from_streams` when the load is stated as sessions.
"""

from __future__ import annotations

import math
import random
from typing import NamedTuple

#: CloudWatch's finest publication granularity for high-resolution metrics. No
#: reactive policy can respond faster than this, which is why it floors the derived
#: scale-out cooldown: a cooldown below one metric period lets the policy act twice
#: on the same datapoint. Also the period the deployed alarm reads at, so it is the
#: resolution at which ``C_scale_max`` is compared against anything.
CLOUDWATCH_HIGH_RES_PERIOD_S = 10.0

#: Hard ceiling on a single SageMaker real-time invocation, seconds. Not a policy
#: choice and not tunable: the runtime closes the connection at 60s regardless of
#: what the container is doing. Everything a request spends — queueing wait plus
#: service — has to fit inside it, so it caps ``W_max`` no matter how generous the
#: SLO is. Containers set ``MAX_REQUEST_AGE_S`` a few seconds below to shed a
#: doomed request rather than have the client see a truncated stream.
SAGEMAKER_INVOCATION_CEILING_S = 60.0

#: Simulated surges per :func:`shed_probability` call. 200 puts the standard error
#: near two percentage points, which is enough to tell "this threshold sheds load"
#: from "this one does not" without making a planning run feel slow.
DEFAULT_SHED_TRIALS = 200


def request_deadline_s(max_added_wait_s: float, s_p95_s: float) -> float:
    """Worst-case wall-clock a queued request occupies, seconds.

    ``W_max + S_p95``: the wait it absorbs in the queue plus the service it then
    receives. Judged against :data:`SAGEMAKER_INVOCATION_CEILING_S` rather than
    against the SLO — the SLO is when audio *starts*, while this is when the
    invocation *ends*, and only the latter is what the runtime times out.

    ``s_p95_s``, not a mean: a mean-sized deadline is missed by half the requests
    that reach it, which is not a deadline.

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

    ``s_p95_s``, not a mean, for the reason :func:`request_deadline_s` gives. The result
    is therefore the wait a *tail* request can absorb and still make the promise.

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

    The SLO analogue of :func:`fits_invocation_ceiling`. Both bounds apply and neither
    implies the other: the 60s ceiling is the platform hanging up, this is the promise
    being broken, and a config can satisfy the ceiling while missing the SLO by an order
    of magnitude.

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


class ScaleThresholds(NamedTuple):
    """The two derived scaling thresholds, plus the fleet size they are safe at.

    A NamedTuple so callers can unpack it positionally *and* so three bare floats
    cannot be transposed at a call site — these go straight into a deployed policy,
    where swapping scale-out for scale-in is a silent outage rather than a type error.
    """

    #: Concurrency at or above which to add an instance.
    c_scale_max: float
    #: Concurrency at or below which to remove one.
    c_scale_min: float
    #: Smallest fleet at which removing one instance does not immediately push the
    #: survivors back over ``c_scale_max``. ``None`` when no fleet size is safe.
    min_safe_instances: int | None


def scale_thresholds(q_max: float, max_scaling_per_t_total: float) -> ScaleThresholds:
    """Scale-out and scale-in thresholds as fractions of measured ``Q_max``.

    With ``h = max_scaling_per_t_total - 1`` the headroom fraction to hold in reserve::

        C_scale_max = (1 - h)     x Q_max   # scale out once headroom drops below h
        C_scale_min = (1 - 2 x h) x Q_max   # scale in once there is h of excess

    At ``Q_max=50`` and a 1.25 surge ratio: 37.5 and 25.0.

    ``min_safe_instances`` exists because scale-in redistributes rather than removes
    load. Dropping one of ``N`` instances multiplies each survivor's concurrency by
    ``N/(N-1)``, so the move is only safe when that factor keeps the result under
    ``C_scale_max`` — i.e. ``N >= r/(r-1)`` for ``r = C_scale_max/C_scale_min``. At a
    1.25 ratio ``r`` is 1.5 and ``N`` must be at least 3; a fleet scaling 2->1 lands
    exactly on ``Q_max`` and breaches the SLO on the way down. Callers should report
    this rather than silently deploy it, since a long scale-in cooldown damps the
    resulting flap but does not remove it.

    Raises:
        ValueError: If ``q_max`` is not positive, or ``max_scaling_per_t_total`` is
            below 1 (a shrinking surge is not a surge) or at/above 1.5. At 1.5 the
            scale-in threshold reaches zero and beyond it goes negative, which would
            deploy as "never scale in" — a plausible-looking number that silently
            disables half the policy, so it is refused rather than clamped.
    """
    if q_max <= 0:
        raise ValueError(f"q_max must be positive, got {q_max}")
    if max_scaling_per_t_total < 1:
        raise ValueError(f"max_scaling_per_t_total must be >= 1, got {max_scaling_per_t_total}")
    if max_scaling_per_t_total >= 1.5:
        raise ValueError(
            f"max_scaling_per_t_total must be < 1.5, got {max_scaling_per_t_total}: "
            f"the scale-in threshold is (1 - 2 x h) x Q_max, which is <= 0 from h=0.5 "
            "up, and a non-positive threshold deploys as a policy that never scales in. "
            "Surviving a surge that large needs standing headroom rather than queue "
            "slots — hold more instances, or shorten T_total."
        )

    headroom = max_scaling_per_t_total - 1.0
    c_scale_max = (1.0 - headroom) * q_max
    c_scale_min = (1.0 - 2.0 * headroom) * q_max

    # r <= 1 means the two thresholds coincide (h == 0): every scale-in lands at or
    # above the scale-out point, so no fleet size is safe and there is no N to report.
    if c_scale_min <= 0 or c_scale_max <= c_scale_min:
        min_safe: int | None = None
    else:
        ratio = c_scale_max / c_scale_min
        bound = ratio / (ratio - 1.0)
        # The bound is inclusive (N/(N-1) <= r is safe), so an N sitting exactly on it
        # must not be pushed to N+1 by representation error — and it routinely would
        # be, because `1 - 2h` is inexact for most h: at h=0.05 the thresholds are
        # 47.5 and 44.999999999999993, putting a mathematically-exact 19 at 18.9999...
        # or 19.0000... depending on rounding direction. The tolerance is ~1e-9
        # relative, many orders below the precision of a measured Q_max, so it can
        # only ever change the answer in cases where both answers are equally true.
        min_safe = max(2, math.ceil(bound - 1e-9))

    return ScaleThresholds(c_scale_max, c_scale_min, min_safe)


def utilization_for_occupancy(occupancy: float) -> float:
    """Utilization implied by a mean concurrency of ``occupancy``, for one server.

    Little's Law inverted for M/M/1: mean number in system is ``rho/(1-rho)``, so
    ``rho = L/(1+L)``. This is the step that makes queue occupancy interpretable as
    load, and it is steeply non-linear — occupancy 4 is 80% utilized while occupancy
    37.5 is 97.4%. That is why a scale-out threshold set at three quarters of ``Q_max``
    is not three quarters of the way to trouble but essentially all of it.

    Returns 0.0 at zero occupancy, and approaches (never reaches) 1.0 as occupancy
    grows. An observed occupancy cannot express ``rho >= 1``: a queue at critical or
    divergent load has no steady-state mean to measure.

    Raises:
        ValueError: If ``occupancy`` is negative.
    """
    if occupancy < 0:
        raise ValueError(f"occupancy must be non-negative, got {occupancy}")
    return occupancy / (1.0 + occupancy)


def shed_probability(
    occupancy: float,
    q_max: float,
    t_total_s: float,
    service_s: float,
    *,
    seed: int = 1234,
    trials: int = DEFAULT_SHED_TRIALS,
) -> float:
    """P(queue reaches ``Q_max``) while waiting out one ``T_total`` from ``occupancy``.

    The falsifiable version of "is this scale-out threshold early enough". A threshold
    is a promise that after it fires, the existing instances hold the SLO for the whole
    time a replacement takes to arrive. Whether they do is a queueing question with a
    numeric answer, so compute it rather than assert it.

    Simulated as an M/M/1 continuous-time chain: arrivals at the rate
    :func:`utilization_for_occupancy` implies for the starting ``occupancy``, service at
    ``1/service_s``, run for ``t_total_s``, and count the runs that touch ``q_max``.
    Exact for the model (no time discretisation) and deterministic given ``seed`` — on a
    private :class:`random.Random`, not the module-global one, so a caller's own seeding
    cannot change a number that gets published in a plan.

    Poisson arrivals matter here and are not a convenience: a *deterministic* fluid
    model says a queue below its drain rate never grows, which is how one arrives at
    the false conclusion that a threshold at high utilization is fine. The variance is
    the entire risk.

    Returns:
        A probability in ``[0, 1]``. 1.0 when ``occupancy`` already meets or exceeds
        ``q_max`` — nothing needs simulating to know that has shed.

    Raises:
        ValueError: If ``q_max`` or ``service_s`` is not positive, ``t_total_s`` or
            ``occupancy`` is negative, or ``trials`` is not positive.
    """
    if q_max <= 0:
        raise ValueError(f"q_max must be positive, got {q_max}")
    if service_s <= 0:
        raise ValueError(f"service_s must be positive, got {service_s}")
    if t_total_s < 0:
        raise ValueError(f"t_total_s must be non-negative, got {t_total_s}")
    if occupancy < 0:
        raise ValueError(f"occupancy must be non-negative, got {occupancy}")
    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")

    depth = math.ceil(q_max)
    start = int(round(occupancy))
    if start >= depth:
        return 1.0
    if t_total_s == 0:
        return 0.0

    mu = 1.0 / service_s
    lam = utilization_for_occupancy(occupancy) * mu
    if lam <= 0:
        return 0.0

    rng = random.Random(seed)
    hits = 0
    for _ in range(trials):
        n = start
        t = 0.0
        while t < t_total_s:
            # Competing exponentials: the next event is an arrival or a departure,
            # whichever clock fires first. Sampling the minimum directly (one draw at
            # rate lam + mu, then a Bernoulli for which fired) is the same distribution
            # as two draws and a comparison. An empty queue has no departure clock.
            rate = lam + mu if n > 0 else lam
            t += rng.expovariate(rate)
            if t >= t_total_s:
                break
            if n == 0 or rng.random() < lam / rate:
                n += 1
                if n >= depth:
                    hits += 1
                    break
            else:
                n -= 1

    return hits / trials


def n_instances(arrival_rate: float, s_mean_s: float, target_concurrency: float) -> int:
    """Instances needed to serve ``arrival_rate`` at ``target_concurrency``.

    Little's Law for the offered concurrency (``lambda x S``), divided by what each
    instance is allowed to carry — ``C_scale_max``, since that is the concurrency the
    deployed policy holds each instance at. Always at least 1 for any positive load: a
    fractional instance does not exist.

    The one surviving rate-to-concurrency conversion, and it is here because a peak
    stated in requests/second has to become a concurrency somehow. It is a *stated*
    input rather than a measurement, which is what makes it safe — nothing measured is
    divided by it. For long-lived bidirectional sessions prefer
    :func:`n_instances_from_streams`, where the input is already a concurrency.

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
