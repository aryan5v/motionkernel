"""Paired measurement protocol: measure the difference, not two medians.

Schema v2 established that this cluster's variance cannot be controlled away.
Exclusive-node allocation left the native LTX arm at 5.57% CV against a 3%
ceiling, and clock locking is unavailable -- ``nvidia-smi -lgc`` needs
privileges Pyxis does not grant, so the GPU idles at 285 MHz against a 2062 MHz
maximum and every run starts from a different point on the boost ramp. The
honest consequence was that LTX dispatch A/Bs became ungateable.

v2 detected the problem. This module removes it, by changing what is measured.

The reason a sequential A/B fails here is not that runs are noisy -- it is that
the noise is *correlated with time*. Fifteen native runs then fifteen candidate
runs means the two arms sample different parts of the thermal and clock
trajectory, so the difference between their medians contains the drift as well
as the effect. Interleaving ABABAB... makes each pair adjacent in time, so
whatever the clock is doing it is doing to both members of the pair, and the
within-pair difference cancels it.

That is why gating consumes the paired delta and never the raw medians. The
medians are still recorded -- they are what a reader expects to see -- but a
gate computed from them would reintroduce exactly the drift the pairing exists
to remove.

Three statistics are reported for the delta:

* the **median paired difference**, which is the effect;
* a **bootstrap confidence interval** on that median, which says how well the
  sample pins it down;
* a **Wilcoxon signed-rank p-value**, which asks whether the differences are
  distributed around zero. It is rank-based and therefore does not assume the
  differences are normal, which they are not when a clock ramps.

A speedup that clears the gate on its point estimate but whose CI spans 1.0 is
reported as inconclusive rather than as a pass. This module never widens a
gate; it refuses to let a gate be applied to a number that cannot support it.

No torch, no GPU, no I/O: the protocol is arithmetic over run times, so it is
exhaustively testable on synthetic data with drift injected deliberately.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "MEASUREMENT_SCHEMA_VERSION_PAIRED",
    "PROTOCOL_PAIRED",
    "PROTOCOL_SEQUENTIAL",
    "PairedResult",
    "ProtocolError",
    "bootstrap_median_ci",
    "interleaved_schedule",
    "paired_speedups",
    "summarize_paired",
    "wilcoxon_signed_rank_p",
]

#: Schema version that carries the paired protocol.
MEASUREMENT_SCHEMA_VERSION_PAIRED = 3

PROTOCOL_PAIRED = "paired_interleaved"
PROTOCOL_SEQUENTIAL = "sequential"

#: Bootstrap resamples for the CI. 2000 is enough for a stable 95% interval on
#: 15 pairs and cheap enough to run inside a measurement.
_BOOTSTRAP_RESAMPLES = 2000
_BOOTSTRAP_SEED = 20260802


class ProtocolError(ValueError):
    """The runs cannot be interpreted under the paired protocol."""


def interleaved_schedule(pairs: int) -> tuple[str, ...]:
    """Arm order for ``pairs`` pairs, alternating within the pair: NC CN NC CN...

    **Not** plain ABAB, and the difference matters. Under ABAB the candidate
    always occupies the later slot of its pair, so it collects one run's worth
    of drift that the native member does not: on a ramping clock that biases
    every pair in the same direction, and the paired estimate inherits it. With
    a 1% per-run ramp the bias is about 1%, which is comparable to the effects
    this gate decides on.

    ABBA cancels it. Half the pairs put the candidate later and half put it
    earlier, so the within-pair offset averages out across pairs instead of
    accumulating.

    Emitted by the launcher and recorded in the measurement so a reader can
    confirm the runs were actually interleaved rather than trust that they
    were.
    """
    if not isinstance(pairs, int) or isinstance(pairs, bool) or pairs < 1:
        raise ProtocolError(f"pairs must be a positive integer, got {pairs!r}")
    if pairs % 2:
        # With an odd count the NC and CN pairs cannot balance, so the median
        # lands in whichever cluster has one extra member and keeps the very
        # offset ABBA exists to remove. Refused rather than silently rounded:
        # a launcher that asked for 15 and got 16 would report the wrong n.
        raise ProtocolError(
            f"ABBA needs an even number of pairs so the NC and CN halves "
            f"balance; got {pairs}. Use {pairs + 1}."
        )
    order: list[str] = []
    for index in range(pairs):
        if index % 2 == 0:
            order.extend(("native", "candidate"))
        else:
            order.extend(("candidate", "native"))
    return tuple(order)


#: Longest run of one arm an interleaved schedule may contain. ABBA
#: (N C C N N C C N) legitimately puts two of the same arm back to back at each
#: quartet boundary; three in a row is a block.
_MAX_CONSECUTIVE = 2


def _is_interleaved(schedule: Sequence[str]) -> bool:
    """Whether ``schedule`` interleaves the arms rather than blocking them.

    The property is *not* strict alternation. ABBA -- which is what cancels the
    within-pair drift offset -- puts two of the same arm adjacent at every
    quartet boundary, so a naive ``a != b`` check would reject the very
    schedule this protocol prefers.

    What is actually required: neither arm runs more than twice consecutively,
    and both arms appear equally often. That admits ABAB and ABBA and rejects a
    sequential block wearing a paired label.

    Does not require starting on ``native``: a launcher alternating from the
    candidate is equally paired.
    """
    if len(schedule) < 2:
        return False
    counts: dict[str, int] = {}
    run = 1
    for index, arm in enumerate(schedule):
        counts[arm] = counts.get(arm, 0) + 1
        if index and arm == schedule[index - 1]:
            run += 1
            if run > _MAX_CONSECUTIVE:
                return False
        else:
            run = 1
    return len(counts) == 2 and len(set(counts.values())) == 1


def paired_speedups(
    native: Sequence[float], candidate: Sequence[float]
) -> tuple[float, ...]:
    """Per-pair speedup ``native[i] / candidate[i]``.

    Pairing is positional and that is the whole point: pair *i* is two runs
    adjacent in time, so their ratio is taken before any aggregation and the
    clock state they shared cancels.
    """
    if len(native) != len(candidate):
        raise ProtocolError(
            f"paired protocol needs equal arms; got {len(native)} native and "
            f"{len(candidate)} candidate runs"
        )
    if not native:
        raise ProtocolError("paired protocol needs at least one pair")
    ratios: list[float] = []
    for index, (n, c) in enumerate(zip(native, candidate)):
        if not (math.isfinite(n) and math.isfinite(c)) or n <= 0 or c <= 0:
            raise ProtocolError(
                f"pair {index} has a non-positive or non-finite run time "
                f"(native={n!r}, candidate={c!r})"
            )
        ratios.append(n / c)
    return tuple(ratios)


def bootstrap_median_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = _BOOTSTRAP_RESAMPLES,
    seed: int = _BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the median of ``values``.

    Seeded, so the same runs produce the same interval. A measurement whose
    confidence interval moved because the bootstrap reseeded would be
    impossible to review.
    """
    if not values:
        raise ProtocolError("cannot bootstrap an empty sample")
    if not 0 < confidence < 1:
        raise ProtocolError(f"confidence must be in (0, 1), got {confidence!r}")
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    rng = random.Random(seed)
    size = len(values)
    medians = []
    for _ in range(resamples):
        sample = [values[rng.randrange(size)] for _ in range(size)]
        medians.append(statistics.median(sample))
    medians.sort()
    tail = (1.0 - confidence) / 2.0
    low = medians[max(0, int(math.floor(tail * resamples)))]
    high = medians[min(resamples - 1, int(math.ceil((1.0 - tail) * resamples)) - 1)]
    return (float(low), float(high))


def wilcoxon_signed_rank_p(differences: Sequence[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank p-value for ``differences`` vs zero.

    Rank-based, so it does not assume the differences are normally
    distributed -- which they are not when a clock is ramping. Zero
    differences are dropped (the standard treatment) and ties share averaged
    ranks.

    Returns None when too few non-zero differences remain to say anything,
    rather than a number that would look like a result.

    Uses the normal approximation with a continuity correction. At 15 pairs
    that is adequate, and the alternative -- an exact distribution -- would add
    a dependency for a decimal place that no gate reads.
    """
    non_zero = [d for d in differences if d != 0.0]
    if len(non_zero) < 6:
        return None

    magnitudes = sorted((abs(d), 1 if d > 0 else -1) for d in non_zero)
    ranks: list[float] = [0.0] * len(magnitudes)
    index = 0
    while index < len(magnitudes):
        stop = index
        while (
            stop + 1 < len(magnitudes)
            and magnitudes[stop + 1][0] == magnitudes[index][0]
        ):
            stop += 1
        average = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[position] = average
        index = stop + 1

    w_plus = sum(r for r, (_, sign) in zip(ranks, magnitudes) if sign > 0)
    count = len(non_zero)
    mean = count * (count + 1) / 4.0
    variance = count * (count + 1) * (2 * count + 1) / 24.0
    if variance <= 0:
        return None
    z = (abs(w_plus - mean) - 0.5) / math.sqrt(variance)
    p = math.erfc(z / math.sqrt(2.0))
    return float(min(1.0, max(0.0, p)))


@dataclass(frozen=True)
class PairedResult:
    """The paired analysis of one A/B, and whether it may gate."""

    protocol: str
    pairs: int
    speedup_paired_median: float
    ci_low: float
    ci_high: float
    wilcoxon_p: float | None
    native_median: float
    candidate_median: float
    speedup_of_medians: float
    valid_for_gating: bool
    invalid_reasons: tuple[str, ...] = ()

    @property
    def conclusive(self) -> bool:
        """Whether the CI excludes 'no difference'."""
        return not (self.ci_low <= 1.0 <= self.ci_high)

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "pairs": self.pairs,
            "speedup_paired_median": round(self.speedup_paired_median, 6),
            "ci95_low": round(self.ci_low, 6),
            "ci95_high": round(self.ci_high, 6),
            "wilcoxon_p": (
                None if self.wilcoxon_p is None else round(self.wilcoxon_p, 6)
            ),
            "conclusive": self.conclusive,
            # Recorded because readers expect them, and explicitly NOT what the
            # gate consumes: a median-of-medians reintroduces the drift the
            # pairing removes.
            "native_median": round(self.native_median, 6),
            "candidate_median": round(self.candidate_median, 6),
            "speedup_of_medians": round(self.speedup_of_medians, 6),
            "gate_input": "speedup_paired_median",
            "valid_for_gating": self.valid_for_gating,
            "invalid_reasons": list(self.invalid_reasons),
        }


def summarize_paired(
    native: Sequence[float],
    candidate: Sequence[float],
    *,
    schedule: Sequence[str] | None = None,
    clock_trace: Sequence[Any] | None = None,
    require_clock_trace: bool = True,
    arms_differentiated: bool | None = None,
) -> PairedResult:
    """Analyse an interleaved A/B and decide whether it may gate.

    Args:
        native / candidate: run times in schedule order, one per pair.
        schedule: the arm order actually executed. Absent or non-alternating
            means the runs were not interleaved, whatever the record claims.
        clock_trace: observed clocks per run. Required by default: without it
            there is no evidence the warmup reached a plateau, and a paired
            measurement taken during a ramp is better than a sequential one but
            still not a controlled one.
        require_clock_trace: set False only for synthetic data in tests.

    Invalidation is additive to v2's rules, never subtractive. A measurement
    can be recorded and unusable; nothing here relaxes a gate.
    """
    ratios = paired_speedups(native, candidate)
    differences = [n - c for n, c in zip(native, candidate)]

    reasons: list[str] = []
    if schedule is None:
        reasons.append("arm schedule was not recorded; cannot confirm interleaving")
    elif not _is_interleaved(schedule):
        reasons.append(
            "arm assignment was not interleaved; the two arms sample different "
            "parts of the clock trajectory and their difference contains it"
        )
    elif len(schedule) != len(native) + len(candidate):
        reasons.append(
            f"schedule length {len(schedule)} does not match "
            f"{len(native) + len(candidate)} recorded runs"
        )
    if require_clock_trace and not clock_trace:
        reasons.append(
            "clock trace absent; no evidence the sustained-load warmup reached "
            "a plateau before timing began"
        )
    if arms_differentiated is False:
        # The arms were not observably different, so whatever was measured, it
        # was not the thing under test. This is the same failure the attention
        # effective-backend check exists for, one level up: a candidate arm
        # whose configuration silently did not take effect produces a clean,
        # tight, conclusive number for an intervention that never happened.
        reasons.append(
            "the candidate arm was not observably different from native; the "
            "intervention under test did not take effect, so this measures "
            "something else"
        )

    median_ratio = statistics.median(ratios)
    ci_low, ci_high = bootstrap_median_ci(ratios)
    native_median = statistics.median(native)
    candidate_median = statistics.median(candidate)

    return PairedResult(
        protocol=PROTOCOL_PAIRED,
        pairs=len(ratios),
        speedup_paired_median=median_ratio,
        ci_low=ci_low,
        ci_high=ci_high,
        wilcoxon_p=wilcoxon_signed_rank_p(differences),
        native_median=native_median,
        candidate_median=candidate_median,
        speedup_of_medians=(
            native_median / candidate_median if candidate_median > 0 else float("nan")
        ),
        valid_for_gating=not reasons,
        invalid_reasons=tuple(reasons),
    )
