"""Sustained-load warmup: time only after the clock stops moving.

The paired protocol cancels drift *within* a pair. It does not help if the
whole measurement sits on the steep part of the boost ramp, where consecutive
runs differ by more than the effect being measured. On this cluster the GPU
idles at 285 MHz against a 2062 MHz maximum, so the first runs after an idle
period are measuring the ramp.

Clock locking would remove the ramp, and it is unavailable. The alternative is
to wait it out under load and to *record* that we did: this module drives a
sustained load until the observed SM clock plateaus, and emits the full clock
trace so a reader can see the plateau rather than take it on trust.

The trace is what makes the claim checkable. A measurement asserting "warmed
up" with no trace is an assertion; one carrying twenty samples that flatten is
evidence. :func:`~autokernel.dispatch.paired.summarize_paired` refuses to gate
without it.

Plateau detection is deliberately crude -- the spread of the last ``window``
samples relative to their mean, under a tolerance -- because the alternative
is a model of the boost curve, and a wrong model would be worse than a blunt
rule that can be read off the trace.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["WarmupResult", "clock_plateaued", "sustained_warmup"]

#: Consecutive samples that must agree before the clock is called settled.
DEFAULT_WINDOW = 5
#: Relative spread within the window, as a fraction of its mean.
DEFAULT_TOLERANCE = 0.02
#: Hard cap: warming forever is a hang, not a warmup.
DEFAULT_MAX_SECONDS = 180.0
#: Seconds between clock samples.
DEFAULT_INTERVAL = 2.0


@dataclass(frozen=True)
class WarmupResult:
    """What the warmup observed, including the samples behind the verdict."""

    plateaued: bool
    seconds: float
    samples: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "plateaued": self.plateaued,
            "seconds": round(self.seconds, 2),
            "samples": list(self.samples),
            "reason": self.reason,
            "window": DEFAULT_WINDOW,
            "tolerance": DEFAULT_TOLERANCE,
        }


def clock_plateaued(
    clocks: Sequence[float],
    *,
    window: int = DEFAULT_WINDOW,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    """Whether the last ``window`` clock samples agree within ``tolerance``.

    Relative spread rather than absolute, so the rule reads the same on a
    2062 MHz part and a 1400 MHz one.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    if len(clocks) < window:
        return False
    tail = [float(value) for value in clocks[-window:]]
    mean = sum(tail) / len(tail)
    if mean <= 0:
        return False
    return (max(tail) - min(tail)) / mean <= tolerance


def sustained_warmup(
    load: Callable[[], Any],
    sample_clock: Callable[[], float | None],
    *,
    window: int = DEFAULT_WINDOW,
    tolerance: float = DEFAULT_TOLERANCE,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    interval: float = DEFAULT_INTERVAL,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> WarmupResult:
    """Drive ``load`` until ``sample_clock`` plateaus, or ``max_seconds``.

    Args:
        load: one unit of GPU work. Called repeatedly; the point is to hold the
            device busy so it boosts and stays boosted.
        sample_clock: current SM clock in MHz, or None when unavailable.
        sleep / monotonic: injectable so the whole loop is testable without
            waiting or a GPU.

    Returns:
        A :class:`WarmupResult` whose ``samples`` are the trace. Timing out is
        reported as ``plateaued=False`` with the trace intact rather than
        raised: a measurement taken on an unsettled clock is worse evidence,
        not no evidence, and the caller decides what to do about it.

    If the clock cannot be sampled at all the result is unplateaued with the
    reason recorded, because "we could not observe it" must not be storable as
    "it was fine".
    """
    started = monotonic()
    samples: list[dict[str, Any]] = []
    clocks: list[float] = []
    unavailable = 0

    while True:
        load()
        clock = sample_clock()
        elapsed = monotonic() - started
        if clock is None:
            unavailable += 1
            samples.append({"t": round(elapsed, 2), "sm_clock_mhz": None})
        else:
            clocks.append(float(clock))
            samples.append({"t": round(elapsed, 2), "sm_clock_mhz": float(clock)})
            if clock_plateaued(clocks, window=window, tolerance=tolerance):
                return WarmupResult(
                    plateaued=True,
                    seconds=elapsed,
                    samples=tuple(samples),
                    reason=(
                        f"last {window} samples within {tolerance:.0%} of their mean"
                    ),
                )
        if elapsed >= max_seconds:
            if unavailable and not clocks:
                reason = (
                    "clock could not be sampled; cannot show the device settled"
                )
            else:
                reason = (
                    f"clock still moving after {max_seconds:.0f}s "
                    f"(last {window} samples spread beyond {tolerance:.0%})"
                )
            return WarmupResult(
                plateaued=False,
                seconds=elapsed,
                samples=tuple(samples),
                reason=reason,
            )
        sleep(interval)
