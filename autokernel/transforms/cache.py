"""Input-similarity caching over the denoising loop (TeaCache family).

The idea: consecutive denoising steps often produce nearly identical transformer
inputs, so the previous output can be reused instead of recomputed. The decision
is made on the *relative L1 distance* between this step's modulated input and
the last one that was actually computed.

The distance is **accumulated**, not compared per step, and the accumulator
resets whenever a step is computed. Comparing each step's distance against the
threshold independently would allow an unbounded number of individually-small
changes to add up to an arbitrarily large drift while every single step looked
safe. Accumulating is what makes the threshold mean "how far the input may
drift before we recompute" rather than "how fast it may drift".

This module holds no tensors and never imports torch. It takes a scalar
distance -- which the caller computes however its model requires -- and returns
a decision. That keeps the policy testable without a GPU, and keeps the part
that is easy to get subtly wrong separate from the part that is expensive to
run.

Three properties are structural rather than incidental:

* **The first step of a generation is never a hit.** There is nothing to reuse.
  This is enforced by the warmup, not left to depend on the accumulator
  happening to start at zero.
* **State resets between generations.** A cache surviving into the next
  generation would reuse the previous prompt's activations for its opening
  steps -- a contamination that produces plausible-looking output and is
  correspondingly hard to notice.
* **Consecutive skips are capped.** See :class:`CachePolicy`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "SCHEDULE_TRANSFORM",
    "CacheDecision",
    "CachePolicy",
    "CacheStats",
    "InputSimilarityCache",
    "TransformError",
]

#: The artifact ``target_kind`` this family registers under.
SCHEDULE_TRANSFORM = "schedule_transform"


class TransformError(ValueError):
    """A transform policy is malformed, or is being driven incorrectly."""


@dataclass(frozen=True)
class CachePolicy:
    """When the loop may reuse a previous result.

    Args:
        threshold: accumulated relative-L1 distance at which a step must be
            recomputed. The searched parameter. Higher is faster and less
            faithful.
        warmup_steps: leading steps that are always computed. Must be at least
            1: the first step has nothing to reuse, and making that structural
            means it cannot become a threshold accident.
        max_consecutive_skips: cap on how many steps may be skipped in a row.
            ``None`` means uncapped.

            The cap exists because the accumulator only measures drift *of the
            input*, and a long run of skips compounds error in the output that
            the input distance never sees. An uncapped policy with a generous
            threshold can skip most of the schedule and produce a video that is
            smooth, plausible, and wrong -- which is the failure mode a
            per-frame perceptual gate catches late and expensively. Capping is
            the cheap structural defence; the tier-2 gate is the backstop.
        cooldown_steps: steps that must be computed after the cap is hit,
            before skipping may resume.
    """

    threshold: float
    warmup_steps: int = 1
    max_consecutive_skips: int | None = 3
    cooldown_steps: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, (int, float))
            or not math.isfinite(float(self.threshold))
            or float(self.threshold) < 0
        ):
            raise TransformError(
                f"threshold must be a finite non-negative number, "
                f"got {self.threshold!r}"
            )
        if (
            isinstance(self.warmup_steps, bool)
            or not isinstance(self.warmup_steps, int)
            or self.warmup_steps < 1
        ):
            raise TransformError(
                "warmup_steps must be an integer >= 1; the first step of a "
                "generation has nothing to reuse"
            )
        if self.max_consecutive_skips is not None and (
            isinstance(self.max_consecutive_skips, bool)
            or not isinstance(self.max_consecutive_skips, int)
            or self.max_consecutive_skips < 1
        ):
            raise TransformError(
                "max_consecutive_skips must be an integer >= 1, or None"
            )
        if (
            isinstance(self.cooldown_steps, bool)
            or not isinstance(self.cooldown_steps, int)
            or self.cooldown_steps < 1
        ):
            raise TransformError("cooldown_steps must be an integer >= 1")

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": float(self.threshold),
            "warmup_steps": self.warmup_steps,
            "max_consecutive_skips": self.max_consecutive_skips,
            "cooldown_steps": self.cooldown_steps,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> CachePolicy:
        if raw is None or not hasattr(raw, "get"):
            raise TransformError("cache policy must be a mapping")
        unknown = set(raw) - {
            "threshold",
            "warmup_steps",
            "max_consecutive_skips",
            "cooldown_steps",
        }
        if unknown:
            raise TransformError(f"unknown cache policy fields: {sorted(unknown)}")
        if "threshold" not in raw:
            raise TransformError(
                "cache policy must declare a threshold; it is the searched "
                "parameter and must live in the manifest, not in the payload"
            )
        return cls(
            threshold=raw["threshold"],
            warmup_steps=raw.get("warmup_steps", 1),
            max_consecutive_skips=raw.get("max_consecutive_skips", 3),
            cooldown_steps=raw.get("cooldown_steps", 1),
        )


@dataclass(frozen=True)
class CacheDecision:
    """What the loop should do for one step, and why."""

    step: int
    compute: bool
    reason: str
    accumulated: float

    @property
    def skipped(self) -> bool:
        return not self.compute


@dataclass(frozen=True)
class CacheStats:
    """What a generation actually did.

    Reported because a transform that never fires and one that fires on every
    step both look like "it ran". The hit rate is what distinguishes a genuine
    1.8x from a no-op, and it is the number that makes a speedup claim
    interpretable.
    """

    steps_total: int
    steps_computed: int
    steps_skipped: int
    max_consecutive_skips_used: int

    @property
    def hit_rate(self) -> float:
        if self.steps_total == 0:
            return 0.0
        return self.steps_skipped / self.steps_total

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps_total": self.steps_total,
            "steps_computed": self.steps_computed,
            "steps_skipped": self.steps_skipped,
            "max_consecutive_skips_used": self.max_consecutive_skips_used,
            "hit_rate": self.hit_rate,
        }


class InputSimilarityCache:
    """Decides, per step, whether the transformer must run.

    Drive it once per denoising step with the relative L1 distance between the
    current modulated input and the one from the last *computed* step:

        cache.begin_generation()
        for step in range(num_steps):
            decision = cache.step(step, distance)
            if decision.compute:
                output = transformer(...)
                cache.record(output_marker)
            ...

    The class does not hold the cached tensor; the caller does. Holding model
    outputs here would make the policy untestable without a GPU for no benefit.
    """

    def __init__(self, policy: CachePolicy) -> None:
        self.policy = policy
        self._started = False
        self._reset_counters()

    def _reset_counters(self) -> None:
        self._accumulated = 0.0
        self._consecutive_skips = 0
        self._cooldown_remaining = 0
        self._steps_total = 0
        self._steps_computed = 0
        self._steps_skipped = 0
        self._max_run = 0
        self._last_step: int | None = None

    def begin_generation(self) -> None:
        """Clear all state. Must be called before each generation.

        Not calling it is a caller error rather than a silent carry-over: the
        first :meth:`step` of an unstarted cache raises, because a cache that
        quietly reused the previous generation's state would produce opening
        frames contaminated by the previous prompt.
        """
        self._started = True
        self._reset_counters()

    def step(self, step: int, distance: float) -> CacheDecision:
        """Decide whether step ``step`` must be computed."""
        if not self._started:
            raise TransformError(
                "begin_generation() must be called before the first step; "
                "reusing state across generations would contaminate the "
                "opening frames with the previous prompt"
            )
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise TransformError(f"step must be a non-negative integer, got {step!r}")
        if self._last_step is not None and step <= self._last_step:
            raise TransformError(
                f"steps must advance: got {step} after {self._last_step}"
            )
        if (
            isinstance(distance, bool)
            or not isinstance(distance, (int, float))
            or not math.isfinite(float(distance))
            or float(distance) < 0
        ):
            raise TransformError(
                f"distance must be a finite non-negative number, got {distance!r}"
            )
        self._last_step = step
        self._steps_total += 1

        decision = self._decide(step, float(distance))
        if decision.compute:
            self._steps_computed += 1
            self._accumulated = 0.0
            self._consecutive_skips = 0
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
        else:
            self._steps_skipped += 1
            self._consecutive_skips += 1
            self._max_run = max(self._max_run, self._consecutive_skips)
            cap = self.policy.max_consecutive_skips
            if cap is not None and self._consecutive_skips >= cap:
                self._cooldown_remaining = self.policy.cooldown_steps
        return decision

    def _decide(self, step: int, distance: float) -> CacheDecision:
        if step < self.policy.warmup_steps:
            # Structural, not a threshold accident: there is nothing to reuse.
            return CacheDecision(
                step, True, "warmup: nothing cached yet", self._accumulated
            )

        accumulated = self._accumulated + distance
        self._accumulated = accumulated

        if self._cooldown_remaining > 0:
            return CacheDecision(
                step,
                True,
                (
                    f"cooldown: {self._cooldown_remaining} step(s) must be "
                    f"computed after the consecutive-skip cap"
                ),
                accumulated,
            )

        cap = self.policy.max_consecutive_skips
        if cap is not None and self._consecutive_skips >= cap:
            return CacheDecision(
                step,
                True,
                f"consecutive-skip cap of {cap} reached",
                accumulated,
            )

        if accumulated >= self.policy.threshold:
            return CacheDecision(
                step,
                True,
                (
                    f"accumulated distance {accumulated:.6g} reached threshold "
                    f"{self.policy.threshold:.6g}"
                ),
                accumulated,
            )

        return CacheDecision(
            step,
            False,
            (
                f"accumulated distance {accumulated:.6g} below threshold "
                f"{self.policy.threshold:.6g}"
            ),
            accumulated,
        )

    @property
    def stats(self) -> CacheStats:
        return CacheStats(
            steps_total=self._steps_total,
            steps_computed=self._steps_computed,
            steps_skipped=self._steps_skipped,
            max_consecutive_skips_used=self._max_run,
        )
