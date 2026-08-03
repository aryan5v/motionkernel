"""The artifact side of a schedule transform: the thing the hook calls.

FastVideo's ``LoopTransform`` hook provides a decision point and nothing else.
Everything about *when* a step may be skipped lives here, in the artifact,
because the artifact is what gets versioned, hash-verified, dispatched and
promoted. Policy that lived in the framework would be unversioned behaviour no
gate could check -- and a cache whose threshold cannot be tied to the
measurement that justified it is not evidence of anything.

The three structural guarantees are enforced here, at the point of decision,
rather than trusted to the caller:

1. **The first step is never a hit.** Enforced by warmup, not by the
   accumulator happening to start at zero.
2. **No cross-generation state.** Driving without ``begin_generation`` raises;
   a silently carried cache opens the next prompt with the previous one's
   activations, which looks plausible and is very hard to notice.
3. **Consecutive skips are capped.** The input-distance accumulator measures
   drift of the *input* and is blind to error compounding in the *output*, so
   an uncapped policy with a generous threshold can skip most of a schedule and
   produce something smooth, plausible and wrong.

The distance function is injected rather than hardcoded. Relative L1 over the
modulated input is what TeaCache uses, but it is a property of the model's
conditioning path, not of caching, and baking one in would make this
family-specific -- the thing the hook was built to avoid.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .cache import CachePolicy, InputSimilarityCache, TransformError

__all__ = ["ScheduleTransformRuntime", "relative_l1"]


def relative_l1(current: Any, previous: Any) -> float:
    """Relative L1 distance, the TeaCache metric.

    ``mean(|a - b|) / mean(|b|)``, scale-free so one threshold means the same
    thing at any latent magnitude. Torch is imported lazily: this module is
    exercised in tests with plain floats and never needs a GPU to be checked.
    """
    if previous is None:
        return float("inf")
    try:
        numerator = (current - previous).abs().mean()
        denominator = previous.abs().mean()
        value = float(numerator / denominator) if float(denominator) else float("inf")
    except AttributeError:
        # Plain numbers, for tests and for models whose conditioning is scalar.
        denominator = abs(previous)
        value = abs(current - previous) / denominator if denominator else float("inf")
    return value


class ScheduleTransformRuntime:
    """Adapts :class:`InputSimilarityCache` to FastVideo's LoopTransform hook.

    Holds the cached output -- the framework must not, because the framework
    has no way to know when it is stale.
    """

    def __init__(
        self,
        policy: CachePolicy,
        *,
        distance: Callable[[Any, Any], float] = relative_l1,
        signal: Callable[..., Any] | None = None,
    ) -> None:
        self.policy = policy
        self._cache = InputSimilarityCache(policy)
        self._distance = distance
        # What the distance is computed over. Defaults to the latents the hook
        # already passes; a model whose conditioning is better represented by
        # its modulated embedding supplies its own.
        self._signal = signal or (lambda **kw: kw.get("latents"))
        self._previous_signal: Any = None
        self._cached_output: Any = None
        self._started = False

    # -- LoopTransform protocol -----------------------------------------

    def begin_generation(self, *, num_steps: int) -> None:
        """Clear everything. Guarantee 2 lives here."""
        self._cache.begin_generation()
        self._previous_signal = None
        self._cached_output = None
        self._started = True

    def before_step(self, *, step: int, timestep: Any, latents: Any, **kw: Any):
        from fastvideo.optimization.loop_transform import StepDecision  # local

        if not self._started:
            raise TransformError(
                "begin_generation() was not called; refusing to reuse state "
                "that may belong to a previous generation"
            )
        signal = self._signal(step=step, timestep=timestep, latents=latents, **kw)
        distance = (
            0.0
            if self._previous_signal is None
            else self._distance(signal, self._previous_signal)
        )
        # A non-finite distance means the signal is unusable; compute rather
        # than reuse. Skipping on an unmeasurable input is the one case where
        # the cache would be guessing.
        if distance != distance or distance == float("inf"):
            self._pending_signal = signal
            return StepDecision(True, "distance unavailable; computing")

        decision = self._cache.step(step, distance)
        self._pending_signal = signal
        if decision.compute:
            return StepDecision(True, decision.reason)
        if self._cached_output is None:
            # Nothing to reuse. Should be unreachable given warmup, but a
            # cache that returned None here would hand the pipeline garbage.
            return StepDecision(True, "no cached output yet")
        return StepDecision(False, decision.reason)

    def after_step(
        self, *, step: int, timestep: Any, latents: Any, output: Any, **kw: Any
    ) -> Any:
        """Store a computed output, or return the cached one on a skip."""
        self._previous_signal = getattr(self, "_pending_signal", None)
        if output is None:
            return self._cached_output
        self._cached_output = output
        return output

    def end_generation(self) -> dict[str, Any]:
        """Diagnostics for the measurement record."""
        stats = self._cache.stats
        self._started = False
        return {
            "policy": self.policy.as_dict(),
            **stats.as_dict(),
        }
