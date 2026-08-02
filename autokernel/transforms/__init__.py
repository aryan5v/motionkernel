"""Transforms that wrap the denoising loop instead of replacing a region.

Every artifact this project has promoted so far answers "replace these ops with
this kernel". A schedule transform answers a different question: "given what
the loop has computed so far, does this step need to run at all?"

That difference is not cosmetic, and three consequences drive the design here:

* **There is no graph fingerprint to dispatch on.** A loop wrapper has no
  captured region, so it cannot be matched the way a ``module`` or ``subgraph``
  artifact is. It carries its own compatibility identity instead.
* **There is no per-call reference to compare against.** The whole point is
  that a call *did not happen*, so per-call output comparison is meaningless.
  The only honest comparison is the final decoded frames, which is exactly what
  fidelity tier 2 exists for.
* **Amdahl reasoning does not apply.** A kernel makes a step cheaper; a cache
  removes steps. ``estimated_max_e2e_improvement`` -- a ``share x 0.9`` bound
  that already overstated a kernel's return by more than 8x in R4 -- is
  meaningless here and must not be reused.
"""

from __future__ import annotations

from .cache import (
    SCHEDULE_TRANSFORM,
    CacheDecision,
    CachePolicy,
    CacheStats,
    InputSimilarityCache,
    TransformError,
)

__all__ = [
    "SCHEDULE_TRANSFORM",
    "CacheDecision",
    "CachePolicy",
    "CacheStats",
    "InputSimilarityCache",
    "TransformError",
]
