"""Parity policy: the workload's output contract, carried down to the verifier.

A workload declares how its outputs must be preserved (``workload.parity.policy``
in the YAML). Before this module existed the declaration only reached the
*final* full-generation frame comparison; every upstream gate -- the search
benchmark and the independent isolated validation -- compared with whatever
tolerance happened to be lying around. For a ``byte_equal`` workload that is a
fail-open hole: a kernel using approximate hardware math (``rcp.approx.ftz``,
``tanh.approx``) passes a 1e-2 tolerance comfortably and is packaged, and the
run only discovers the problem hours later when full-generation parity fails.

The policy travels with the candidate so every stage answers the same question.

Policies
--------
``byte_equal``
    Outputs must be bitwise identical to the reference. Tolerances are not
    consulted at all, adversarial-input relaxation is disabled, and the tensor
    contract (shape, dtype, and layout where it is observable) must match.
``tolerance``
    Outputs must agree within the tolerance declared *for their own dtype*. A
    dtype with no declared tolerance is an error, not an invitation to guess.
``frames_only``
    Only the final decoded frames are contracted; intermediate kernels are held
    to the declared tolerance. Treated as ``tolerance`` for kernel-level gates.

This module never imports torch.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..specs.types import SpecValidationError, Tolerance

__all__ = [
    "APPROXIMATE_MATH_MARKERS",
    "EXACT_POLICIES",
    "KNOWN_POLICIES",
    "ParityPolicy",
    "ToleranceResolutionError",
    "detect_approximate_math",
    "resolve_leaf_tolerance",
]

#: Policies that admit no numerical difference whatsoever.
EXACT_POLICIES = frozenset({"byte_equal"})

#: Every policy name a workload may declare. Mirrors
#: ``autokernel.workload.types._PARITY_POLICIES``; kept as a separate constant
#: so the verification package stays importable without the workload package.
KNOWN_POLICIES = frozenset({"byte_equal", "tolerance", "frames_only"})

#: PTX/intrinsic fragments that make a kernel's arithmetic non-reproducible
#: against an eager reference. These are legitimate optimizations -- but only
#: for a workload that has explicitly opted into approximate parity.
APPROXIMATE_MATH_MARKERS = (
    "rcp.approx",
    "tanh.approx",
    "sin.approx",
    "cos.approx",
    "lg2.approx",
    "ex2.approx",
    "rsqrt.approx",
    "sqrt.approx.f32",
    "div.approx",
    "fdividef",
    "__expf",
    "__logf",
    "__powf",
    "__sinf",
    "__cosf",
    "__tanf",
    "allow_tf32",
)


class ToleranceResolutionError(ValueError):
    """A leaf dtype has no tolerance and the policy forbids guessing one."""


@dataclass(frozen=True)
class ParityPolicy:
    """How strictly a candidate's outputs must match the reference.

    Args:
        policy: one of :data:`KNOWN_POLICIES`.
        atol / rtol: optional workload-level override applied to every floating
            leaf. Ignored entirely under an exact policy.
        allow_approximate_math: when false (the default for an exact policy) a
            candidate whose source contains approximate-math intrinsics is
            rejected before it is ever benchmarked.
        max_absolute_error: hard ceiling on absolute error, applied *in addition
            to* atol/rtol. This is what stops a large-magnitude tensor from
            hiding a 32768.0 error behind a 1% relative tolerance.
        relax_allowed: whether the numerical-stability stage may widen
            tolerances for adversarial inputs.
    """

    policy: str = "byte_equal"
    atol: float | None = None
    rtol: float | None = None
    allow_approximate_math: bool | None = None
    max_absolute_error: float | None = None
    relax_allowed: bool | None = None

    def __post_init__(self) -> None:
        if self.policy not in KNOWN_POLICIES:
            raise SpecValidationError(
                f"unknown parity policy {self.policy!r}; "
                f"expected one of {sorted(KNOWN_POLICIES)}"
            )
        for field_name in ("atol", "rtol", "max_absolute_error"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SpecValidationError(
                    f"ParityPolicy.{field_name} must be a number, got {value!r}"
                )
            if not math.isfinite(value) or value < 0:
                raise SpecValidationError(
                    f"ParityPolicy.{field_name} must be finite and non-negative, "
                    f"got {value!r}"
                )

    # -- derived predicates ---------------------------------------------

    @property
    def exact(self) -> bool:
        """True when only bitwise-identical outputs are acceptable."""
        return self.policy in EXACT_POLICIES

    @property
    def approximate_math_allowed(self) -> bool:
        """Whether approximate hardware intrinsics may appear in a kernel."""
        if self.allow_approximate_math is not None:
            return bool(self.allow_approximate_math)
        return not self.exact

    @property
    def relaxation_allowed(self) -> bool:
        """Whether the stability stage may widen tolerances."""
        if self.relax_allowed is not None:
            return bool(self.relax_allowed)
        return not self.exact

    @property
    def requires_declared_tolerance(self) -> bool:
        """Whether an undeclared leaf dtype is an error rather than a default.

        Under any policy, silently substituting 1e-2/1e-2 for a dtype the spec
        never described is how an approximate kernel got packaged in R4. The
        workload may still supply a blanket atol/rtol; what is forbidden is the
        *harness* inventing one.
        """
        return self.atol is None or self.rtol is None

    def effective_relax(self, relax: float) -> float:
        """Clamp a stability-stage relaxation factor to what the policy allows."""
        if not self.relaxation_allowed:
            return 1.0
        return relax

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "atol": self.atol,
            "rtol": self.rtol,
            "exact": self.exact,
            "allow_approximate_math": self.approximate_math_allowed,
            "max_absolute_error": self.max_absolute_error,
            "relax_allowed": self.relaxation_allowed,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> ParityPolicy:
        """Build from a manifest/JSON fragment, tolerating absent fields."""
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise SpecValidationError(
                f"parity policy must be a mapping, got {type(raw).__name__}"
            )
        return cls(
            policy=str(raw.get("policy", "byte_equal")),
            atol=raw.get("atol"),
            rtol=raw.get("rtol"),
            allow_approximate_math=raw.get("allow_approximate_math"),
            max_absolute_error=raw.get("max_absolute_error"),
            relax_allowed=raw.get("relax_allowed"),
        )

    @classmethod
    def from_workload(cls, workload: Any) -> ParityPolicy:
        """Derive the policy from a loaded workload definition.

        Accepts anything exposing a ``parity`` attribute with ``policy``/
        ``atol``/``rtol``; a workload with no parity block defaults to the
        strictest policy, because failing closed is the point.
        """
        parity = getattr(workload, "parity", None)
        if parity is None:
            return cls()
        return cls(
            policy=str(getattr(parity, "policy", "byte_equal")),
            atol=getattr(parity, "atol", None),
            rtol=getattr(parity, "rtol", None),
        )


def resolve_leaf_tolerance(
    dtype_name: str | None,
    tolerances: Mapping[str, Tolerance],
    policy: ParityPolicy,
    *,
    default: Tolerance | None = None,
    path: str = "output",
) -> Tolerance:
    """Pick the tolerance for one floating leaf under ``policy``.

    Resolution order:

    1. an exact policy pins every leaf to ``atol=rtol=0``;
    2. a workload-level ``atol``/``rtol`` override applies to every leaf;
    3. the tolerance the spec declared for this leaf's own dtype;
    4. ``default`` -- but only when the policy permits an undeclared dtype.

    Raises:
        ToleranceResolutionError: when no tolerance is declared for
            ``dtype_name`` and the policy forbids inventing one. This converts
            the old silent fail-open into a loud failure.
    """
    if policy.exact:
        return Tolerance(atol=0.0, rtol=0.0)
    if policy.atol is not None and policy.rtol is not None:
        return Tolerance(atol=policy.atol, rtol=policy.rtol)
    if dtype_name is not None:
        declared = tolerances.get(dtype_name)
        if declared is not None:
            return declared
    if default is not None and not policy.requires_declared_tolerance:
        return default
    if default is not None and tolerances:
        # The spec described *some* dtypes but not this one. That is a spec gap
        # worth reporting rather than papering over.
        raise ToleranceResolutionError(
            f"{path}: no tolerance declared for dtype {dtype_name!r} under "
            f"parity policy {policy.policy!r}; declared dtypes are "
            f"{sorted(tolerances)}"
        )
    raise ToleranceResolutionError(
        f"{path}: parity policy {policy.policy!r} requires an explicit "
        f"tolerance for dtype {dtype_name!r}, but the spec declares none and "
        f"the workload supplies no atol/rtol override"
    )


def detect_approximate_math(source: str) -> tuple[str, ...]:
    """Return the approximate-math markers present in ``source``.

    Used to reject a kernel that cannot possibly satisfy an exact policy
    *before* spending GPU time benchmarking it. Purely textual: it is a cheap
    screen, not a proof, and it never relaxes any downstream numeric gate.
    """
    if not source:
        return ()
    lowered = source.lower()
    return tuple(
        marker for marker in APPROXIMATE_MATH_MARKERS if marker.lower() in lowered
    )
