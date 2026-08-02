"""Tiered fidelity contracts: how much output drift a workload will buy speed with.

:mod:`autokernel.verification.policy` answers "how closely must these tensors
match?". That question has only one honest answer for a kernel that is meant to
be a drop-in replacement: exactly. It is the right contract for a fused
LayerNorm and the wrong one for the two families this project is moving toward.

A different attention backend (SageAttention2) and a cross-step cache
(TeaCache) are not bit-exact and cannot be made bit-exact -- the first changes
the accumulation order and numeric format of the attention product, the second
*skips recomputing* a step and reuses a previous result. Under ``byte_equal``
both are rejected before they are ever benchmarked. Under a loosened numeric
tolerance both pass for the wrong reason: a tolerance wide enough to admit a
cache hit is wide enough to admit a broken kernel, which is precisely how R4
packaged four ``rcp.approx.ftz`` VAE artifacts carrying 131072.0 of absolute
error at ``match=true``.

The resolution is to stop widening the numeric axis and add a second one. A
workload declares a *fidelity tier* saying which axis its outputs are contracted
on:

``exact`` (tier 1)
    Bitwise identity, enforced by the parity policy. Perceptual evidence is not
    consulted, because there is nothing left for it to decide. This is what the
    promoted LTX2 CUDA-graph artifact satisfies.

``perceptual`` (tier 2)
    Numerics may drift; the *rendered frames* must remain perceptually
    indistinguishable from the reference generation at a fixed seed. Gated on
    declared SSIM/LPIPS/VBench thresholds. Attention backends and schedule
    transforms are promoted here or not at all.

``advisory`` (tier 3)
    Quality is measured and recorded but never gates. An advisory artifact is
    **never** automatically promoted -- the tier exists so discovery campaigns
    can explore aggressive transforms and leave evidence behind, not so a
    promotion can be waved through.

Thresholds are declared per workload and never invented by the harness. This is
the same rule :func:`~autokernel.verification.policy.resolve_leaf_tolerance`
enforces on tolerances, for the same reason: a 1.3B model at 480p and a 14B
model at 720p do not share a perceptual noise floor, and a default that fits
neither is a fail-open hole wearing a number.

This module never imports torch. Metric *computation* lives in
:mod:`autokernel.verification.perceptual`; what lives here is the contract and
the gate, so a promotion decision can be tested without a GPU.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..specs.types import SpecValidationError

__all__ = [
    "ADVISORY",
    "EXACT",
    "KNOWN_TIERS",
    "PERCEPTUAL",
    "TIER_NUMBERS",
    "FidelityBudget",
    "FidelityError",
    "FidelityVerdict",
    "MetricMargin",
    "PerceptualEvidence",
    "evaluate_fidelity",
    "tier_number",
]

#: Tier 1 -- bitwise identity, decided entirely by the parity policy.
EXACT = "exact"
#: Tier 2 -- perceptually indistinguishable rendered output.
PERCEPTUAL = "perceptual"
#: Tier 3 -- measured and recorded, never gating, never auto-promoted.
ADVISORY = "advisory"

#: Every tier a workload may declare.
KNOWN_TIERS = frozenset({EXACT, PERCEPTUAL, ADVISORY})

#: The plan and the docs refer to these by number; keep one mapping so
#: "Tier 2" and ``perceptual`` can never drift apart.
TIER_NUMBERS: Mapping[str, int] = {EXACT: 1, PERCEPTUAL: 2, ADVISORY: 3}

#: Metrics whose value must be *at least* the declared threshold.
_MINIMUM_METRICS = ("ssim", "vbench")
#: Metrics whose value must be *at most* the declared threshold.
_MAXIMUM_METRICS = ("lpips",)

#: Closed ranges each metric must fall in. A metric outside its own range is a
#: broken measurement, not a bad candidate, and is reported as such: an SSIM of
#: 1.4 means the harness is miscomputing, and silently comparing it against a
#: 0.98 floor would report a pass.
_METRIC_RANGES: Mapping[str, tuple[float, float]] = {
    "ssim": (-1.0, 1.0),
    "lpips": (0.0, math.inf),
    "vbench": (0.0, 1.0),
}


class FidelityError(ValueError):
    """A fidelity contract is malformed, or its evidence cannot be trusted."""


def tier_number(tier: str) -> int:
    """Return the 1/2/3 number for ``tier``, for docs and report rendering."""
    try:
        return TIER_NUMBERS[tier]
    except KeyError:
        raise FidelityError(
            f"unknown fidelity tier {tier!r}; expected one of {sorted(KNOWN_TIERS)}"
        ) from None


def _check_threshold(name: str, value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FidelityError(f"fidelity threshold {name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise FidelityError(f"fidelity threshold {name} must be finite, got {value!r}")
    low, high = _METRIC_RANGES[name]
    if not low <= number <= high:
        raise FidelityError(
            f"fidelity threshold {name}={number!r} is outside the metric's "
            f"range [{low}, {high}]"
        )
    return number


@dataclass(frozen=True)
class MetricMargin:
    """How much room one metric had against its threshold.

    The margin is always signed so that **positive means passing**, whichever
    direction the metric runs. A promotion record that says "SSIM 0.991,
    threshold 0.98, margin +0.011" is auditable months later; one that says
    "passed" is not.
    """

    metric: str
    value: float
    threshold: float
    margin: float
    passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "margin": self.margin,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class FidelityBudget:
    """The fidelity contract a workload declares.

    Args:
        tier: one of :data:`KNOWN_TIERS`.
        min_ssim: structural-similarity floor against the reference frames.
        max_lpips: perceptual-distance ceiling. Lower is more similar.
        min_vbench: floor on the VBench-subset score, when one is scored.
        frame_set: identifier of the fixed-seed frame set the evidence must
            have been measured on. Recorded so a promotion cannot be justified
            by evidence from a different set of frames.
        seed: the generation seed the frame set was produced at.

    A ``perceptual`` budget must declare at least one threshold. A budget that
    names the tier but sets no bar is not a weaker contract, it is *no*
    contract, and it would promote anything that ran.
    """

    tier: str = EXACT
    min_ssim: float | None = None
    max_lpips: float | None = None
    min_vbench: float | None = None
    frame_set: str = ""
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.tier not in KNOWN_TIERS:
            raise FidelityError(
                f"unknown fidelity tier {self.tier!r}; "
                f"expected one of {sorted(KNOWN_TIERS)}"
            )
        object.__setattr__(self, "min_ssim", _check_threshold("ssim", self.min_ssim))
        object.__setattr__(self, "max_lpips", _check_threshold("lpips", self.max_lpips))
        object.__setattr__(
            self, "min_vbench", _check_threshold("vbench", self.min_vbench)
        )
        if not isinstance(self.frame_set, str):
            raise FidelityError("fidelity frame_set must be a string")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise FidelityError(f"fidelity seed must be an integer, got {self.seed!r}")

        if self.tier == PERCEPTUAL:
            if not self.declared_metrics:
                raise FidelityError(
                    "a 'perceptual' fidelity budget must declare at least one of "
                    "min_ssim/max_lpips/min_vbench; a tier with no threshold "
                    "promotes anything that runs"
                )
            if not self.frame_set:
                raise FidelityError(
                    "a 'perceptual' fidelity budget must name the fixed-seed "
                    "frame_set its evidence is measured on"
                )
        if self.tier == EXACT and self.declared_metrics:
            raise FidelityError(
                "an 'exact' fidelity budget must not declare perceptual "
                "thresholds; bitwise identity is decided by the parity policy "
                "and a perceptual bar could only weaken it"
            )

    @property
    def declared_metrics(self) -> tuple[str, ...]:
        """Metric names this budget actually gates on, in report order."""
        declared = []
        if self.min_ssim is not None:
            declared.append("ssim")
        if self.max_lpips is not None:
            declared.append("lpips")
        if self.min_vbench is not None:
            declared.append("vbench")
        return tuple(declared)

    def threshold_for(self, metric: str) -> float | None:
        return {
            "ssim": self.min_ssim,
            "lpips": self.max_lpips,
            "vbench": self.min_vbench,
        }.get(metric)

    @property
    def number(self) -> int:
        return tier_number(self.tier)

    @property
    def gates_on_perception(self) -> bool:
        """Whether a promotion at this tier requires perceptual evidence."""
        return self.tier == PERCEPTUAL

    @property
    def auto_promotable(self) -> bool:
        """Whether a passing measurement at this tier may promote on its own.

        Advisory artifacts never may. The tier is for leaving evidence behind,
        and an automated pipeline that could promote from it would make the
        distinction from ``perceptual`` decorative.
        """
        return self.tier in (EXACT, PERCEPTUAL)

    def as_manifest_dict(self) -> dict[str, Any]:
        """The declared form, round-trippable through :meth:`from_dict`.

        Distinct from :meth:`as_dict` because that one adds the derived
        ``tier_number`` for report readers, and a workload manifest that
        re-parsed its own serialized output would reject the extra field.
        """
        payload = self.as_dict()
        payload.pop("tier_number", None)
        return payload

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"tier": self.tier, "tier_number": self.number}
        if self.min_ssim is not None:
            payload["min_ssim"] = self.min_ssim
        if self.max_lpips is not None:
            payload["max_lpips"] = self.max_lpips
        if self.min_vbench is not None:
            payload["min_vbench"] = self.min_vbench
        if self.frame_set:
            payload["frame_set"] = self.frame_set
        if self.seed is not None:
            payload["seed"] = self.seed
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> FidelityBudget:
        """Build from a manifest/YAML fragment.

        An absent block means ``exact``: a workload that never considered the
        question gets the strictest contract, not the most convenient one.
        """
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise FidelityError(
                f"fidelity budget must be a mapping, got {type(raw).__name__}"
            )
        unknown = set(raw) - {
            "tier",
            "min_ssim",
            "max_lpips",
            "min_vbench",
            "frame_set",
            "seed",
        }
        if unknown:
            raise FidelityError(
                f"unknown fidelity budget fields: {sorted(unknown)}"
            )
        return cls(
            tier=str(raw.get("tier", EXACT)),
            min_ssim=raw.get("min_ssim"),
            max_lpips=raw.get("max_lpips"),
            min_vbench=raw.get("min_vbench"),
            frame_set=str(raw.get("frame_set", "")),
            seed=raw.get("seed"),
        )

    @classmethod
    def from_workload(cls, workload: Any) -> FidelityBudget:
        """Derive the budget from a loaded workload definition.

        A workload with no fidelity block defaults to ``exact``, matching
        :meth:`ParityPolicy.from_workload`'s choice to fail closed.
        """
        fidelity = getattr(workload, "fidelity", None)
        if fidelity is None:
            return cls()
        if isinstance(fidelity, FidelityBudget):
            return fidelity
        if isinstance(fidelity, Mapping):
            return cls.from_dict(fidelity)
        return cls(
            tier=str(getattr(fidelity, "tier", EXACT)),
            min_ssim=getattr(fidelity, "min_ssim", None),
            max_lpips=getattr(fidelity, "max_lpips", None),
            min_vbench=getattr(fidelity, "min_vbench", None),
            frame_set=str(getattr(fidelity, "frame_set", "") or ""),
            seed=getattr(fidelity, "seed", None),
        )


@dataclass(frozen=True)
class PerceptualEvidence:
    """Measured perceptual metrics from one fixed-seed frame comparison.

    Carries measurements, never conclusions -- :func:`evaluate_fidelity` derives
    the verdict, so the gate lives in one place and is testable without a GPU
    (the same split :class:`~autokernel.artifact.finalizer.GenerationOutcome`
    already uses).

    ``frame_set`` and ``seed`` are compared against the budget's. Evidence from
    a different frame set is not weak evidence, it is evidence about a different
    question, and it is refused rather than discounted.
    """

    frame_set: str
    seed: int
    frames_compared: int
    ssim: float | None = None
    lpips: float | None = None
    vbench: float | None = None
    stage_status: str = "ok"

    def __post_init__(self) -> None:
        if not isinstance(self.frame_set, str) or not self.frame_set.strip():
            raise FidelityError("perceptual evidence: frame_set must be non-empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise FidelityError("perceptual evidence: seed must be an integer")
        if (
            isinstance(self.frames_compared, bool)
            or not isinstance(self.frames_compared, int)
            or self.frames_compared <= 0
        ):
            raise FidelityError(
                "perceptual evidence: frames_compared must be a positive integer"
            )
        if self.stage_status not in {"ok", "failed"}:
            raise FidelityError(
                "perceptual evidence: stage_status must be 'ok' or 'failed'"
            )
        for name in ("ssim", "lpips", "vbench"):
            object.__setattr__(
                self, name, _check_threshold(name, getattr(self, name))
            )

    @property
    def measured(self) -> bool:
        """Whether the harness produced usable numbers."""
        return self.stage_status == "ok" and bool(self.available_metrics)

    @property
    def available_metrics(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in ("ssim", "lpips", "vbench")
            if getattr(self, name) is not None
        )

    def value_for(self, metric: str) -> float | None:
        return getattr(self, metric, None)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "frame_set": self.frame_set,
            "seed": self.seed,
            "frames_compared": self.frames_compared,
            "stage_status": self.stage_status,
        }
        for name in ("ssim", "lpips", "vbench"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload


@dataclass(frozen=True)
class FidelityVerdict:
    """The outcome of checking evidence against a budget."""

    tier: str
    passed: bool
    reason: str
    margins: tuple[MetricMargin, ...] = ()
    evidence_required: bool = False

    @property
    def number(self) -> int:
        return tier_number(self.tier)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "tier_number": self.number,
            "passed": self.passed,
            "reason": self.reason,
            "evidence_required": self.evidence_required,
            "margins": [margin.as_dict() for margin in self.margins],
        }


def evaluate_fidelity(
    budget: FidelityBudget,
    evidence: PerceptualEvidence | None,
    *,
    parity_passed: bool | None = None,
) -> FidelityVerdict:
    """Decide whether ``evidence`` satisfies ``budget``.

    Args:
        budget: the workload's declared contract.
        evidence: measured perceptual metrics, or None when the stage did not
            run. Required at the ``perceptual`` tier and ignored at ``exact``.
        parity_passed: the numeric parity result, consulted only at ``exact``
            where it is the whole contract. ``None`` means "not reported",
            which at that tier is missing evidence rather than a pass.

    Returns:
        A :class:`FidelityVerdict` carrying the decision, a human-readable
        reason, and the signed margin for every gated metric. Margins are
        recorded on passes as well as failures -- a promotion that cleared its
        SSIM floor by 0.0004 is a fact worth having in the manifest.

    Every failure path returns a verdict rather than raising. A malformed
    *contract* raises (that is a bug in the workload), but a candidate that
    misses its bar is an ordinary negative result.
    """
    if budget.tier == EXACT:
        if parity_passed is None:
            return FidelityVerdict(
                tier=budget.tier,
                passed=False,
                reason=(
                    "held: tier 1 (exact) requires a bitwise parity result and "
                    "none was reported"
                ),
            )
        if parity_passed:
            return FidelityVerdict(
                tier=budget.tier,
                passed=True,
                reason="tier 1 (exact): outputs are bitwise identical",
            )
        return FidelityVerdict(
            tier=budget.tier,
            passed=False,
            reason="tier 1 (exact): outputs are not bitwise identical",
        )

    if budget.tier == ADVISORY:
        # Advisory never gates and never promotes. Report what was measured so
        # the campaign has a record, and say plainly why this cannot promote.
        margins = _margins_for(budget, evidence) if evidence is not None else ()
        return FidelityVerdict(
            tier=budget.tier,
            passed=False,
            reason=(
                "tier 3 (advisory): quality recorded but this tier never "
                "auto-promotes; promotion requires a human decision"
            ),
            margins=margins,
        )

    # -- tier 2, the only tier that actually gates on perception ----------
    if evidence is None:
        return FidelityVerdict(
            tier=budget.tier,
            passed=False,
            reason=(
                "held: tier 2 (perceptual) requires perceptual evidence and the "
                "harness produced none"
            ),
            evidence_required=True,
        )
    if evidence.stage_status != "ok":
        return FidelityVerdict(
            tier=budget.tier,
            passed=False,
            reason=(
                "held: tier 2 (perceptual) evidence is incomplete "
                f"(stage_status={evidence.stage_status!r})"
            ),
            evidence_required=True,
        )
    if budget.frame_set and evidence.frame_set != budget.frame_set:
        return FidelityVerdict(
            tier=budget.tier,
            passed=False,
            reason=(
                "held: perceptual evidence was measured on frame set "
                f"{evidence.frame_set!r}, but the budget contracts "
                f"{budget.frame_set!r}"
            ),
            evidence_required=True,
        )
    if budget.seed is not None and evidence.seed != budget.seed:
        return FidelityVerdict(
            tier=budget.tier,
            passed=False,
            reason=(
                f"held: perceptual evidence was measured at seed {evidence.seed}, "
                f"but the budget contracts seed {budget.seed}"
            ),
            evidence_required=True,
        )

    missing = [
        metric
        for metric in budget.declared_metrics
        if evidence.value_for(metric) is None
    ]
    if missing:
        return FidelityVerdict(
            tier=budget.tier,
            passed=False,
            reason=(
                "held: the budget gates on "
                f"{', '.join(missing)} but the harness measured "
                f"{', '.join(evidence.available_metrics) or 'nothing'}"
            ),
            evidence_required=True,
        )

    margins = _margins_for(budget, evidence)
    failed = [margin for margin in margins if not margin.passed]
    detail = ", ".join(
        f"{m.metric}={m.value:.6g} vs {m.threshold:.6g} (margin {m.margin:+.6g})"
        for m in margins
    )
    if failed:
        return FidelityVerdict(
            tier=budget.tier,
            passed=False,
            reason=(
                "tier 2 (perceptual): "
                f"{', '.join(m.metric for m in failed)} outside budget over "
                f"{evidence.frames_compared} frames -- {detail}"
            ),
            margins=margins,
            evidence_required=True,
        )
    return FidelityVerdict(
        tier=budget.tier,
        passed=True,
        reason=(
            "tier 2 (perceptual): within budget over "
            f"{evidence.frames_compared} frames -- {detail}"
        ),
        margins=margins,
        evidence_required=True,
    )


def _margins_for(
    budget: FidelityBudget, evidence: PerceptualEvidence
) -> tuple[MetricMargin, ...]:
    """Signed margins for every metric the budget gates on and evidence has."""
    margins: list[MetricMargin] = []
    for metric in budget.declared_metrics:
        threshold = budget.threshold_for(metric)
        value = evidence.value_for(metric)
        if threshold is None or value is None:
            continue
        if metric in _MAXIMUM_METRICS:
            margin = threshold - value
        else:
            margin = value - threshold
        margins.append(
            MetricMargin(
                metric=metric,
                value=value,
                threshold=threshold,
                margin=margin,
                passed=margin >= 0.0,
            )
        )
    return tuple(margins)
