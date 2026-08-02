"""Candidate impact ranking and Amdahl-style end-to-end ceilings.

Uses measured CUDA time share and an optimistic reducible fraction to decide
whether a region is worth searching. Defaults match the universal plan:

- impact floor: do not search when optimistic e2e improvement is below 0.5%
- promotion preference: candidates that can plausibly exceed 1% e2e gain
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .safety import reject_region
from .types import GraphRegion, OperatorHotspot

DEFAULT_IMPACT_FLOOR = 0.005  # 0.5% optimistic end-to-end
DEFAULT_PROMOTION_TARGET = 0.01  # 1%
DEFAULT_REDUCIBLE_FRACTION = 0.9  # optimistic upper bound on kernel speedup


@dataclass(frozen=True)
class RankedCandidate:
    """One discovery region with model-level value estimates."""

    region: GraphRegion
    share_of_e2e: float
    estimated_reducible_fraction: float
    estimated_max_e2e_improvement: float
    confidence: float
    search_worthy: bool
    meets_promotion_target: bool
    rejection_reasons: tuple[str, ...]
    pattern_family: str | None = None
    selection_mode: str = "whole_region"
    parent_rejection_reasons: tuple[str, ...] = ()
    selected_node_count: int | None = None
    impact_estimate_kind: str = "measured_region"

    def as_dict(self) -> dict:
        return {
            "name": self.region.name,
            "fingerprint": self.region.fingerprint,
            "share_of_e2e": self.share_of_e2e,
            "estimated_reducible_fraction": self.estimated_reducible_fraction,
            "estimated_max_e2e_improvement": self.estimated_max_e2e_improvement,
            "confidence": self.confidence,
            "search_worthy": self.search_worthy,
            "meets_promotion_target": self.meets_promotion_target,
            "rejection_reasons": list(self.rejection_reasons),
            "selection_mode": self.selection_mode,
            "parent_rejection_reasons": list(self.parent_rejection_reasons),
            "selected_node_count": self.selected_node_count,
            "impact_estimate_kind": self.impact_estimate_kind,
            "pattern_family": self.pattern_family or self.region.pattern_family,
            "calls": self.region.calls,
            "cuda_time_us": self.region.cuda_time_us,
        }


def e2e_share(cuda_time_us: float, total_cuda_time_us: float) -> float:
    """Fraction of end-to-end CUDA time attributable to one region.

    Clamped to 1.0. A region cannot occupy more than the whole model, but the
    inputs can say otherwise: attributed time sums inclusive record_function
    ranges while the total sums exclusive operator time, so nested or
    overlapping ranges can push the numerator past the denominator. Run r4
    reported share_of_e2e = 1.0333 for transformer.model.transformer_blocks
    (2864051.95us attributed against a 2771790.13us total), which fed an
    "optimistic improvement" of 93% of end-to-end into candidate ranking.

    Clamping keeps the ratio interpretable; it does not make an over-attributed
    region trustworthy, which is why measured_e2e_improvement supersedes this
    estimate as soon as a kernel has actually been benchmarked.
    """
    if total_cuda_time_us <= 0:
        return 0.0
    return min(1.0, max(0.0, float(cuda_time_us) / float(total_cuda_time_us)))


def optimistic_e2e_improvement(
    share: float,
    *,
    reducible_fraction: float = DEFAULT_REDUCIBLE_FRACTION,
) -> float:
    """Amdahl-style upper bound: share * reducible_fraction.

    If a region is 2% of e2e and we can remove 90% of its time, the model
    improves by at most ~1.8%.
    """
    if reducible_fraction < 0.0 or reducible_fraction > 1.0:
        raise ValueError("reducible_fraction must be in [0, 1]")
    return max(0.0, share * reducible_fraction)


def measured_e2e_improvement(share: float, measured_speedup: float) -> float:
    """End-to-end gain a region actually delivers at ``measured_speedup``.

    :func:`optimistic_e2e_improvement` answers "is this region worth searching
    at all", assuming the kernel's cost drops to nearly zero. Once a kernel has
    been benchmarked that assumption is no longer needed, and keeping it is
    actively misleading: a region that is 2.3% of end-to-end running 1.115x
    faster returns ``0.023 * (1 - 1/1.115)`` = 0.24% of end-to-end, not the
    2.1% the upper bound predicted -- an order of magnitude apart.

    Run r4 packaged four VAE artifacts on the strength of the upper bound. Their
    measured speedups summed to 0.63% of end-to-end against a 1% campaign
    target, so no combination of them could ever have passed, and the run spent
    a full A/B generation pair discovering that.
    """
    if measured_speedup <= 0.0:
        raise ValueError("measured_speedup must be positive")
    if measured_speedup <= 1.0:
        return 0.0
    return max(0.0, share * (1.0 - 1.0 / measured_speedup))


def measured_e2e_improvement_from_latency(
    *,
    baseline_us: float,
    candidate_us: float,
    calls_per_generation: float,
    total_generation_us: float,
) -> float:
    """End-to-end gain from an absolute per-call saving and a real call count.

    Prefer this over :func:`measured_e2e_improvement` for a
    ``derived_subregion`` candidate. There, ``share_of_e2e`` describes the
    *parent* region while the benchmark measures only the selected subregion,
    so multiplying the two attributes the subregion's speedup to the whole
    parent. For the repaired r4 transformer candidate -- 22 selected nodes of
    ``transformer.model.transformer_blocks``, measured at 259.05us -> 134.90us
    -- the share-based form claims 47.9% of end-to-end. The saving is
    124.15us across 384 invocations per generation against a 3281831us
    generation: 1.45%.

    Both figures come from measurements; only one answers the question.
    """
    if total_generation_us <= 0:
        raise ValueError("total_generation_us must be positive")
    if calls_per_generation < 0:
        raise ValueError("calls_per_generation must not be negative")
    saving_us = (baseline_us - candidate_us) * calls_per_generation
    return max(0.0, saving_us / total_generation_us)


def projected_end_to_end_speedup(
    improvements: Sequence[float],
    *,
    dispatch_overhead_fraction: float = 0.0,
) -> float:
    """Combine per-region end-to-end gains into a projected model speedup.

    Regions are assumed disjoint, so their fractional savings add. Dispatch
    overhead is charged against the total: every replaced region pays for graph
    matching, parameter materialization and an extra Python frame per call, and
    a candidate whose savings do not exceed that cost is a regression however
    good its isolated benchmark looked.
    """
    total = sum(max(0.0, value) for value in improvements)
    net = total - max(0.0, dispatch_overhead_fraction)
    if net >= 1.0:  # pragma: no cover - defensive
        raise ValueError("combined improvement cannot reach 100% of end-to-end")
    return 1.0 / (1.0 - net)


def meets_end_to_end_target(
    improvements: Sequence[float],
    *,
    min_end_to_end_speedup: float,
    dispatch_overhead_fraction: float = 0.0,
) -> bool:
    """Whether measured savings can reach the campaign's speedup gate."""
    projected = projected_end_to_end_speedup(
        improvements, dispatch_overhead_fraction=dispatch_overhead_fraction
    )
    return projected >= min_end_to_end_speedup


def _probe_safe_subregion(region: GraphRegion) -> tuple[int | None, str | None]:
    """Return the safe node count for an executable parent, or a rejection.

    Discovery times repeated parent modules because that is the stable runtime
    boundary. An export graph may contain attention, mutation, or other nodes
    that cannot be searched while still enclosing a connected pure-tensor
    epilogue that can. Spec generation is the authoritative safety boundary,
    so ranking asks it to validate the exact same derived component instead of
    treating an unsafe parent as an unsafe replacement target.

    The import is deliberately late: ``specgen`` consumes discovery reports,
    while this optional probe only runs after both packages are initialized.
    """
    attributes = region.attributes
    if not isinstance(attributes, Mapping) or not isinstance(
        attributes.get("executable_ir"), Mapping
    ):
        return None, "safe_subregion_unavailable:no_executable_ir"
    try:
        from autokernel.specgen import SpecGenerationError, derive_safe_subregion

        derived = derive_safe_subregion(region)
    except (ImportError, SpecGenerationError) as exc:
        return None, f"safe_subregion_unavailable:{type(exc).__name__}:{exc}"
    return len(derived.ir.nodes), None


def classify_pattern_family(operations: Sequence[str]) -> str:
    """Coarse family label for reports; not a correctness claim."""
    ops = " ".join(operations).lower()
    if any(tok in ops for tok in ("layer_norm", "rms_norm", "native_layer_norm")):
        if any(tok in ops for tok in ("mul", "add")):
            return "residual_gate_norm"
        return "normalization"
    if any(tok in ops for tok in ("silu", "gelu", "relu", "sigmoid")):
        return "activation_epilogue"
    if any(
        tok in ops for tok in ("permute", "transpose", "contiguous", "clone", "to")
    ) and all(
            any(x in o for x in ("permute", "transpose", "contiguous", "clone", "to", "view", "reshape"))
            for o in operations
    ):
        return "layout_cast_copy"
    if any(tok in ops for tok in ("mul", "add", "sub", "div")):
        return "elementwise_chain"
    return "unknown"


def rank_regions(
    regions: Sequence[GraphRegion],
    *,
    total_cuda_time_us: float,
    impact_floor: float = DEFAULT_IMPACT_FLOOR,
    promotion_target: float = DEFAULT_PROMOTION_TARGET,
    reducible_fraction: float = DEFAULT_REDUCIBLE_FRACTION,
) -> tuple[RankedCandidate, ...]:
    """Rank graph regions by optimistic end-to-end value."""
    ranked: list[RankedCandidate] = []
    for region in regions:
        share = e2e_share(region.self_cuda_time_us, total_cuda_time_us)
        safety_reasons = tuple(reject_region(region.operations))
        if region.rejection_reasons:
            safety_reasons = tuple(
                dict.fromkeys([*region.rejection_reasons, *safety_reasons])
            )
        improvement = optimistic_e2e_improvement(
            share, reducible_fraction=reducible_fraction
        )
        safe = not safety_reasons
        selection_mode = "whole_region"
        parent_rejection_reasons: tuple[str, ...] = ()
        selected_node_count: int | None = None
        impact_estimate_kind = "measured_region"
        derivation_error: str | None = None
        if not safe:
            selected_node_count, derivation_error = _probe_safe_subregion(region)
            if selected_node_count is not None:
                selection_mode = "derived_subregion"
                parent_rejection_reasons = safety_reasons
                impact_estimate_kind = "parent_region_upper_bound"
        # Confidence: higher when more calls and pure allowlist.
        confidence = 0.4
        if safe:
            confidence += 0.3
        if region.calls >= 8:
            confidence += 0.2
        if region.shape_frequency:
            confidence += 0.1
        confidence = min(1.0, confidence)

        executable = safe or selection_mode == "derived_subregion"
        search_worthy = executable and improvement >= impact_floor
        reasons = [] if executable else list(safety_reasons)
        if not executable and derivation_error is not None:
            reasons.append(derivation_error)
        if executable and improvement < impact_floor:
            reasons.append(
                f"below_impact_floor: optimistic e2e {improvement:.4f} "
                f"< floor {impact_floor:.4f}"
            )
            search_worthy = False

        family = region.pattern_family or classify_pattern_family(
            region.operations
        )
        ranked.append(
            RankedCandidate(
                region=region,
                share_of_e2e=share,
                estimated_reducible_fraction=reducible_fraction,
                estimated_max_e2e_improvement=improvement,
                confidence=confidence,
                search_worthy=search_worthy,
                meets_promotion_target=improvement >= promotion_target,
                rejection_reasons=tuple(reasons),
                pattern_family=family,
                selection_mode=selection_mode,
                parent_rejection_reasons=parent_rejection_reasons,
                selected_node_count=selected_node_count,
                impact_estimate_kind=impact_estimate_kind,
            )
        )

    ranked.sort(
        key=lambda item: (
            -item.estimated_max_e2e_improvement,
            -item.share_of_e2e,
            -item.confidence,
            item.region.name,
        )
    )
    return tuple(ranked)


def rank_operators(
    operators: Sequence[OperatorHotspot],
    *,
    total_cuda_time_us: float,
) -> tuple[tuple[OperatorHotspot, float], ...]:
    """Return operators sorted by self CUDA e2e share (descending).

    ``cuda_time_us`` is inclusive in torch profiler exports, so ranking with it
    counts the same device work once for every enclosing record_function or
    custom-op scope. ``total_cuda_time_us`` is the sum of self CUDA time and
    operator shares must use the same accounting basis.
    """
    rows = [
        (op, e2e_share(op.self_cuda_time_us, total_cuda_time_us))
        for op in operators
    ]
    rows.sort(key=lambda item: (-item[1], -item[0].calls, item[0].name))
    return tuple(rows)
