"""CPU tests for tiered fidelity contracts.

The two tests this file exists to carry are
:func:`test_intentionally_lossy_control_is_rejected_at_tier_2` and
:func:`test_known_good_approximate_artifact_passes_with_recorded_margins`.
Everything else guards a way the tier could quietly become decorative.

The failure being designed against is R4's: four VAE artifacts built on
``rcp.approx.ftz.f32`` reached packaging carrying up to 131072.0 of absolute
error at ``match=true``, because a tolerance wide enough to admit them was wide
enough to admit anything. Tier 2 must not become the same hole with perceptual
metrics in it -- so an undeclared threshold is an error, evidence from the wrong
frame set is refused rather than discounted, and every margin is recorded on
passes as well as failures.

These tests never import torch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autokernel.artifact import GenerationOutcome, finalize_bundle, read_manifest
from autokernel.verification.fidelity import (
    ADVISORY,
    EXACT,
    PERCEPTUAL,
    FidelityBudget,
    FidelityError,
    PerceptualEvidence,
    evaluate_fidelity,
    tier_number,
)
from autokernel.workload.types import FidelitySpec, WorkloadError

from test_artifact_finalization import _bundle, _outcome

# A budget a Wan-style workload might declare for an attention swap.
WAN_T2 = FidelityBudget(
    tier=PERCEPTUAL,
    min_ssim=0.98,
    max_lpips=0.02,
    frame_set="wan-1.3b-fixed-seed-8",
    seed=1234,
)


def _evidence(**overrides) -> PerceptualEvidence:
    values = {
        "frame_set": "wan-1.3b-fixed-seed-8",
        "seed": 1234,
        "frames_compared": 8,
        "ssim": 0.9912,
        "lpips": 0.0071,
    }
    values.update(overrides)
    return PerceptualEvidence(**values)


# -- tier semantics -----------------------------------------------------


def test_tier_numbers_match_the_names_the_plan_uses() -> None:
    assert tier_number(EXACT) == 1
    assert tier_number(PERCEPTUAL) == 2
    assert tier_number(ADVISORY) == 3


def test_an_undeclared_budget_is_exact_not_permissive() -> None:
    assert FidelityBudget().tier == EXACT
    assert FidelityBudget.from_dict(None).tier == EXACT
    assert FidelityBudget.from_workload(object()).tier == EXACT


def test_perceptual_tier_must_declare_a_threshold() -> None:
    # A tier with no bar is not a weaker contract, it is no contract.
    with pytest.raises(FidelityError, match="at least one of"):
        FidelityBudget(tier=PERCEPTUAL, frame_set="fs")


def test_perceptual_tier_must_name_its_frame_set() -> None:
    with pytest.raises(FidelityError, match="frame_set"):
        FidelityBudget(tier=PERCEPTUAL, min_ssim=0.98)


def test_exact_tier_may_not_carry_perceptual_thresholds() -> None:
    # A perceptual bar under an exact tier could only ever weaken it.
    with pytest.raises(FidelityError, match="must not declare"):
        FidelityBudget(tier=EXACT, min_ssim=0.9)


def test_thresholds_outside_a_metric_range_are_rejected() -> None:
    with pytest.raises(FidelityError, match="outside the metric's range"):
        FidelityBudget(tier=PERCEPTUAL, min_ssim=1.4, frame_set="fs")


def test_unknown_tier_is_rejected() -> None:
    with pytest.raises(FidelityError, match="unknown fidelity tier"):
        FidelityBudget(tier="lossy")


# -- the gate -----------------------------------------------------------


def test_tier_1_defers_entirely_to_bitwise_parity() -> None:
    budget = FidelityBudget()
    assert evaluate_fidelity(budget, None, parity_passed=True).passed
    assert not evaluate_fidelity(budget, None, parity_passed=False).passed
    # Not reported is missing evidence, not a pass.
    assert not evaluate_fidelity(budget, None, parity_passed=None).passed


def test_tier_2_ignores_bitwise_parity_failure() -> None:
    """The point of tier 2: byte_equal fails by construction for these."""
    verdict = evaluate_fidelity(WAN_T2, _evidence(), parity_passed=False)
    assert verdict.passed
    assert verdict.number == 2


def test_tier_2_without_evidence_is_held_not_passed() -> None:
    verdict = evaluate_fidelity(WAN_T2, None, parity_passed=True)
    assert not verdict.passed
    assert "requires perceptual evidence" in verdict.reason


def test_tier_2_refuses_evidence_from_a_different_frame_set() -> None:
    verdict = evaluate_fidelity(WAN_T2, _evidence(frame_set="some-other-set"))
    assert not verdict.passed
    assert "frame set" in verdict.reason


def test_tier_2_refuses_evidence_from_a_different_seed() -> None:
    verdict = evaluate_fidelity(WAN_T2, _evidence(seed=999))
    assert not verdict.passed
    assert "seed 999" in verdict.reason


def test_tier_2_refuses_evidence_missing_a_gated_metric() -> None:
    budget = FidelityBudget(
        tier=PERCEPTUAL, min_ssim=0.98, min_vbench=0.8, frame_set="fs", seed=1
    )
    evidence = PerceptualEvidence(
        frame_set="fs", seed=1, frames_compared=4, ssim=0.99
    )
    verdict = evaluate_fidelity(budget, evidence)
    assert not verdict.passed
    assert "vbench" in verdict.reason


def test_tier_2_holds_when_the_harness_stage_failed() -> None:
    verdict = evaluate_fidelity(WAN_T2, _evidence(stage_status="failed"))
    assert not verdict.passed
    assert "incomplete" in verdict.reason


def test_tier_3_records_quality_but_never_auto_promotes() -> None:
    budget = FidelityBudget(tier=ADVISORY, min_ssim=0.5, frame_set="fs", seed=1)
    # Evidence that clears the bar by a mile still does not promote.
    evidence = PerceptualEvidence(
        frame_set="fs", seed=1, frames_compared=4, ssim=0.999
    )
    verdict = evaluate_fidelity(budget, evidence, parity_passed=True)
    assert not verdict.passed
    assert not budget.auto_promotable
    assert "never" in verdict.reason
    # ...but the measurement is still recorded.
    assert [m.metric for m in verdict.margins] == ["ssim"]


def test_margins_are_signed_so_positive_always_means_passing() -> None:
    verdict = evaluate_fidelity(WAN_T2, _evidence())
    by_metric = {m.metric: m for m in verdict.margins}
    # SSIM runs upward: value - threshold.
    assert by_metric["ssim"].margin == pytest.approx(0.9912 - 0.98)
    # LPIPS runs downward: threshold - value. Positive still means passing.
    assert by_metric["lpips"].margin == pytest.approx(0.02 - 0.0071)
    assert all(m.passed for m in verdict.margins)


# -- workload manifest surface ------------------------------------------


def test_workload_fidelity_block_round_trips() -> None:
    spec = FidelitySpec.from_dict(
        {
            "tier": "perceptual",
            "min_ssim": 0.98,
            "max_lpips": 0.02,
            "frame_set": "wan-1.3b-fixed-seed-8",
            "seed": 1234,
        },
        source="<test>",
        location="fidelity",
    )
    assert spec.tier == PERCEPTUAL
    assert FidelitySpec.from_dict(
        spec.as_dict(), source="<test>", location="fidelity"
    ) == spec


def test_workload_fidelity_errors_surface_as_workload_errors() -> None:
    with pytest.raises(WorkloadError, match="at least one of"):
        FidelitySpec.from_dict(
            {"tier": "perceptual", "frame_set": "fs"},
            source="<test>",
            location="fidelity",
        )


# -- exit criteria: the two artifacts Track B must classify correctly ----


def test_intentionally_lossy_control_is_rejected_at_tier_2(tmp_path: Path) -> None:
    """A deliberately degraded artifact must not survive a tier-2 promotion.

    This is the control. It is fast (1.21x, well past the 1.01x bar), its
    dispatch selected cleanly, and its end-to-end classification is 'improved'
    -- everything the speed gate looks at says promote. Only the perceptual
    budget stands between it and promotion, which is exactly the property that
    was missing in R4.
    """
    bundle = _bundle(tmp_path)
    lossy = _outcome(
        parity_passed=False,
        fidelity=WAN_T2,
        perceptual=_evidence(ssim=0.71, lpips=0.24),
    )

    decision, reason = lossy.decide()
    assert decision == "quarantined"
    assert "ssim" in reason and "lpips" in reason

    result = finalize_bundle(bundle, lossy)
    assert result.decision == "quarantined"
    # The bundle is left exactly as packaged -- only promoted/rejected rewrite.
    assert read_manifest(bundle).promotion.decision == "quarantined"


def test_known_good_approximate_artifact_passes_with_recorded_margins(
    tmp_path: Path,
) -> None:
    """A bit-inexact but perceptually clean artifact promotes, with evidence.

    ``parity_passed=False`` is the whole point: this artifact cannot be
    bitwise identical (it stands in for a SageAttention2 backend or a TeaCache
    schedule transform), and under tier 1 it would be quarantined before its
    quality was ever examined.
    """
    bundle = _bundle(tmp_path)
    good = _outcome(
        parity_passed=False,
        fidelity=WAN_T2,
        perceptual=_evidence(),
    )

    decision, reason = good.decide()
    assert decision == "promoted"
    assert "tier 2 (perceptual)" in reason

    result = finalize_bundle(bundle, good)
    assert result.decision == "promoted"

    manifest = read_manifest(bundle)
    assert manifest.promotion.decision == "promoted"

    # The margins that justified the promotion are in the manifest, not just
    # the word "passed".
    generation = manifest.as_dict()["evidence"]["generation"]
    recorded = generation["fidelity"]
    assert recorded["budget"]["tier"] == PERCEPTUAL
    assert recorded["budget"]["tier_number"] == 2
    assert recorded["evidence"]["frame_set"] == "wan-1.3b-fixed-seed-8"
    assert recorded["evidence"]["frames_compared"] == 8

    margins = {m["metric"]: m for m in recorded["verdict"]["margins"]}
    assert margins["ssim"]["value"] == pytest.approx(0.9912)
    assert margins["ssim"]["threshold"] == pytest.approx(0.98)
    assert margins["ssim"]["margin"] > 0
    assert margins["lpips"]["margin"] > 0
    assert all(m["passed"] for m in margins.values())


def test_tier_1_behaviour_is_unchanged_by_the_tier_machinery(
    tmp_path: Path,
) -> None:
    """The promoted LTX2 artifact's path must be byte-for-byte the same.

    A parity failure under an absent/exact budget still quarantines with the
    original wording, so existing dashboards and quarantine reports keep
    reading the same string.
    """
    outcome = _outcome(parity_passed=False)
    decision, reason = outcome.decide()
    assert decision == "quarantined"
    assert reason == "held: full-generation output parity (byte_equal) failed"

    # And an exact-tier promotion records no fidelity block at all.
    promoted = _outcome()
    evidence = promoted.generation_evidence(passed=True)
    assert "fidelity" not in evidence
