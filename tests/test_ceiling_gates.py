"""CPU tests for ceiling-derived promotion gates.

The rule is `gate = max(1.10, 1 + 0.5 * (ceiling - 1))`. These tests pin the
two properties that make it worth having over a flat gate: it does not demand
the impossible on a low-ceiling workload, and it does not accept a token win on
a high-ceiling one.
"""

from __future__ import annotations

import pytest

from autokernel.verification.ceiling import (
    CEILING_GATE_RULE,
    GATE_FLOOR,
    CeilingError,
    amdahl_ceiling,
    derive_gate,
    evaluate_gate,
)


# -- the ceiling ---------------------------------------------------------


def test_ceiling_matches_our_measured_shares() -> None:
    # Measured in docs/ATTENTION_CAMPAIGN_RESULTS.md round 2 step 1.
    assert amdahl_ceiling(0.1513) == pytest.approx(1.1783, abs=1e-4)
    assert amdahl_ceiling(0.2717) == pytest.approx(1.3730, abs=1e-4)
    assert amdahl_ceiling(0.3495) == pytest.approx(1.5372, abs=1e-4)


def test_zero_share_has_no_headroom() -> None:
    assert amdahl_ceiling(0.0) == 1.0


def test_a_share_of_one_is_refused_not_infinite() -> None:
    """An infinite ceiling derives an infinite gate and nothing promotes."""
    with pytest.raises(CeilingError, match="entirely this component"):
        amdahl_ceiling(1.0)


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan"), float("inf")])
def test_out_of_range_shares_are_refused(bad) -> None:
    with pytest.raises(CeilingError):
        amdahl_ceiling(bad)


# -- the derived gate ----------------------------------------------------


def test_the_floor_binds_on_a_low_ceiling_workload() -> None:
    """ltx-480p: half of 17.8% headroom is 8.9%, below the 1.10 floor."""
    gate = derive_gate(amdahl_ceiling(0.1513))
    assert gate == pytest.approx(GATE_FLOOR)


def test_wan_480p_lands_between_1_19_and_1_27() -> None:
    """The range in the brief, from the measured share range."""
    low = derive_gate(amdahl_ceiling(0.2717))
    high = derive_gate(amdahl_ceiling(0.3495))
    assert low == pytest.approx(1.1865, abs=1e-3)
    assert high == pytest.approx(1.2686, abs=1e-3)


def test_a_high_ceiling_demands_a_high_gate() -> None:
    """A flat 1.10 would promote something capturing 7% of what was available."""
    assert derive_gate(2.5) == pytest.approx(1.75)


def test_the_gate_asks_for_half_the_headroom() -> None:
    for ceiling in (1.4, 1.8, 2.2, 3.0):
        gate = derive_gate(ceiling)
        assert gate - 1.0 == pytest.approx(0.5 * (ceiling - 1.0))


def test_a_ceiling_below_one_is_refused() -> None:
    with pytest.raises(CeilingError, match="at least 1.0"):
        derive_gate(0.9)


# -- unreachable gates are named, not hidden -----------------------------


def test_a_gate_above_its_own_ceiling_is_flagged_unreachable() -> None:
    """On a very low ceiling the 1.10 floor exceeds what is achievable.

    Not a contradiction to paper over: it means no candidate can pass, and the
    honest response is to skip the workload rather than run guaranteed failures.
    """
    decision = evaluate_gate(1.05, ceiling=amdahl_ceiling(0.05))  # ceiling 1.053
    assert decision.gate == pytest.approx(GATE_FLOOR)
    assert decision.reachable is False


def test_a_normal_workload_is_reachable() -> None:
    decision = evaluate_gate(1.30, ceiling=1.5372)
    assert decision.reachable is True
    assert decision.passed is True


# -- decisions carry their derivation ------------------------------------


def test_the_verdict_records_ceiling_rule_and_margin() -> None:
    decision = evaluate_gate(
        1.22, ceiling=1.3730, ceiling_source="profiler:universal-wan self-time"
    )
    rendered = decision.as_dict()
    assert rendered["measured_ceiling"] == pytest.approx(1.3730)
    assert rendered["rule"] == CEILING_GATE_RULE
    assert rendered["ceiling_source"].startswith("profiler:")
    # 1.22 against a 1.1865 gate.
    assert rendered["margin"] == pytest.approx(1.22 - 1.1865, abs=1e-3)
    assert rendered["passed"] is True


def test_a_candidate_below_its_gate_fails_with_a_negative_margin() -> None:
    decision = evaluate_gate(1.12, ceiling=1.5372)
    assert decision.passed is False
    assert decision.margin < 0


def test_our_rejected_candidates_still_fail_under_the_derived_gate() -> None:
    """SAGE_ATTN 0.8031x and VSA 0.5169x are slower than baseline.

    A softer gate must not rehabilitate them: both are below 1.0, so no
    ceiling-derived gate can admit them.
    """
    for measured in (0.8031, 0.5169):
        decision = evaluate_gate(measured, ceiling=1.5372)
        assert decision.passed is False


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_a_non_positive_measurement_is_refused(bad) -> None:
    with pytest.raises(CeilingError):
        evaluate_gate(bad, ceiling=1.4)


# -- boundary ------------------------------------------------------------


def test_exactly_meeting_the_gate_passes() -> None:
    gate = derive_gate(1.5372)
    assert evaluate_gate(gate, ceiling=1.5372).passed is True


def test_just_below_the_gate_fails() -> None:
    gate = derive_gate(1.5372)
    assert evaluate_gate(gate - 1e-6, ceiling=1.5372).passed is False
