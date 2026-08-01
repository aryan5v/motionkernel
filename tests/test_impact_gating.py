"""Measured-impact gating for candidate packaging.

Run ltx-v1-overnight-20260801-r4-sol packaged four VAE artifacts whose combined
measured savings were 0.63% of end-to-end against a 1.01x campaign target. No
combination of them could pass, yet the run spent a full A/B generation pair
finding out. These tests pin the arithmetic that makes that decision earlier.
"""

from __future__ import annotations

import pytest

from autokernel.discovery.ranking import (
    e2e_share,
    measured_e2e_improvement,
    meets_end_to_end_target,
    optimistic_e2e_improvement,
    projected_end_to_end_speedup,
)

#: The four artifacts r4 packaged: (share_of_e2e, isolated speedup).
R4_PACKAGED = (
    (0.023336937532381377, 1.1153512914215753),  # mk-bbfe15180d31bf50
    (0.018633836860210973, 1.1104277134087817),  # mk-a81e140d62ff170c
    (0.010810459900891809, 1.1211912267249589),  # mk-baecc3825d4a8c18
    (0.009224912365976008, 1.1063535911602211),  # mk-b6cb64f99049683b
)


def test_upper_bound_and_measured_impact_differ_by_an_order_of_magnitude() -> None:
    share = 0.023336937532381377
    optimistic = optimistic_e2e_improvement(share, reducible_fraction=0.9)
    measured = measured_e2e_improvement(share, 1.1153512914215753)
    assert optimistic == pytest.approx(0.021, abs=1e-3)
    assert measured == pytest.approx(0.0024, abs=1e-4)
    assert optimistic > measured * 8


def test_r4_artifact_set_could_never_have_reached_the_campaign_target() -> None:
    """The decisive number: 1.006x achievable against a 1.01x gate."""
    improvements = [
        measured_e2e_improvement(share, speedup) for share, speedup in R4_PACKAGED
    ]
    assert sum(improvements) == pytest.approx(0.00632, abs=1e-4)
    projected = projected_end_to_end_speedup(improvements)
    assert projected == pytest.approx(1.00636, abs=1e-4)
    assert not meets_end_to_end_target(improvements, min_end_to_end_speedup=1.01)


def test_upper_bound_would_have_cleared_the_same_gate() -> None:
    """Why r4 packaged them: the bound it used says 5.6%, comfortably over 1%."""
    bounds = [
        optimistic_e2e_improvement(share, reducible_fraction=0.9)
        for share, _ in R4_PACKAGED
    ]
    assert meets_end_to_end_target(bounds, min_end_to_end_speedup=1.01)


def test_dispatch_overhead_is_charged_against_savings() -> None:
    improvements = [measured_e2e_improvement(share, speedup) for share, speedup in R4_PACKAGED]
    assert meets_end_to_end_target(improvements, min_end_to_end_speedup=1.005)
    assert not meets_end_to_end_target(
        improvements,
        min_end_to_end_speedup=1.005,
        dispatch_overhead_fraction=0.004,
    )


def test_r4_observed_overhead_turns_the_set_into_a_regression() -> None:
    """r4 measured 3.2818s native vs 3.9410s optimized: 0.8327x."""
    improvements = [measured_e2e_improvement(share, speedup) for share, speedup in R4_PACKAGED]
    observed_overhead = (3.94101024675183 - 3.2818314481992275) / 3.2818314481992275
    projected = projected_end_to_end_speedup(
        improvements, dispatch_overhead_fraction=observed_overhead
    )
    assert projected < 1.0, "overhead dominates the savings"


def test_a_speedup_at_or_below_one_returns_nothing() -> None:
    assert measured_e2e_improvement(0.5, 1.0) == 0.0
    assert measured_e2e_improvement(0.5, 0.9) == 0.0
    with pytest.raises(ValueError):
        measured_e2e_improvement(0.5, 0.0)


def test_a_worthwhile_candidate_still_passes() -> None:
    """The transformer region: 2.557x on a large share clears 1.01x easily."""
    improvement = measured_e2e_improvement(0.35, 2.557)
    assert improvement == pytest.approx(0.2131, abs=1e-3)
    assert meets_end_to_end_target(
        [improvement], min_end_to_end_speedup=1.01, dispatch_overhead_fraction=0.01
    )


# -- metric integrity ---------------------------------------------------


def test_share_of_e2e_is_clamped_to_one() -> None:
    """r4 reported 1.0333 for transformer.model.transformer_blocks."""
    assert e2e_share(2864051.9510001247, 2771790.125000082) == 1.0
    assert e2e_share(1385895.0, 2771790.125000082) == pytest.approx(0.5, abs=1e-6)
    assert e2e_share(100.0, 0.0) == 0.0


def test_region_calls_use_the_named_range_not_the_event_sum() -> None:
    """r4 reported calls=195661 for a region invoked 1151 times."""
    from autokernel.discovery.correlation import _aggregate_region_timing
    from autokernel.discovery.types import GraphRegion, OperatorHotspot

    region = GraphRegion.build(
        name="transformer.model.transformer_blocks",
        operations=("aten::mul",),
        inputs=(),
        outputs=(),
        dependencies=(),
        calls=0,
        cuda_time_us=0.0,
        self_cuda_time_us=0.0,
    )
    rows = [
        OperatorHotspot(
            name="transformer.model.transformer_blocks",
            op_key="region::transformer_blocks",
            calls=1151,
            cuda_time_us=2_000_000.0,
            self_cuda_time_us=0.0,
        ),
        OperatorHotspot(
            name="aten::mul", op_key="aten::mul", calls=100_000,
            cuda_time_us=5_000.0, self_cuda_time_us=5_000.0
        ),
        OperatorHotspot(
            name="aten::add", op_key="aten::add", calls=94_510,
            cuda_time_us=4_000.0, self_cuda_time_us=4_000.0
        ),
    ]
    _, _, calls = _aggregate_region_timing(region, rows)
    assert calls == 1151, "the record_function range counts invocations"
    assert calls != 195_661, "not the sum of aten event counts"


def test_calls_without_a_named_range_do_not_sum_event_counts() -> None:
    from autokernel.discovery.correlation import _aggregate_region_timing
    from autokernel.discovery.types import GraphRegion, OperatorHotspot

    region = GraphRegion.build(
        name="some.region",
        operations=("aten::mul",),
        inputs=(),
        outputs=(),
        dependencies=(),
        calls=0,
        cuda_time_us=0.0,
        self_cuda_time_us=0.0,
    )
    rows = [
        OperatorHotspot(
            name="aten::mul", op_key="aten::mul", calls=40,
            cuda_time_us=10.0, self_cuda_time_us=10.0,
        ),
        OperatorHotspot(
            name="aten::add", op_key="aten::add", calls=25,
            cuda_time_us=8.0, self_cuda_time_us=8.0,
        ),
    ]
    _, _, calls = _aggregate_region_timing(region, rows)
    assert calls == 40, "bounded below by the largest row, never 65"
