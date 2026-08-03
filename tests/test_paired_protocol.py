"""CPU tests for the paired measurement protocol (schema v3).

The central test is
:func:`test_paired_recovers_an_effect_the_sequential_protocol_misses`. It
injects a known speedup and a known clock drift into synthetic run times and
shows the sequential protocol reports the wrong answer while the paired one
recovers the injected effect. Everything else guards a way the protocol could
be claimed without being applied.

No torch, no GPU: the protocol is arithmetic over run times.
"""

from __future__ import annotations

import pytest

from autokernel.dispatch.paired import (
    PROTOCOL_PAIRED,
    ProtocolError,
    bootstrap_median_ci,
    interleaved_schedule,
    paired_speedups,
    summarize_paired,
    wilcoxon_signed_rank_p,
)
from autokernel.dispatch.warmup import clock_plateaued, sustained_warmup


# -- the drift simulation ------------------------------------------------


def _drifting_runs(pairs: int, *, true_speedup: float, drift_per_run: float):
    """Synthesize an A/B where the device speeds up over the session.

    Base cost falls linearly with run index -- the boost ramp -- and the
    candidate is genuinely ``true_speedup`` faster. Returns both the
    interleaved runs and the sequential ones, drawn from the *same* drift
    trajectory, so the only difference between the protocols is when each arm
    was sampled.
    """
    total_runs = pairs * 2

    def cost(index: int) -> float:
        # Runs get cheaper as clocks climb; index 0 is the cold, slow end.
        return 10.0 * (1.0 - drift_per_run * index)

    # Interleaved ABBA: pairs alternate NC, CN, so the within-pair drift
    # offset cancels across pairs instead of always favouring the candidate.
    schedule = interleaved_schedule(pairs)
    inter_native, inter_candidate = [], []
    for index in range(total_runs):
        base = cost(index)
        if schedule[index] == "native":
            inter_native.append(base)
        else:
            inter_candidate.append(base / true_speedup)

    # Sequential: all native first, then all candidate -- the candidate arm
    # runs entirely on the warmer, faster half of the trajectory.
    seq_native = [cost(i) for i in range(pairs)]
    seq_candidate = [cost(pairs + i) / true_speedup for i in range(pairs)]
    return (inter_native, inter_candidate), (seq_native, seq_candidate)


def test_paired_recovers_an_effect_the_sequential_protocol_misses() -> None:
    """Drift inflates the sequential answer; pairing cancels it.

    The candidate is truly 1.05x. Under sequential scheduling the candidate arm
    runs later, on a faster device, so the measured ratio absorbs the drift and
    overstates the effect. Under interleaving each pair is adjacent in time and
    the ratio recovers the injected value.
    """
    true_speedup = 1.05
    (in_n, in_c), (sq_n, sq_c) = _drifting_runs(
        16, true_speedup=true_speedup, drift_per_run=0.01
    )

    paired = summarize_paired(
        in_n, in_c,
        schedule=interleaved_schedule(16),
        clock_trace=[{"sm_clock_mhz": 2000}],
    )
    sequential_ratio = (
        sorted(sq_n)[len(sq_n) // 2] / sorted(sq_c)[len(sq_c) // 2]
    )

    # Paired lands on the injected effect.
    assert paired.speedup_paired_median == pytest.approx(true_speedup, abs=0.01)
    # Sequential does not -- it is inflated by the drift.
    assert sequential_ratio > true_speedup + 0.05
    # And the error is large enough to change a gate decision at 1.10.
    assert sequential_ratio > 1.10 > paired.speedup_paired_median


def test_paired_is_not_fooled_when_there_is_no_real_effect() -> None:
    """Drift alone must not manufacture a speedup."""
    (in_n, in_c), (sq_n, sq_c) = _drifting_runs(
        16, true_speedup=1.0, drift_per_run=0.01
    )
    paired = summarize_paired(
        in_n, in_c,
        schedule=interleaved_schedule(16),
        clock_trace=[{"sm_clock_mhz": 2000}],
    )
    sequential_ratio = (
        sorted(sq_n)[len(sq_n) // 2] / sorted(sq_c)[len(sq_c) // 2]
    )
    assert paired.speedup_paired_median == pytest.approx(1.0, abs=0.02)
    # The sequential protocol reports a speedup that does not exist.
    assert sequential_ratio > 1.05


# -- gating consumes the paired delta ------------------------------------


def test_the_record_names_the_paired_delta_as_the_gate_input() -> None:
    result = summarize_paired(
        [10.0] * 8, [9.0] * 8,
        schedule=interleaved_schedule(8),
        clock_trace=[{"sm_clock_mhz": 2000}],
    )
    rendered = result.as_dict()
    assert rendered["gate_input"] == "speedup_paired_median"
    # Raw medians are recorded but explicitly not the gate input.
    assert "native_median" in rendered and "speedup_of_medians" in rendered


def test_a_ci_spanning_one_is_inconclusive() -> None:
    """A point estimate over the gate with a CI across 1.0 is not a pass."""
    native = [10.0, 9.0, 11.0, 10.5, 9.5, 10.2, 9.8, 10.1]
    candidate = [9.9, 9.3, 10.8, 10.9, 9.2, 10.4, 9.6, 10.3]
    result = summarize_paired(
        native, candidate,
        schedule=interleaved_schedule(8),
        clock_trace=[{"sm_clock_mhz": 2000}],
    )
    assert result.ci_low <= 1.0 <= result.ci_high
    assert result.conclusive is False


def test_a_clear_effect_is_conclusive() -> None:
    result = summarize_paired(
        [10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9],
        [5.0, 5.05, 4.95, 5.1, 4.9, 5.0, 5.05, 4.95],
        schedule=interleaved_schedule(8),
        clock_trace=[{"sm_clock_mhz": 2000}],
    )
    assert result.conclusive is True
    assert result.ci_low > 1.0


# -- invalidation rules (additive to v2) ---------------------------------


def test_non_interleaved_assignment_is_invalid() -> None:
    schedule = ("native",) * 4 + ("candidate",) * 4
    result = summarize_paired(
        [10.0] * 4, [9.0] * 4, schedule=schedule,
        clock_trace=[{"sm_clock_mhz": 2000}],
    )
    assert result.valid_for_gating is False
    assert any("interleaved" in r for r in result.invalid_reasons)


def test_a_missing_schedule_is_invalid() -> None:
    result = summarize_paired(
        [10.0] * 4, [9.0] * 4, schedule=None,
        clock_trace=[{"sm_clock_mhz": 2000}],
    )
    assert result.valid_for_gating is False
    assert any("schedule" in r for r in result.invalid_reasons)


def test_a_missing_clock_trace_is_invalid() -> None:
    """No trace means no evidence the warmup ever reached a plateau."""
    result = summarize_paired(
        [10.0] * 4, [9.0] * 4, schedule=interleaved_schedule(4), clock_trace=None
    )
    assert result.valid_for_gating is False
    assert any("clock trace" in r for r in result.invalid_reasons)


def test_a_well_formed_measurement_is_valid() -> None:
    result = summarize_paired(
        [10.0] * 6, [9.0] * 6,
        schedule=interleaved_schedule(6),
        clock_trace=[{"sm_clock_mhz": 2000}],
    )
    assert result.valid_for_gating is True
    assert result.invalid_reasons == ()
    assert result.protocol == PROTOCOL_PAIRED


def test_a_schedule_starting_on_the_candidate_still_counts_as_paired() -> None:
    """Alternation is the property, not which arm goes first."""
    schedule = ("candidate", "native") * 4
    result = summarize_paired(
        [10.0] * 4, [9.0] * 4, schedule=schedule,
        clock_trace=[{"sm_clock_mhz": 2000}],
    )
    assert result.valid_for_gating is True


# -- statistics ----------------------------------------------------------


def test_unequal_arms_are_refused() -> None:
    with pytest.raises(ProtocolError, match="equal arms"):
        paired_speedups([1.0, 2.0], [1.0])


def test_non_positive_run_times_are_refused() -> None:
    with pytest.raises(ProtocolError, match="non-positive"):
        paired_speedups([1.0, 0.0], [1.0, 1.0])


def test_bootstrap_is_deterministic() -> None:
    values = [1.1, 1.2, 1.05, 1.3, 1.15, 1.25, 1.0, 1.4]
    assert bootstrap_median_ci(values) == bootstrap_median_ci(values)


def test_wilcoxon_detects_a_consistent_shift() -> None:
    p = wilcoxon_signed_rank_p([0.5] * 10)
    assert p is not None and p < 0.01


def test_wilcoxon_returns_none_when_the_sample_is_too_small() -> None:
    """Better than a number that would look like a result."""
    assert wilcoxon_signed_rank_p([0.1, 0.2, 0.3]) is None


def test_wilcoxon_on_symmetric_noise_is_not_significant() -> None:
    p = wilcoxon_signed_rank_p([0.1, -0.1, 0.2, -0.2, 0.15, -0.15, 0.05, -0.05])
    assert p is not None and p > 0.05


def test_interleaved_schedule_is_abba_not_abab() -> None:
    """ABAB would put the candidate later in every pair and bias the estimate.

    On a ramping clock that offset is about one run's worth of drift in a
    consistent direction -- comparable to the effects this gate decides on.
    """
    schedule = interleaved_schedule(4)
    assert schedule == (
        "native", "candidate", "candidate", "native",
        "native", "candidate", "candidate", "native",
    )
    # ABBA puts two of the same arm adjacent at each quartet boundary, so the
    # interleaving check must accept that rather than demand strict alternation.
    from autokernel.dispatch.paired import _is_interleaved

    assert _is_interleaved(schedule) is True


def test_an_odd_pair_count_is_refused() -> None:
    """The NC and CN halves cannot balance, so the offset would survive."""
    with pytest.raises(ProtocolError, match="even number of pairs"):
        interleaved_schedule(15)


def test_three_in_a_row_is_a_block_not_an_interleave() -> None:
    from autokernel.dispatch.paired import _is_interleaved

    assert _is_interleaved(("native", "native", "native", "candidate",
                            "candidate", "candidate")) is False


# -- warmup --------------------------------------------------------------


def test_plateau_needs_a_full_window() -> None:
    assert clock_plateaued([2000, 2000], window=5) is False
    assert clock_plateaued([2000] * 5, window=5) is True


def test_plateau_is_relative_not_absolute() -> None:
    # 1% spread passes at a 2% tolerance on either clock scale.
    assert clock_plateaued([1980, 2000, 1990, 1995, 2000], tolerance=0.02) is True
    assert clock_plateaued([1400, 1410, 1405, 1408, 1402], tolerance=0.02) is True


def test_warmup_stops_once_the_clock_settles() -> None:
    trace = [300, 900, 1500, 1900, 2000, 2000, 2000, 2000, 2000]
    pulled = iter(trace)
    result = sustained_warmup(
        load=lambda: None,
        sample_clock=lambda: next(pulled, 2000),
        sleep=lambda _: None,
        monotonic=iter(range(0, 100, 2)).__next__,
    )
    assert result.plateaued is True
    # The trace is retained so a reader can see the plateau, not take it on trust.
    assert len(result.samples) >= 5
    assert result.as_dict()["samples"][0]["sm_clock_mhz"] == 300


def test_warmup_times_out_without_raising_and_keeps_the_trace() -> None:
    """An unsettled clock is worse evidence, not no evidence."""
    climbing = iter(range(300, 5000, 50))
    result = sustained_warmup(
        load=lambda: None,
        sample_clock=lambda: next(climbing),
        max_seconds=20.0,
        sleep=lambda _: None,
        monotonic=iter(range(0, 200, 2)).__next__,
    )
    assert result.plateaued is False
    assert result.samples
    assert "still moving" in result.reason


def test_an_unsamplable_clock_is_not_recorded_as_settled() -> None:
    """'We could not observe it' must not be storable as 'it was fine'."""
    result = sustained_warmup(
        load=lambda: None,
        sample_clock=lambda: None,
        max_seconds=10.0,
        sleep=lambda _: None,
        monotonic=iter(range(0, 100, 2)).__next__,
    )
    assert result.plateaued is False
    assert "could not be sampled" in result.reason


def test_undifferentiated_arms_are_invalid() -> None:
    """A candidate arm whose configuration never took effect measures nothing.

    This is the failure that nearly published a retraction of the V1 LTX
    headline: SLURM 1078 set FASTVIDEO_OPTIMIZATION_ARTIFACT_DIR on the
    candidate arm, but that FastVideo checkout reads only
    FASTVIDEO_OPTIMIZATION_CAPTURE, so both arms ran native. The result was
    tight, conclusive, and about nothing.
    """
    result = summarize_paired(
        [10.0] * 6, [11.0] * 6,
        schedule=interleaved_schedule(6),
        clock_trace=[{"sm_clock_mhz": 2000}],
        arms_differentiated=False,
    )
    assert result.valid_for_gating is False
    assert any("not observably different" in r for r in result.invalid_reasons)


def test_differentiated_arms_stay_valid() -> None:
    result = summarize_paired(
        [10.0] * 6, [9.0] * 6,
        schedule=interleaved_schedule(6),
        clock_trace=[{"sm_clock_mhz": 2000}],
        arms_differentiated=True,
    )
    assert result.valid_for_gating is True
