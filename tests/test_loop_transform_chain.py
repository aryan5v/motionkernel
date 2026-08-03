"""The schedule-transform chain, end to end against a stub pipeline.

Covers what a GPU campaign would otherwise be the first thing to exercise:
that the hook is consulted, that hits and misses land where the policy says,
that the three structural guarantees are enforced at the point of decision, and
that a bundle carrying the policy round-trips through the artifact schema.

The stub pipeline is a dozen lines and stands in for the denoising loop. That
is the point of the hook's design: a loop that consults a decision function
each step can be modelled exactly, so the interesting logic is testable without
a model.
"""

from __future__ import annotations

import sys
import types

import pytest

from autokernel.transforms import CachePolicy, TransformError
from autokernel.transforms.runtime import ScheduleTransformRuntime, relative_l1


# -- stub the FastVideo hook module so this runs with no FastVideo -------


@pytest.fixture(autouse=True)
def _stub_fastvideo(monkeypatch):
    """Provide fastvideo.optimization.loop_transform.StepDecision.

    The runtime imports it lazily and uses only the two fields, so a stub is a
    faithful stand-in and keeps these tests free of a FastVideo checkout.
    """
    class StepDecision:
        def __init__(self, compute, reason=""):
            self.compute = compute
            self.reason = reason

        @property
        def skipped(self):
            return not self.compute

    mod = types.ModuleType("fastvideo.optimization.loop_transform")
    mod.StepDecision = StepDecision
    pkg = types.ModuleType("fastvideo.optimization")
    root = types.ModuleType("fastvideo")
    monkeypatch.setitem(sys.modules, "fastvideo", root)
    monkeypatch.setitem(sys.modules, "fastvideo.optimization", pkg)
    monkeypatch.setitem(sys.modules, "fastvideo.optimization.loop_transform", mod)
    yield


def _policy(**kw) -> CachePolicy:
    values = {"threshold": 1.0, "warmup_steps": 1, "max_consecutive_skips": None}
    values.update(kw)
    return CachePolicy(**values)


class StubPipeline:
    """A denoising loop reduced to what the hook sees."""

    def __init__(self, transform, signals):
        self.transform = transform
        self.signals = signals
        self.computed: list[int] = []
        self.outputs: list[object] = []

    def run(self):
        self.transform.begin_generation(num_steps=len(self.signals))
        for step, signal in enumerate(self.signals):
            decision = self.transform.before_step(
                step=step, timestep=step, latents=signal
            )
            if decision.compute:
                self.computed.append(step)
                output = f"out-{step}"
                used = self.transform.after_step(
                    step=step, timestep=step, latents=signal, output=output
                )
            else:
                used = self.transform.after_step(
                    step=step, timestep=step, latents=signal, output=None
                )
            self.outputs.append(used)
        return self.transform.end_generation()


# -- hits and misses -----------------------------------------------------


def test_identical_inputs_produce_hits_after_warmup() -> None:
    rt = ScheduleTransformRuntime(_policy(threshold=1.0))
    pipe = StubPipeline(rt, [10.0] * 6)
    stats = pipe.run()
    # Step 0 always computes; the rest reuse because the signal never moves.
    assert pipe.computed == [0]
    assert stats["steps_skipped"] == 5
    assert stats["hit_rate"] == pytest.approx(5 / 6)


def test_a_moving_input_forces_recomputes() -> None:
    rt = ScheduleTransformRuntime(_policy(threshold=0.05))
    # Each step moves 10% -- above the threshold, so nothing may be reused.
    signals = [10.0 * (1.1**i) for i in range(6)]
    pipe = StubPipeline(rt, signals)
    stats = pipe.run()
    assert stats["steps_skipped"] == 0
    assert pipe.computed == list(range(6))


def test_the_skipped_step_reuses_the_previous_output() -> None:
    rt = ScheduleTransformRuntime(_policy(threshold=1.0))
    pipe = StubPipeline(rt, [10.0] * 4)
    pipe.run()
    # Every skipped step hands back step 0's output, not None.
    assert pipe.outputs == ["out-0"] * 4


# -- the three structural guarantees -------------------------------------


def test_guarantee_1_first_step_is_never_a_hit() -> None:
    """Enforced by warmup, not by the accumulator starting at zero."""
    rt = ScheduleTransformRuntime(_policy(threshold=0.0))
    pipe = StubPipeline(rt, [10.0] * 3)
    pipe.run()
    assert 0 in pipe.computed


def test_guarantee_2_no_cross_generation_state() -> None:
    rt = ScheduleTransformRuntime(_policy(threshold=1.0))
    with pytest.raises(TransformError, match="begin_generation"):
        rt.before_step(step=0, timestep=0, latents=10.0)


def test_guarantee_2_a_second_generation_starts_clean() -> None:
    rt = ScheduleTransformRuntime(_policy(threshold=1.0))
    first = StubPipeline(rt, [10.0] * 4)
    first.run()
    second = StubPipeline(rt, [99.0] * 4)
    second.run()
    # Step 0 of the second generation recomputes; it does not reuse the
    # previous prompt's output.
    assert 0 in second.computed
    assert second.outputs[0] == "out-0"


def test_guarantee_3_consecutive_skips_are_capped() -> None:
    rt = ScheduleTransformRuntime(
        _policy(threshold=1000.0, max_consecutive_skips=2, cooldown_steps=1)
    )
    pipe = StubPipeline(rt, [10.0] * 10)
    stats = pipe.run()
    assert stats["max_consecutive_skips_used"] <= 2
    assert len(pipe.computed) > 1  # the cap forced recomputes


# -- refusing to guess ---------------------------------------------------


def test_an_unmeasurable_distance_computes_rather_than_reuses() -> None:
    """Skipping on an unusable signal is the one case the cache would guess."""
    rt = ScheduleTransformRuntime(
        _policy(threshold=1000.0), distance=lambda a, b: float("nan")
    )
    pipe = StubPipeline(rt, [10.0] * 5)
    stats = pipe.run()
    assert stats["steps_skipped"] == 0


def test_relative_l1_is_scale_free() -> None:
    # Same 10% move at two magnitudes gives the same distance, so one
    # threshold means the same thing at any latent scale.
    assert relative_l1(11.0, 10.0) == pytest.approx(0.1)
    assert relative_l1(1100.0, 1000.0) == pytest.approx(0.1)


def test_no_previous_signal_is_infinite_distance() -> None:
    assert relative_l1(10.0, None) == float("inf")


# -- diagnostics and the artifact round-trip -----------------------------


def test_diagnostics_carry_the_policy_and_the_hit_rate() -> None:
    rt = ScheduleTransformRuntime(_policy(threshold=1.0, warmup_steps=2))
    stats = StubPipeline(rt, [10.0] * 8).run()
    assert stats["policy"]["threshold"] == 1.0
    assert stats["policy"]["warmup_steps"] == 2
    assert 0.0 <= stats["hit_rate"] <= 1.0
    assert stats["steps_total"] == 8


def test_the_bundle_round_trips_the_searched_threshold() -> None:
    """The threshold must be reconstructible from the artifact alone."""
    from autokernel.artifact.types import OperationIdentity

    operation = {
        "name": "teacache",
        "graph_fingerprint": "0" * 32,
        "parent_module": "transformer",
        "operations": ["denoising_loop"],
        "target_kind": "schedule_transform",
        "transform_family": "input_similarity_cache",
        "transform_policy": {"threshold": 0.18, "max_consecutive_skips": 3},
    }
    parsed = OperationIdentity.from_dict(
        operation, source="<test>", location="operation"
    )
    policy = CachePolicy.from_dict(parsed.transform_policy)
    assert policy.threshold == pytest.approx(0.18)
    # And a runtime built from it behaves as the manifest said it would.
    rt = ScheduleTransformRuntime(policy)
    stats = StubPipeline(rt, [10.0] * 6).run()
    assert stats["policy"]["threshold"] == pytest.approx(0.18)
