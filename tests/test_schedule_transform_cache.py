"""CPU tests for input-similarity caching over the denoising loop.

Numpy-free, torch-free: the policy is deliberately separable from the tensors
so the part that is easy to get subtly wrong can be tested exhaustively without
a GPU.

Each test corresponds to a way a cache can be fast and wrong.
"""

from __future__ import annotations

import pytest

from autokernel.transforms import (
    CachePolicy,
    InputSimilarityCache,
    TransformError,
)


def _cache(**overrides) -> InputSimilarityCache:
    values = {"threshold": 1.0, "warmup_steps": 1, "max_consecutive_skips": None}
    values.update(overrides)
    cache = InputSimilarityCache(CachePolicy(**values))
    cache.begin_generation()
    return cache


# -- policy validation --------------------------------------------------


def test_threshold_is_required_and_must_come_from_the_manifest() -> None:
    with pytest.raises(TransformError, match="must declare a threshold"):
        CachePolicy.from_dict({"warmup_steps": 2})


def test_policy_round_trips() -> None:
    policy = CachePolicy(threshold=0.15, warmup_steps=2, max_consecutive_skips=4)
    assert CachePolicy.from_dict(policy.as_dict()) == policy


def test_a_negative_or_nan_threshold_is_refused() -> None:
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(TransformError, match="threshold"):
            CachePolicy(threshold=bad)


def test_warmup_below_one_is_refused() -> None:
    # The first step has nothing to reuse; zero warmup is incoherent.
    with pytest.raises(TransformError, match="nothing to reuse"):
        CachePolicy(threshold=1.0, warmup_steps=0)


def test_unknown_policy_fields_are_refused() -> None:
    with pytest.raises(TransformError, match="unknown cache policy fields"):
        CachePolicy.from_dict({"threshold": 0.1, "thresh0ld": 0.2})


# -- the structural guarantees ------------------------------------------


def test_the_first_step_is_never_a_hit() -> None:
    """Structural, not a threshold accident: distance 0.0 would otherwise skip."""
    cache = _cache(threshold=0.0)
    decision = cache.step(0, 0.0)
    assert decision.compute
    assert "warmup" in decision.reason


def test_driving_without_begin_generation_is_refused() -> None:
    cache = InputSimilarityCache(CachePolicy(threshold=1.0))
    with pytest.raises(TransformError, match="begin_generation"):
        cache.step(0, 0.0)


def test_state_does_not_leak_between_generations() -> None:
    """A carried-over cache contaminates the next prompt's opening frames."""
    cache = _cache(threshold=10.0)
    for step in range(4):
        cache.step(step, 0.1)
    assert cache.stats.steps_skipped == 3

    cache.begin_generation()
    first = cache.step(0, 0.0)
    assert first.compute  # warmup again, not a continuation
    assert first.accumulated == 0.0
    assert cache.stats.steps_total == 1
    assert cache.stats.steps_skipped == 0


def test_steps_must_advance() -> None:
    cache = _cache()
    cache.step(0, 0.0)
    cache.step(1, 0.1)
    with pytest.raises(TransformError, match="must advance"):
        cache.step(1, 0.1)


# -- accumulation, which is the whole point -----------------------------


def test_distance_accumulates_rather_than_being_compared_per_step() -> None:
    """Many individually-small deltas must eventually force a recompute.

    Per-step comparison would let the input drift arbitrarily far while every
    single step looked safe.
    """
    cache = _cache(threshold=1.0)
    cache.step(0, 0.0)  # warmup
    for step in range(1, 10):
        decision = cache.step(step, 0.3)
        if decision.compute:
            assert step == 4  # 0.3 * 4 = 1.2 >= 1.0
            assert "accumulated" in decision.reason
            break
    else:
        pytest.fail("accumulated distance never reached the threshold")


def test_the_accumulator_resets_after_a_compute() -> None:
    cache = _cache(threshold=1.0)
    cache.step(0, 0.0)
    cache.step(1, 0.6)  # skip, accumulated 0.6
    forced = cache.step(2, 0.6)  # accumulated 1.2 -> compute
    assert forced.compute
    following = cache.step(3, 0.6)  # accumulator reset, so 0.6 -> skip
    assert following.skipped
    assert following.accumulated == pytest.approx(0.6)


def test_a_zero_threshold_never_skips() -> None:
    cache = _cache(threshold=0.0)
    for step in range(5):
        assert cache.step(step, 0.0).compute
    assert cache.stats.steps_skipped == 0


# -- the consecutive-skip cap -------------------------------------------


def test_consecutive_skips_are_capped_even_below_threshold() -> None:
    """The input-distance accumulator cannot see output error compounding.

    With a generous threshold an uncapped policy would skip the whole schedule
    and produce something smooth, plausible and wrong.
    """
    cache = _cache(threshold=1000.0, max_consecutive_skips=3, cooldown_steps=1)
    cache.step(0, 0.0)  # warmup
    decisions = [cache.step(step, 0.001) for step in range(1, 9)]
    computed = [d.step for d in decisions if d.compute]
    assert computed, "the cap must force at least one recompute"
    assert cache.stats.max_consecutive_skips_used <= 3
    assert any("cap" in d.reason or "cooldown" in d.reason for d in decisions)


def test_an_uncapped_policy_skips_everything_below_threshold() -> None:
    # The cap is a policy choice, not a hidden floor -- None really means none.
    cache = _cache(threshold=1000.0, max_consecutive_skips=None)
    cache.step(0, 0.0)
    for step in range(1, 20):
        assert cache.step(step, 0.001).skipped
    assert cache.stats.max_consecutive_skips_used == 19


# -- reporting ----------------------------------------------------------


def test_hit_rate_is_reported() -> None:
    """A transform that never fires and one that always fires both "ran"."""
    cache = _cache(threshold=1.0)
    cache.step(0, 0.0)
    for step in range(1, 5):
        cache.step(step, 0.1)
    stats = cache.stats
    assert stats.steps_total == 5
    assert stats.steps_computed == 1
    assert stats.steps_skipped == 4
    assert stats.hit_rate == pytest.approx(0.8)
    assert stats.as_dict()["hit_rate"] == pytest.approx(0.8)


def test_hit_rate_of_an_empty_generation_is_zero_not_a_crash() -> None:
    cache = _cache()
    assert cache.stats.hit_rate == 0.0


# -- the artifact kind --------------------------------------------------


def test_the_kind_registered_without_touching_validation() -> None:
    """Track A's registry claim, checked rather than assumed."""
    from autokernel.artifact.kinds import (
        SCHEDULE_TRANSFORM,
        known_target_kinds,
        target_kind_spec,
    )

    assert SCHEDULE_TRANSFORM in known_target_kinds()
    spec = target_kind_spec(SCHEDULE_TRANSFORM)
    # It wraps the loop, so it must not be dispatched by graph fingerprint.
    assert spec.replaces_region is False


def _operation(**overrides) -> dict:
    values = {
        "name": "teacache",
        "graph_fingerprint": "0" * 32,
        "parent_module": "transformer",
        "operations": ["denoising_loop"],
        "target_kind": "schedule_transform",
        "transform_family": "input_similarity_cache",
        "transform_policy": {"threshold": 0.15},
    }
    values.update(overrides)
    return values


def test_schedule_transform_round_trips_through_the_manifest() -> None:
    from autokernel.artifact.types import OperationIdentity

    parsed = OperationIdentity.from_dict(
        _operation(), source="<test>", location="operation"
    )
    assert parsed.transform_family == "input_similarity_cache"
    assert parsed.transform_policy["threshold"] == 0.15
    assert OperationIdentity.from_dict(
        parsed.as_dict(), source="<test>", location="operation"
    ) == parsed


def test_the_threshold_travels_in_the_manifest_not_the_payload() -> None:
    """The searched parameter must be reconstructible from the artifact alone."""
    from autokernel.artifact.types import OperationIdentity

    parsed = OperationIdentity.from_dict(
        _operation(transform_policy={"threshold": 0.22, "max_consecutive_skips": 2}),
        source="<test>",
        location="operation",
    )
    policy = CachePolicy.from_dict(parsed.transform_policy)
    assert policy.threshold == 0.22
    assert policy.max_consecutive_skips == 2


def test_a_schedule_transform_must_declare_family_and_policy() -> None:
    from autokernel.artifact.types import ArtifactError, OperationIdentity

    incomplete = _operation()
    del incomplete["transform_policy"]
    with pytest.raises(ArtifactError, match="requires"):
        OperationIdentity.from_dict(
            incomplete, source="<test>", location="operation"
        )


def test_a_schedule_transform_may_not_carry_subgraph_fields() -> None:
    from autokernel.artifact.types import ArtifactError, OperationIdentity

    with pytest.raises(ArtifactError, match="belongs to a 'subgraph' target"):
        OperationIdentity.from_dict(
            _operation(capture_mode="export"),
            source="<test>",
            location="operation",
        )


# -- issues found in review ---------------------------------------------


def test_decide_is_pure_given_the_accumulated_value() -> None:
    """Accumulation belongs to step(), not to the decision.

    A decision function that advanced the cache could not be called twice for
    the same step without changing the answer.
    """
    cache = _cache(threshold=1.0)
    cache.step(0, 0.0)
    before = cache.stats.steps_total
    first = cache._decide(5, 0.5)
    second = cache._decide(5, 0.5)
    assert first == second
    assert cache.stats.steps_total == before  # nothing advanced


def test_a_malformed_policy_is_rejected_when_the_manifest_is_parsed() -> None:
    """Not at the first step of a campaign that already booked a GPU."""
    from autokernel.artifact.types import ArtifactError, OperationIdentity

    with pytest.raises(ArtifactError, match="threshold"):
        OperationIdentity.from_dict(
            _operation(transform_policy={"threshold": -1.0}),
            source="<test>",
            location="operation",
        )


def test_an_unknown_transform_family_is_left_alone() -> None:
    """The registry lists families this module can check, not those allowed."""
    from autokernel.artifact.types import OperationIdentity

    parsed = OperationIdentity.from_dict(
        _operation(
            transform_family="some_future_family",
            transform_policy={"anything": 1},
        ),
        source="<test>",
        location="operation",
    )
    assert parsed.transform_family == "some_future_family"
