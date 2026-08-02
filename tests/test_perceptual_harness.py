"""CPU tests for the tier-2 perceptual harness.

Numpy only -- no torch, no pretrained networks. The properties under test are
the ones that decide whether tier 2 is a real gate or a number that always
says yes.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from autokernel.verification.fidelity import (
    PERCEPTUAL,
    FidelityBudget,
    evaluate_fidelity,
)
from autokernel.verification.perceptual import (
    FrameSet,
    MetricUnavailable,
    PerceptualError,
    compare_frame_sets,
    ssim,
    vbench_backend,
)


def _frames(count: int = 4, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((count, 32, 32, 3))


# -- SSIM ---------------------------------------------------------------


def test_ssim_of_identical_frames_is_one() -> None:
    frame = _frames(1)[0]
    assert ssim(frame, frame) == pytest.approx(1.0, abs=1e-9)


def test_ssim_falls_as_noise_grows() -> None:
    frame = _frames(1)[0]
    rng = np.random.default_rng(7)
    slight = np.clip(frame + rng.normal(0, 0.01, frame.shape), 0, 1)
    heavy = np.clip(frame + rng.normal(0, 0.30, frame.shape), 0, 1)
    assert 1.0 > ssim(frame, slight) > ssim(frame, heavy)


def test_ssim_matches_the_reference_implementation() -> None:
    """Cross-check the hand-written SSIM against scikit-image.

    SSIM is implemented here so tier 2 has no optional dependency in its common
    path, which means the implementation has to be checked against something
    rather than trusted. Skipped when scikit-image is not installed; it is not
    a runtime dependency.
    """
    sk = pytest.importorskip("skimage.metrics")
    rng = np.random.default_rng(0)
    for trial in range(4):
        a = rng.random((64, 64, 3))
        b = np.clip(a + rng.normal(0, 0.05 * (trial + 1), a.shape), 0, 1)
        reference = sk.structural_similarity(
            a,
            b,
            channel_axis=2,
            data_range=1.0,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
        assert ssim(a, b) == pytest.approx(reference, abs=1e-12)


def test_ssim_rejects_mismatched_shapes() -> None:
    with pytest.raises(PerceptualError, match="shape mismatch"):
        ssim(_frames(1)[0], np.zeros((16, 16, 3)))


def test_ssim_rejects_frames_smaller_than_its_window() -> None:
    tiny = np.zeros((4, 4, 3))
    with pytest.raises(PerceptualError, match="smaller than"):
        ssim(tiny, tiny)


# -- aggregation --------------------------------------------------------


def test_worst_frame_decides_not_the_mean() -> None:
    """One destroyed frame in eight must not average its way to a pass.

    This is the schedule-transform failure mode: a cache that reuses a stale
    result produces a single badly wrong frame, not uniform mild degradation.
    """
    reference = _frames(8)
    candidate = reference.copy()
    rng = np.random.default_rng(3)
    candidate[5] = rng.random(candidate[5].shape)  # one frame, fully wrong

    evidence = compare_frame_sets(
        FrameSet("fs", 42, reference), FrameSet("fs", 42, candidate)
    )
    assert evidence.frames_compared == 8
    # Seven frames are identical (SSIM 1.0); a mean would sit near 0.9.
    assert evidence.ssim < 0.5

    budget = FidelityBudget(
        tier=PERCEPTUAL, min_ssim=0.98, frame_set="fs", seed=42
    )
    assert not evaluate_fidelity(budget, evidence).passed


def test_identical_frame_sets_score_one() -> None:
    frames = _frames(4)
    evidence = compare_frame_sets(
        FrameSet("fs", 1, frames), FrameSet("fs", 1, frames.copy())
    )
    assert evidence.ssim == pytest.approx(1.0, abs=1e-9)
    assert evidence.lpips is None  # not requested, so absent


# -- refusing to guess --------------------------------------------------


def test_absent_lpips_backend_leaves_the_metric_absent_and_holds_the_gate() -> None:
    """No backend must mean 'unknown', never a substituted default."""
    frames = _frames(3)
    evidence = compare_frame_sets(
        FrameSet("fs", 1, frames),
        FrameSet("fs", 1, frames.copy()),
        lpips_fn=None,
        want_lpips=False,
    )
    assert evidence.lpips is None

    budget = FidelityBudget(
        tier=PERCEPTUAL, min_ssim=0.9, max_lpips=0.02, frame_set="fs", seed=1
    )
    verdict = evaluate_fidelity(budget, evidence)
    assert not verdict.passed
    assert "lpips" in verdict.reason


def test_lpips_uses_worst_frame_when_a_backend_is_supplied() -> None:
    frames = _frames(3)
    scores = iter([0.001, 0.400, 0.002])
    evidence = compare_frame_sets(
        FrameSet("fs", 1, frames),
        FrameSet("fs", 1, frames.copy()),
        lpips_fn=lambda a, b: next(scores),
    )
    assert evidence.lpips == pytest.approx(0.400)


def test_vbench_is_passed_in_from_its_isolated_stage() -> None:
    frames = _frames(2)
    evidence = compare_frame_sets(
        FrameSet("fs", 1, frames), FrameSet("fs", 1, frames.copy()), vbench_score=0.83
    )
    assert evidence.vbench == pytest.approx(0.83)
    # And the in-process hook refuses rather than importing the model zoo.
    with pytest.raises(MetricUnavailable, match="isolated stage"):
        vbench_backend()


# -- broken runs are errors, not low scores -----------------------------


def test_differing_frame_counts_is_an_error() -> None:
    with pytest.raises(PerceptualError, match="shape mismatch"):
        compare_frame_sets(
            FrameSet("fs", 1, _frames(8)), FrameSet("fs", 1, _frames(7))
        )


def test_differing_seeds_is_an_error() -> None:
    with pytest.raises(PerceptualError, match="different seeds"):
        compare_frame_sets(
            FrameSet("fs", 1, _frames(4)), FrameSet("fs", 2, _frames(4))
        )


def test_nan_frames_are_an_error_not_a_nan_score() -> None:
    broken = _frames(2)
    broken[0, 0, 0, 0] = np.nan
    with pytest.raises(PerceptualError, match="NaN"):
        compare_frame_sets(
            FrameSet("fs", 1, _frames(2)), FrameSet("fs", 1, broken)
        )


def test_integer_frames_are_scaled_to_unit_range() -> None:
    rng = np.random.default_rng(11)
    frames = rng.integers(0, 256, (3, 32, 32, 3), dtype=np.uint8)
    evidence = compare_frame_sets(
        FrameSet("fs", 1, frames), FrameSet("fs", 1, frames.copy())
    )
    assert evidence.ssim == pytest.approx(1.0, abs=1e-9)
