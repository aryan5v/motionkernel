"""Perceptual metrics over fixed-seed frame sets.

:mod:`autokernel.verification.fidelity` owns the contract and the gate; this
module owns the measurement. The split is deliberate -- the gate must be
testable without a GPU, an image backend, or a pretrained network, so nothing
here is imported at contract-evaluation time.

What a tier-2 measurement has to be careful about
-------------------------------------------------

A perceptual metric is a *summary*, and every summary can be gamed by the thing
it summarizes. The rules below exist because the alternative is a tier-2 gate
that reports 0.99 while the video is visibly wrong:

* **Frames are compared pairwise and aggregated by worst case, not mean.** A
  cache that is perfect on 7 frames and destroys the 8th averages to a
  comfortable pass. Video artifacts are temporally local -- a single broken
  frame is the exact failure mode a schedule transform produces -- so the
  aggregate a budget is checked against is the worst frame, and the per-frame
  values are reported alongside it.
* **A metric that cannot be computed is absent, never substituted.** If LPIPS
  has no backend installed, the evidence carries ``lpips=None`` and a budget
  that gates on LPIPS *holds* the artifact. Reporting a default would rebuild
  the R4 hole with a different number in it.
* **Frame count and shape mismatches are errors.** Comparing 8 reference frames
  against 7 candidate frames is not a low score, it is a broken run.

SSIM is computed here directly (numpy only) so the common case has no optional
dependency. LPIPS and VBench need pretrained networks; they are looked up
through pluggable backends and are simply unavailable when not installed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .fidelity import PerceptualEvidence

__all__ = [
    "FrameSet",
    "MetricUnavailable",
    "PerceptualError",
    "compare_frame_sets",
    "lpips_backend",
    "ssim",
    "vbench_backend",
]

#: SSIM's stabilizing constants, from Wang et al. 2004, for data in [0, 1].
_K1 = 0.01
_K2 = 0.03
#: Gaussian window, 11x11 with sigma 1.5 -- the reference implementation's.
_WINDOW_SIZE = 11
_WINDOW_SIGMA = 1.5


class PerceptualError(ValueError):
    """A frame comparison cannot be performed as specified."""


class MetricUnavailable(RuntimeError):
    """A metric was requested but no backend is installed to compute it.

    Raised by a backend, caught by :func:`compare_frame_sets`, and turned into
    an *absent* metric rather than a bad score -- the fidelity gate then holds
    any artifact whose budget depends on it.
    """


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - environment dependent
        raise PerceptualError(
            "perceptual comparison requires numpy"
        ) from error
    return np


@dataclass(frozen=True)
class FrameSet:
    """Frames from one generation, plus the identity that makes them fixed.

    ``name`` and ``seed`` are carried into the evidence and checked against the
    budget, so a promotion cannot be justified with frames from a different
    prompt, resolution, or seed than the one the workload contracted.
    """

    name: str
    seed: int
    frames: Any  # (N, H, W, C) or (N, H, W) array-like, any numeric dtype

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise PerceptualError("frame set name must be a non-empty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise PerceptualError("frame set seed must be an integer")

    @property
    def count(self) -> int:
        return int(len(self.frames))


def _as_float_array(frames: Any, *, label: str) -> Any:
    """Normalize frames to float64 in [0, 1] with shape (N, H, W, C)."""
    np = _numpy()
    array = np.asarray(frames)
    if array.ndim == 3:
        array = array[..., None]
    if array.ndim != 4:
        raise PerceptualError(
            f"{label}: expected frames shaped (N, H, W) or (N, H, W, C), "
            f"got {array.shape}"
        )
    if array.size == 0:
        raise PerceptualError(f"{label}: frame set is empty")

    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        scale = float(info.max)
        array = array.astype(np.float64) / scale
    else:
        array = array.astype(np.float64)

    if not np.all(np.isfinite(array)):
        # A NaN frame would make every metric NaN and every comparison
        # meaningless. R4's comparator reported max_abs_error=nan while
        # allclose returned True; that class of bug does not get a second run.
        raise PerceptualError(f"{label}: frames contain NaN or infinite values")
    return array


def _gaussian_window(np: Any) -> Any:
    coords = np.arange(_WINDOW_SIZE, dtype=np.float64) - (_WINDOW_SIZE - 1) / 2.0
    kernel = np.exp(-(coords**2) / (2.0 * _WINDOW_SIGMA**2))
    kernel /= kernel.sum()
    return np.outer(kernel, kernel)


def _filter2d(np: Any, plane: Any, window: Any) -> Any:
    """Valid-mode 2-D correlation, written with strides to avoid scipy."""
    height, width = plane.shape
    k = window.shape[0]
    if height < k or width < k:
        raise PerceptualError(
            f"frames are {height}x{width}, smaller than the {k}x{k} SSIM window"
        )
    windows = np.lib.stride_tricks.sliding_window_view(plane, (k, k))
    return np.einsum("ijkl,kl->ij", windows, window)


def ssim(reference: Any, candidate: Any) -> float:
    """Mean SSIM over one pair of frames, averaged across channels.

    Implemented directly rather than pulled from scikit-image so tier 2 has no
    optional dependency in its common path. Follows Wang et al. 2004: an 11x11
    Gaussian window with sigma 1.5, valid-mode, on data scaled to [0, 1].
    """
    np = _numpy()
    ref = _as_float_array(reference[None, ...], label="reference")[0]
    cand = _as_float_array(candidate[None, ...], label="candidate")[0]
    if ref.shape != cand.shape:
        raise PerceptualError(
            f"frame shape mismatch: reference {ref.shape} vs candidate {cand.shape}"
        )

    window = _gaussian_window(np)
    c1 = (_K1 * 1.0) ** 2
    c2 = (_K2 * 1.0) ** 2

    scores = []
    for channel in range(ref.shape[-1]):
        x = ref[..., channel]
        y = cand[..., channel]
        mu_x = _filter2d(np, x, window)
        mu_y = _filter2d(np, y, window)
        mu_x_sq, mu_y_sq, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sigma_x = _filter2d(np, x * x, window) - mu_x_sq
        sigma_y = _filter2d(np, y * y, window) - mu_y_sq
        sigma_xy = _filter2d(np, x * y, window) - mu_xy
        numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
        scores.append(float(np.mean(numerator / denominator)))
    return float(sum(scores) / len(scores))


def lpips_backend() -> Callable[[Any, Any], float]:
    """Return an LPIPS callable, or raise :class:`MetricUnavailable`.

    LPIPS needs a pretrained network, so unlike SSIM it cannot be a hard
    dependency of the harness. Absence is reported honestly rather than
    papered over with a substitute metric.
    """
    try:
        import lpips as lpips_module
        import torch
    except ImportError as error:
        raise MetricUnavailable(
            "LPIPS requires the 'lpips' package and torch; install them to gate "
            "a tier-2 budget on max_lpips"
        ) from error

    network = lpips_module.LPIPS(net="alex")
    network.eval()

    def _score(reference: Any, candidate: Any) -> float:
        np = _numpy()
        ref = _as_float_array(np.asarray(reference)[None, ...], label="reference")
        cand = _as_float_array(np.asarray(candidate)[None, ...], label="candidate")
        # LPIPS wants NCHW in [-1, 1].
        to_nchw = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).float() * 2 - 1
        with torch.no_grad():
            return float(network(to_nchw(ref), to_nchw(cand)).item())

    return _score


def vbench_backend() -> Callable[[Sequence[Any]], float]:
    """Return a VBench-subset scorer, or raise :class:`MetricUnavailable`.

    VBench is heavyweight and pulls its own model zoo; the plan calls for it to
    run in an isolated stage. This hook is where that stage's result is read
    back in, so nothing in the promotion path imports it directly.
    """
    raise MetricUnavailable(
        "VBench scoring runs in an isolated stage; wire its result in through "
        "compare_frame_sets(vbench_score=...) rather than importing it here"
    )


def compare_frame_sets(
    reference: FrameSet,
    candidate: FrameSet,
    *,
    want_lpips: bool = False,
    lpips_fn: Callable[[Any, Any], float] | None = None,
    vbench_score: float | None = None,
) -> PerceptualEvidence:
    """Measure a candidate frame set against its reference.

    Args:
        reference: frames from the native generation.
        candidate: frames from the optimized generation, at the same seed.
        want_lpips: compute LPIPS. When no backend is available the metric is
            reported absent and any budget gating on it will hold the artifact.
        lpips_fn: an explicit scorer, mainly for tests.
        vbench_score: a score computed by the isolated VBench stage, passed in
            rather than computed here.

    Returns:
        :class:`PerceptualEvidence` carrying the **worst-frame** SSIM and LPIPS.
        Worst-case, not mean: a transform that ruins one frame in eight must not
        average its way to a pass.

    Raises:
        PerceptualError: the two sets are not comparable (different frame
            counts, shapes, or non-finite values). That is a broken run, not a
            low score, and it must not be recorded as one.
    """
    if reference.seed != candidate.seed:
        raise PerceptualError(
            f"frame sets were generated at different seeds "
            f"({reference.seed} vs {candidate.seed}); a perceptual comparison "
            f"is only meaningful at a fixed seed"
        )
    ref_frames = _as_float_array(reference.frames, label="reference")
    cand_frames = _as_float_array(candidate.frames, label="candidate")
    if ref_frames.shape != cand_frames.shape:
        raise PerceptualError(
            f"frame set shape mismatch: reference {ref_frames.shape} vs "
            f"candidate {cand_frames.shape}"
        )

    count = int(ref_frames.shape[0])
    worst_ssim = math.inf
    for index in range(count):
        worst_ssim = min(worst_ssim, ssim(ref_frames[index], cand_frames[index]))

    worst_lpips: float | None = None
    if want_lpips or lpips_fn is not None:
        scorer = lpips_fn
        if scorer is None:
            try:
                scorer = lpips_backend()
            except MetricUnavailable:
                scorer = None
        if scorer is not None:
            worst_lpips = max(
                scorer(ref_frames[index], cand_frames[index])
                for index in range(count)
            )

    return PerceptualEvidence(
        frame_set=candidate.name,
        seed=candidate.seed,
        frames_compared=count,
        ssim=worst_ssim,
        lpips=worst_lpips,
        vbench=vbench_score,
    )
