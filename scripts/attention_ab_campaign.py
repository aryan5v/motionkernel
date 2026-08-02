"""Wan attention A/B campaign: measure a backend swap, honestly.

Runs the same Wan generation twice -- once per attention backend -- at a fixed
seed, and reports the end-to-end distribution, the perceptual distance between
the two frame sets, and, critically, **which backend actually executed in each
arm**.

That last point is the reason this script exists rather than a shell loop.
FastVideo's selector substitutes FlashAttention silently when an optional
backend cannot be imported (``fastvideo/platforms/cuda.py``), so a candidate arm
on a mis-provisioned node produces a successful generation at baseline speed
with baseline numerics. Measured naively, that is a 1.00x "result" for a
backend that never ran; measured over a noisy cluster it could as easily look
like a win. Every arm therefore records the resolved backend class and the
campaign refuses to report a comparison when the candidate fell back.

Run under Pyxis on the cluster; see scripts/attention_ab_campaign.sbatch.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def _resolved_backend(backend_name: str) -> dict[str, object]:
    """Ask FastVideo which backend class it resolves ``backend_name`` to.

    Done in a subprocess with the env var set exactly as the arm will set it,
    so the answer reflects the arm's real configuration rather than this
    process's.
    """
    probe = (
        "import json, os;"
        "from fastvideo.platforms.interface import AttentionBackendEnum;"
        "from fastvideo.platforms import current_platform;"
        "import torch;"
        "name=os.environ['FASTVIDEO_ATTENTION_BACKEND'];"
        "sel=AttentionBackendEnum[name];"
        "cls=current_platform.get_attn_backend_cls(sel, 128, torch.bfloat16);"
        "print(json.dumps({'requested': name, 'resolved_class': cls}))"
    )
    env = dict(os.environ, FASTVIDEO_ATTENTION_BACKEND=backend_name)
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env
    )
    for line in reversed(completed.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return {
        "requested": backend_name,
        "resolved_class": None,
        "error": completed.stderr[-2000:],
    }


def _effective_name(resolved_class: str | None) -> str | None:
    """Map a resolved backend class path back to its enum name."""
    if not resolved_class:
        return None
    from autokernel.attention import KNOWN_BACKENDS

    for name, identity in KNOWN_BACKENDS.items():
        if identity.class_path == resolved_class:
            return name
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--runs", type=int, default=None)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from autokernel.attention import (
        AttentionFallbackError,
        verify_effective_backend,
    )
    from autokernel.verification.fidelity import FidelityBudget
    from autokernel.workload import load_workload

    workload = load_workload(args.workload)
    budget = FidelityBudget.from_workload(workload)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    arms = {
        mode: workload.mode_env.for_mode(mode)["FASTVIDEO_ATTENTION_BACKEND"]
        for mode in ("native", "optimized")
    }
    print(f"workload : {workload.workload_id}")
    print(f"budget   : tier {budget.number} ({budget.tier})")
    print(f"arms     : {arms}")

    # -- 1. which backend will actually run in each arm --------------
    resolution = {mode: _resolved_backend(name) for mode, name in arms.items()}
    for mode, info in resolution.items():
        effective = _effective_name(info.get("resolved_class"))
        info["effective"] = effective
        print(
            f"  {mode:10s} requested={info['requested']:16s} "
            f"effective={effective}"
        )

    candidate = resolution["optimized"]
    try:
        verify_effective_backend(
            candidate["requested"], candidate.get("effective")
        )
    except AttentionFallbackError as error:
        # Refuse before spending GPU time. The comparison would be baseline
        # against baseline, and every number it produced would be noise
        # wearing a backend's name.
        receipt = {
            "workload_id": workload.workload_id,
            "status": "refused",
            "reason": str(error),
            "resolution": resolution,
        }
        (output / "campaign.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        print(f"\nREFUSED: {error}")
        return 2

    # -- 2. timed A/B ------------------------------------------------
    runs = args.runs or workload.measurement.runs
    warmups = workload.measurement.warmups
    print(f"\ntimed runs per arm: {runs} (warmups {warmups})")

    from fastvideo import VideoGenerator  # noqa: F401  (import cost, once)

    results: dict[str, dict[str, object]] = {}
    for mode, backend in arms.items():
        os.environ["FASTVIDEO_ATTENTION_BACKEND"] = backend
        timings, frames = _run_arm(workload, mode, backend, runs, warmups, output)
        results[mode] = {
            "backend": backend,
            "effective": resolution[mode]["effective"],
            "timings": timings,
            "median": statistics.median(timings),
            "stdev": statistics.pstdev(timings) if len(timings) > 1 else 0.0,
            "min": min(timings),
            "frames": str(frames) if frames else None,
        }
        print(
            f"  {mode:10s} median={results[mode]['median']:.4f}s "
            f"stdev={results[mode]['stdev']:.4f} min={results[mode]['min']:.4f}"
        )

    native, cand = results["native"], results["optimized"]
    speedup_median = native["median"] / cand["median"]
    speedup_min = native["min"] / cand["min"]

    receipt = {
        "workload_id": workload.workload_id,
        "status": "measured",
        "budget": budget.as_dict(),
        "resolution": resolution,
        "runs_per_arm": runs,
        "arms": results,
        "speedup_median": speedup_median,
        "speedup_min_to_min": speedup_min,
        "min_end_to_end_speedup": workload.performance.min_end_to_end_speedup,
    }
    (output / "campaign.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"\nspeedup: {speedup_median:.4f}x median, {speedup_min:.4f}x min-to-min "
        f"(gate {workload.performance.min_end_to_end_speedup}x)"
    )
    return 0


def _run_arm(workload, mode, backend, runs, warmups, output):
    """Generate ``runs`` times, returning wall times and a frame path."""
    from fastvideo import VideoGenerator

    generator = VideoGenerator.from_pretrained(
        workload.model.model_id,
        num_gpus=workload.runtime.num_gpus,
        text_encoder_cpu_offload=workload.runtime.text_encoder_cpu_offload,
        dit_cpu_offload=workload.runtime.dit_cpu_offload,
        vae_cpu_offload=workload.runtime.vae_cpu_offload,
    )
    sampling = workload.sampling
    kwargs = dict(
        prompt=workload.prompt,
        height=sampling.height,
        width=sampling.width,
        num_frames=sampling.num_frames,
        num_inference_steps=sampling.num_inference_steps,
        guidance_scale=sampling.guidance_scale,
        seed=sampling.seed,
        return_frames=True,
        save_video=False,
        output_path=str(output / mode),
    )

    for _ in range(warmups):
        generator.generate_video(**kwargs)

    timings: list[float] = []
    frames = None
    for index in range(runs):
        start = time.perf_counter()
        result = generator.generate_video(**kwargs)
        timings.append(time.perf_counter() - start)
        if index == 0:
            frames = _save_frames(result, output / f"{mode}_frames.npy")
    return timings, frames


def _save_frames(result, path: Path):
    """Persist one generation's frames for the perceptual comparison."""
    import numpy as np

    array = np.asarray(result[0] if isinstance(result, (list, tuple)) else result)
    np.save(path, array)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
