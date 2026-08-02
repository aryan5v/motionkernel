"""Capture the launch controls a timed measurement ran under.

A per-call overhead number is only as good as its variance discipline. R4's
own history: a 38.4% native-arm spread within one trial reversed
per-artifact verdicts (docs/LTX_V1_R4_ROOT_CAUSE.md section 3). Every timed
measurement therefore records the controls it ran under -- node exclusivity
and GPU clock state -- and the measured native-arm coefficient of variation
against a declared ceiling. A measurement whose native arm exceeds the
ceiling is recorded, but it is not promotion evidence.

The controls are *captured*, not assumed: the sbatch requests an exclusive
node and attempts a clock lock, and this module records what actually
happened, including failure to lock.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Mapping, Sequence

#: Declared ceiling for the native-arm coefficient of variation.
DEFAULT_CV_CEILING = 0.03


def _parse_clock_row(row: str) -> tuple[int, ...]:
    values = []
    for part in row.split(","):
        part = part.strip()
        if not part.isdigit():
            raise ValueError(f"unparseable clock value {part!r}")
        values.append(int(part))
    return tuple(values)


def query_gpu_clocks(
    *,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    """Current and max SM/memory clocks from nvidia-smi, best-effort."""
    try:
        completed = runner(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"state": "unknown", "error": f"{type(exc).__name__}: {exc}"}
    if completed.returncode != 0:
        return {
            "state": "unknown",
            "error": f"nvidia-smi exited {completed.returncode}: {(completed.stderr or '').strip()[:200]}",
        }
    try:
        sm, mem, sm_max, mem_max = _parse_clock_row(completed.stdout.strip().splitlines()[0])
    except (ValueError, IndexError) as exc:
        return {"state": "unknown", "error": f"unparseable nvidia-smi output: {exc}"}
    return {
        "sm_clock_mhz": sm,
        "mem_clock_mhz": mem,
        "sm_clock_max_mhz": sm_max,
        "mem_clock_max_mhz": mem_max,
        "at_max": sm == sm_max and mem == mem_max,
    }


def capture_controls(
    env: Mapping[str, str] | None = None,
    *,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    """The launch controls for one measurement, as actually applied.

    ``MOTIONKERNEL_NODE_EXCLUSIVE`` / ``MOTIONKERNEL_CLOCKS_LOCKED`` are set
    by the launch script after it *successfully* applies each control; their
    absence records that the control was not applied, not that it was.
    """
    environ = dict(os.environ if env is None else env)
    node_exclusive = environ.get("MOTIONKERNEL_NODE_EXCLUSIVE") == "1"
    clocks_locked = environ.get("MOTIONKERNEL_CLOCKS_LOCKED") == "1"
    clocks = query_gpu_clocks(runner=runner)
    if "state" not in clocks:
        clocks["state"] = "locked" if clocks_locked else "default"
    return {
        "node_exclusive": node_exclusive,
        "gpu_clocks": clocks,
        "slurm_job_id": environ.get("SLURM_JOB_ID") or None,
    }


def variance_block(
    native_wall_seconds: Sequence[float],
    candidate_wall_seconds: Sequence[float],
    *,
    cv_ceiling: float = DEFAULT_CV_CEILING,
) -> dict[str, Any]:
    """Measured per-arm coefficients of variation and the gating verdict."""
    import statistics

    def _cv(values: Sequence[float]) -> float | None:
        samples = [float(value) for value in values]
        if len(samples) < 2:
            return None
        mean = statistics.fmean(samples)
        if mean <= 0:
            return None
        return statistics.stdev(samples) / mean

    native_cv = _cv(native_wall_seconds)
    candidate_cv = _cv(candidate_wall_seconds)
    if native_cv is None:
        valid = False
        reason = "native-arm variance could not be measured (fewer than two runs)"
    elif native_cv > cv_ceiling:
        valid = False
        reason = (
            f"native-arm coefficient of variation {native_cv:.4f} exceeds the "
            f"{cv_ceiling:.2f} ceiling; invalid for gating"
        )
    else:
        valid = True
        reason = (
            f"native-arm coefficient of variation {native_cv:.4f} within the "
            f"{cv_ceiling:.2f} ceiling"
        )
    return {
        "native_cv": round(native_cv, 4) if native_cv is not None else None,
        "candidate_cv": round(candidate_cv, 4) if candidate_cv is not None else None,
        "cv_ceiling": cv_ceiling,
        "valid_for_gating": valid,
        "reason": reason,
    }
