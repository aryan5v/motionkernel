"""Default stage driver subprocess for the optimize control plane.

Production stages call into real MotionKernel / FastVideo tools when available.
CPU tests inject alternate stage commands instead of this driver.

Environment:
  MOTIONKERNEL_RUN_DIR — run directory
  MOTIONKERNEL_STAGE — stage name
  MOTIONKERNEL_BASELINE — eager|compile

Optional simulate mode for offline smoke (writes synthetic results):
  MOTIONKERNEL_SIMULATE=1
  MOTIONKERNEL_SIMULATE_OUTCOME=promoted|no_worthwhile_candidate|fail_at:<stage>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .adapters import ProductionAdapterError, run_production_stage
from .types import STAGE_RESULT_SCHEMA_VERSION, default_repo_root


def _write_result(run_dir: Path, stage: str, payload: dict[str, Any]) -> Path:
    path = run_dir / "stages" / stage / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _simulate(stage: str, run_dir: Path, outcome: str) -> int:
    """Deterministic CPU simulation used by integration tests / smoke."""
    if outcome.startswith("fail_at:"):
        fail_stage = outcome.split(":", 1)[1]
        if stage == fail_stage:
            _write_result(
                run_dir,
                stage,
                {
                    "schema_version": STAGE_RESULT_SCHEMA_VERSION,
                    "stage": stage,
                    "status": "failed",
                    "message": f"simulated failure at {stage}",
                },
            )
            return 1

    candidates: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    message = f"simulated {stage} ok"
    extra: dict[str, Any] = {}

    if stage == "baseline":
        metrics = {
            "median_wall_seconds": 10.0,
            "baseline_mode": os.environ.get("MOTIONKERNEL_BASELINE", "eager"),
        }
    elif stage == "profile":
        metrics = {"total_cuda_time_us": 1_000_000.0}
    elif stage == "discover":
        if outcome == "no_worthwhile_candidate":
            candidates = []
            message = "no search-worthy candidates above impact floor"
            extra["recommendation"] = "no_worthwhile_candidate"
        else:
            candidates = [
                {
                    "name": "toy.elementwise",
                    "fingerprint": "fp_toy_001",
                    "estimated_max_e2e_improvement": 0.05,
                }
            ]
            message = "1 search-worthy candidate"
    elif stage == "specgen":
        metrics = {"specs_generated": 1 if outcome != "no_worthwhile_candidate" else 0}
        if outcome == "no_worthwhile_candidate":
            extra["recommendation"] = "no_worthwhile_candidate"
    elif stage == "search":
        metrics = {"kernels_written": 1 if outcome != "no_worthwhile_candidate" else 0}
    elif stage == "isolated_validate":
        # Isolated speedup alone must not promote.
        metrics = {
            "isolated_speedup": 8.5 if outcome == "promoted" else 1.02,
            "isolated_correct": True,
        }
        message = "isolated validation complete (not sufficient for promotion)"
    elif stage == "package":
        artifact_dir = run_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "manifest.json").write_text(
            json.dumps({"schema_version": 1, "opaque": True}) + "\n",
            encoding="utf-8",
        )
        metrics = {"artifact_dir": str(artifact_dir)}
        extra["artifact_dir"] = str(artifact_dir)
    elif stage == "end_to_end_validate":
        if outcome == "promoted":
            metrics = {
                "native_median_wall_seconds": 10.0,
                "optimized_median_wall_seconds": 9.0,
                "end_to_end_speedup": 10.0 / 9.0,
                "classification": "improved",
            }
            message = "e2e improvement meets promotion threshold"
            extra["recommendation"] = "promoted"
        elif outcome == "no_worthwhile_candidate":
            metrics = {
                "native_median_wall_seconds": 10.0,
                "optimized_median_wall_seconds": 10.0,
                "end_to_end_speedup": 1.0,
                "classification": "neutral",
            }
            message = "e2e neutral — do not promote despite isolated speedup"
            extra["recommendation"] = "no_worthwhile_candidate"
        else:
            metrics = {"classification": "failed"}
            extra["recommendation"] = "failed"
    elif stage == "finalize":
        if outcome == "promoted":
            metrics = {"artifacts_promoted": 1, "artifacts_rejected": 0}
            message = "quarantined bundle finalized as promoted"
            extra["recommendation"] = "promoted"
        elif outcome == "no_worthwhile_candidate":
            metrics = {"artifacts_promoted": 0, "artifacts_rejected": 1}
            message = "measured bundle finalized as rejected"
            extra["recommendation"] = "no_worthwhile_candidate"
        else:
            metrics = {"artifacts_promoted": 0, "artifacts_rejected": 0}
            message = "no bundle could be finalized; quarantine preserved"
            extra["recommendation"] = "failed"

    payload: dict[str, Any] = {
        "schema_version": STAGE_RESULT_SCHEMA_VERSION,
        "stage": stage,
        "status": "ok",
        "message": message,
        "metrics": metrics,
        **extra,
    }
    if stage == "discover":
        payload["candidates"] = candidates
    _write_result(run_dir, stage, payload)
    return 0


def _run_real(stage: str, run_dir: Path, repo_root: Path) -> int:
    """Run one concrete adapter, failing through the stage-result contract."""
    try:
        payload = run_production_stage(stage, run_dir)
    except Exception as exc:  # noqa: BLE001 - preserve the subprocess JSON contract
        message = str(exc)
        if not isinstance(exc, ProductionAdapterError):
            message = f"{type(exc).__name__}: {message}"
        _write_result(
            run_dir,
            stage,
            {
                "schema_version": STAGE_RESULT_SCHEMA_VERSION,
                "stage": stage,
                "status": "failed",
                "message": message,
                "metrics": {"repo_root": str(repo_root)},
            },
        )
        return 1
    _write_result(run_dir, stage, payload)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MotionKernel optimize stage driver")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args(argv)

    run_dir = args.run_dir
    stage = args.stage
    repo_root = args.repo_root or default_repo_root()

    if os.environ.get("MOTIONKERNEL_SIMULATE") == "1":
        outcome = os.environ.get("MOTIONKERNEL_SIMULATE_OUTCOME", "promoted")
        return _simulate(stage, run_dir, outcome)
    return _run_real(stage, run_dir, repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
