"""Drive the dispatch-overhead measurement against a FastVideo checkout.

This is the regression harness for the published per-call overhead number. It
reproduces, on demand, the two measurements that produced the stale 3.104 ms
figure (docs/LTX_V1_R4_ROOT_CAUSE.md section 7):

1. an end-to-end A/B (native vs candidate, >=15 timed runs per arm, same
   process structure, frames compared under the workload parity policy);
2. an in-situ shadow profile (``FASTVIDEO_OPTIMIZATION_ARTIFACT_TIMING=shadow``)
   attributing the candidate path against the native forward on identical
   inputs.

Everything lands in one immutable output directory: the derived workload
variants, per-arm launcher results, frame arrays, dispatch diagnostics, the
timing report, and one validated measurement record JSON. A record is only
written when parity holds and the artifact path actually engaged -- an
overhead number without byte-equal output is not a measurement of anything
worth publishing.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from autokernel._io import write_json_atomic
from autokernel.workload import load_workload
from autokernel.workload.launcher import run_mode
from autokernel.workload.result import compare_frame_outputs, load_generation_result
from autokernel.workload.types import MeasurementSpec, WorkloadManifest, dump_workload

from .overhead import (
    DEFAULT_CALL_VOLUMES,
    DEFAULT_GATE,
    DispatchAnalysisError,
    TimingReport,
    attribute_overhead,
    breakeven_curve,
    overhead_from_e2e,
)

MEASUREMENT_SCHEMA = "motionkernel.dispatch-measurement"
MEASUREMENT_SCHEMA_VERSION = 1

#: Measurement discipline (docs/agent-briefs/TRACK_D_DISPATCH.md section 5):
#: runs: 2 and runs: 5 both produced conclusions that later reversed.
MIN_TIMED_RUNS = 15

#: The shadow profile doubles the region's cost (an extra native forward per
#: dispatched call), so it runs as a short separate generation, not the 15-run
#: arm. 3 generations still give >1000 attributed calls on a 384-call stack.
PROFILE_RUNS = 2


class MeasurementError(RuntimeError):
    """The measurement could not be completed or produced unusable evidence."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _arm_stats(wall_seconds: Sequence[float]) -> dict[str, Any]:
    if not wall_seconds:
        raise MeasurementError("an arm produced no timed runs")
    return {
        "runs": [round(value, 4) for value in wall_seconds],
        "count": len(wall_seconds),
        "median": round(statistics.median(wall_seconds), 4),
        "stdev": (
            round(statistics.stdev(wall_seconds), 4) if len(wall_seconds) > 1 else 0.0
        ),
        "min": round(min(wall_seconds), 4),
        "max": round(max(wall_seconds), 4),
    }


def _derive_workload(
    manifest: WorkloadManifest,
    *,
    runs: int,
    warmups: int,
    path: Path,
) -> Path:
    """Write a measurement-variant manifest with explicit run counts."""
    derived = replace(
        manifest,
        measurement=MeasurementSpec(
            warmups=warmups,
            runs=runs,
            save_frames=(manifest.measurement or MeasurementSpec()).save_frames,
            save_video=False,
        ),
    )
    dump_workload(derived, path)
    return path


def _artifact_env(
    manifest: WorkloadManifest,
    *,
    artifact_root: Path,
    model_id: str,
    diagnostics_path: Path,
    shadow: bool,
) -> dict[str, str]:
    mode_env: dict[str, str] = {}
    if manifest.mode_env is not None:
        mode_env = dict(manifest.mode_env.for_mode("candidate"))
    env = {
        **mode_env,
        "FASTVIDEO_OPTIMIZATION_ARTIFACT_DIR": str(artifact_root),
        "FASTVIDEO_OPTIMIZATION_ARTIFACT_MODEL_ID": model_id,
        "FASTVIDEO_OPTIMIZATION_ARTIFACT_VALIDATION": "1",
        "FASTVIDEO_OPTIMIZATION_ARTIFACT_DIAGNOSTICS": str(diagnostics_path),
    }
    if shadow:
        env["FASTVIDEO_OPTIMIZATION_ARTIFACT_TIMING"] = "shadow"
    return env


def _candidate_calls_from_diagnostics(path: Path) -> tuple[int, int]:
    """Total candidate calls and runtime fallbacks across dispatch decisions."""
    if not path.is_file():
        raise MeasurementError(f"dispatch diagnostics missing: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MeasurementError(f"invalid dispatch diagnostics {path}: {exc}") from exc
    decisions = raw.get("decisions")
    if not isinstance(decisions, list):
        raise MeasurementError(f"dispatch diagnostics {path}: decisions must be a list")
    candidate_calls = 0
    fallbacks = 0
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise MeasurementError(f"dispatch diagnostics {path}: decision must be an object")
        candidate_calls += int(decision.get("candidate_calls") or 0)
        fallbacks += int(decision.get("runtime_fallbacks") or 0)
    return candidate_calls, fallbacks


def run_dispatch_measurement(
    *,
    fastvideo_checkout: str | Path,
    workload: str | Path,
    artifact_root: str | Path,
    output_dir: str | Path,
    model_override: str | None = None,
    runs: int = MIN_TIMED_RUNS,
    warmups: int = 1,
    kernel_saving_ms_per_call: float | None = None,
    gate: float = DEFAULT_GATE,
    call_volumes: Sequence[int] = DEFAULT_CALL_VOLUMES,
    python: str | None = None,
    env_extra: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Run the full measurement and write the measurement record.

    Arms run in separate processes through the FastVideo launcher, so mode
    environment never leaks across them. The A/B arms and the shadow profile
    run on the same node in the same session, back to back.
    """
    if runs < MIN_TIMED_RUNS:
        raise MeasurementError(
            f"runs={runs} is below the {MIN_TIMED_RUNS}-run minimum; "
            "R4's runs: 2 and runs: 5 measurements both reversed"
        )
    fastvideo = Path(fastvideo_checkout).expanduser().resolve()
    workload_path = Path(workload).expanduser().resolve()
    artifacts = Path(artifact_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not fastvideo.is_dir():
        raise MeasurementError(f"FastVideo checkout not found: {fastvideo}")
    if not artifacts.is_dir():
        raise MeasurementError(f"artifact root not found: {artifacts}")
    manifest = load_workload(workload_path)
    model_id = model_override or manifest.model.model_id
    output.mkdir(parents=True, exist_ok=True)

    timed_workload = _derive_workload(
        manifest, runs=runs, warmups=warmups, path=output / "workload_timed.yaml"
    )
    profile_workload = _derive_workload(
        manifest,
        runs=PROFILE_RUNS,
        warmups=warmups,
        path=output / "workload_profile.yaml",
    )

    native_dir = output / "native"
    candidate_dir = output / "candidate"
    profile_dir = output / "profile-shadow"

    native_env: dict[str, str] = {}
    if manifest.mode_env is not None:
        native_env = dict(manifest.mode_env.for_mode("native"))
    if env_extra:
        native_env.update(env_extra)
    run_mode(
        fastvideo_checkout=fastvideo,
        workload=timed_workload,
        mode="native",
        output_dir=native_dir,
        model_override=model_override,
        env=native_env or None,
        python=python,
        timeout=timeout,
    )
    candidate_env = _artifact_env(
        manifest,
        artifact_root=artifacts,
        model_id=model_id,
        diagnostics_path=candidate_dir / "dispatch.json",
        shadow=False,
    )
    if env_extra:
        candidate_env.update(env_extra)
    run_mode(
        fastvideo_checkout=fastvideo,
        workload=timed_workload,
        mode="candidate",
        output_dir=candidate_dir,
        model_override=model_override,
        env=candidate_env,
        python=python,
        timeout=timeout,
    )
    shadow_env = _artifact_env(
        manifest,
        artifact_root=artifacts,
        model_id=model_id,
        diagnostics_path=profile_dir / "dispatch.json",
        shadow=True,
    )
    if env_extra:
        shadow_env.update(env_extra)
    run_mode(
        fastvideo_checkout=fastvideo,
        workload=profile_workload,
        mode="candidate",
        output_dir=profile_dir,
        model_override=model_override,
        env=shadow_env,
        python=python,
        timeout=timeout,
    )

    native_result = load_generation_result(native_dir / "native_result.json")
    candidate_result = load_generation_result(candidate_dir / "candidate_result.json")
    if native_result.status != "ok":
        raise MeasurementError(f"native arm failed: {native_result.failure_reason}")
    if candidate_result.status != "ok":
        raise MeasurementError(f"candidate arm failed: {candidate_result.failure_reason}")

    parity_policy = (manifest.parity.policy if manifest.parity is not None else "byte_equal")
    parity = compare_frame_outputs(
        native_result.frames_path,
        candidate_result.frames_path,
        policy=parity_policy,
    )

    native_stats = _arm_stats(list(native_result.wall_seconds))
    candidate_stats = _arm_stats(list(candidate_result.wall_seconds))
    speedup_median = native_stats["median"] / candidate_stats["median"]
    speedup_min_to_min = native_stats["min"] / candidate_stats["min"]

    candidate_calls, runtime_fallbacks = _candidate_calls_from_diagnostics(
        candidate_dir / "dispatch.json"
    )
    generations = warmups + runs
    if candidate_calls == 0:
        raise MeasurementError(
            "candidate arm made zero candidate calls; the artifact path never "
            "engaged, so there is no dispatch path to measure"
        )
    calls_per_generation = candidate_calls / generations

    timing_report = TimingReport.from_dict(
        json.loads((profile_dir / "timing.json").read_text(encoding="utf-8")),
        source=str(profile_dir / "timing.json"),
    )
    attribution = attribute_overhead(timing_report)

    e2e_overhead = None
    curve = None
    if kernel_saving_ms_per_call is not None:
        e2e_overhead = overhead_from_e2e(
            native_median_seconds=native_stats["median"],
            candidate_median_seconds=candidate_stats["median"],
            calls_per_generation=round(calls_per_generation),
            kernel_saving_ms_per_call=kernel_saving_ms_per_call,
        )
        curve = breakeven_curve(
            native_e2e_seconds=native_stats["median"],
            overhead_ms_per_call=e2e_overhead.overhead_ms_per_call,
            gate=gate,
            call_volumes=call_volumes,
        )

    environment = dict(candidate_result.environment or {})
    capability = str(environment.get("gpu_capability") or "")
    arch = f"sm{capability.replace('.', '')}" if capability else "unknown"

    record: dict[str, Any] = {
        "schema": MEASUREMENT_SCHEMA,
        "schema_version": MEASUREMENT_SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "workload_id": manifest.workload_id,
        "model_id": model_id,
        "artifact_root": str(artifacts),
        "fastvideo_checkout": str(fastvideo),
        "environment": environment,
        "arch": arch,
        "gate": gate,
        "generations_per_arm": generations,
        "timed_runs_per_arm": runs,
        "native": native_stats,
        "candidate": candidate_stats,
        "parity": parity,
        "e2e": {
            "speedup_median": round(speedup_median, 4),
            "speedup_min_to_min": round(speedup_min_to_min, 4),
        },
        "candidate_calls": candidate_calls,
        "runtime_fallbacks": runtime_fallbacks,
        "calls_per_generation": round(calls_per_generation, 2),
        "shadow_attribution": attribution.as_dict(),
        "timing_report_path": str(profile_dir / "timing.json"),
        "dispatch_diagnostics_path": str(candidate_dir / "dispatch.json"),
    }
    if e2e_overhead is not None:
        record["e2e_overhead"] = e2e_overhead.as_dict()
    if curve is not None:
        record["breakeven"] = [point.as_dict() for point in curve]

    if not parity.get("passed"):
        record["status"] = "parity_failed"
        write_json_atomic(output / "measurement.json", record)
        raise MeasurementError(
            f"parity check failed under {parity_policy}: {parity.get('reason')}; "
            "partial record written"
        )
    if runtime_fallbacks:
        record["status"] = "fallbacks_recorded"
    else:
        record["status"] = "ok"
    write_json_atomic(output / "measurement.json", record)
    return record
