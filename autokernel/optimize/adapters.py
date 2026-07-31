"""Production adapters for the built-in optimize stages.

The optimize runner owns process isolation and the versioned ``result.json``
envelope.  This module translates that envelope to the existing FastVideo and
MotionKernel APIs.  Kernel search and isolated validation intentionally remain
external commands; their measured output is consumed by the package adapter.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from autokernel.artifact import ArtifactError, package_artifact
from autokernel.discovery import (
    DiscoveryError,
    correlate_discovery_report,
    load_profiler_export,
    rank_regions,
    write_discovery_report,
)
from autokernel.specgen import SpecGenerationError, write_generated_artifacts
from autokernel.workload import WorkloadError, load_workload
from autokernel.workload.launcher import run_ab, run_mode
from autokernel.workload.result import (
    classify_end_to_end,
    compare_frame_outputs,
    load_generation_result,
)

from .state import read_json
from .types import STAGE_RESULT_SCHEMA_VERSION


class ProductionAdapterError(RuntimeError):
    """A production stage could not satisfy its fail-closed contract."""


def _result(stage: str, message: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": STAGE_RESULT_SCHEMA_VERSION,
        "stage": stage,
        "status": "ok",
        "message": message,
        **extra,
    }


def _stage_input(run_dir: Path, stage: str) -> dict[str, Any]:
    path = run_dir / "stages" / stage / "input.json"
    try:
        payload = read_json(path)
    except Exception as exc:
        raise ProductionAdapterError(f"cannot read stage input {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("stage") != stage:
        raise ProductionAdapterError(f"invalid stage input identity for {stage!r}")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ProductionAdapterError(f"stage input for {stage!r} has no config object")
    return payload


def _prior_result(run_dir: Path, stage: str) -> dict[str, Any]:
    path = run_dir / "stages" / stage / "result.json"
    try:
        payload = read_json(path)
    except Exception as exc:
        raise ProductionAdapterError(
            f"cannot read prerequisite result {path}: {exc}"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != STAGE_RESULT_SCHEMA_VERSION
        or payload.get("stage") != stage
        or payload.get("status") != "ok"
    ):
        raise ProductionAdapterError(f"prerequisite stage {stage!r} is not complete")
    return payload


def _config_path(config: Mapping[str, Any], name: str) -> Path:
    value = config.get(name)
    if not isinstance(value, str) or not value:
        raise ProductionAdapterError(f"campaign config {name!r} must be a path")
    return Path(value)


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ProductionAdapterError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProductionAdapterError(f"{name} must be a finite number") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        qualifier = " positive" if positive else ""
        raise ProductionAdapterError(f"{name} must be a finite{qualifier} number")
    return number


def _baseline(run_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    output = run_dir / "generation"
    payload = run_ab(
        fastvideo_checkout=_config_path(config, "fastvideo_checkout"),
        workload=_config_path(config, "workload"),
        output_dir=output,
        model_override=str(config["model"]),
        modes=("native",),
        resume=True,
    )
    native = payload["results"]["native"]
    median = _finite(native.get("median_wall_seconds"), "native median", positive=True)
    result_path = output / "native_result.json"
    return _result(
        "baseline",
        "native FastVideo baseline complete",
        metrics={
            "median_wall_seconds": median,
            "runs": native.get("runs"),
            "baseline_mode": config.get("baseline", "eager"),
        },
        artifacts={"generation_result": str(result_path)},
    )


def _profile(run_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    output = run_dir / "profile"
    profiler_path = run_dir / "stages" / "profile" / "profiler.json"
    workload = load_workload(_config_path(config, "workload"))
    profile_env = {
        **(workload.mode_env.for_mode("native") if workload.mode_env else {}),
        "FASTVIDEO_OPTIMIZATION_PROFILE_CAPTURE_FX": "1",
        "FASTVIDEO_OPTIMIZATION_PROFILE_FX_TRACER": os.environ.get(
            "FASTVIDEO_OPTIMIZATION_PROFILE_FX_TRACER", "auto"
        ),
    }
    run_mode(
        fastvideo_checkout=_config_path(config, "fastvideo_checkout"),
        workload=_config_path(config, "workload"),
        mode="native",
        output_dir=output,
        model_override=str(config["model"]),
        profile_output=profiler_path,
        env=profile_env,
    )
    generation_path = output / "native_result.json"
    generation = load_generation_result(generation_path)
    if generation.status != "ok" or not profiler_path.is_file():
        raise ProductionAdapterError(
            "FastVideo profiling did not produce a valid export"
        )
    return _result(
        "profile",
        "dedicated post-warmup FastVideo profile complete",
        metrics={
            "median_wall_seconds": generation.median_wall_seconds,
            "runs": generation.runs,
        },
        artifacts={
            "profiler_export": str(profiler_path),
            "generation_result": str(generation_path),
        },
    )


def _discover(run_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    del config
    profile = _prior_result(run_dir, "profile")
    artifacts = profile.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(
        artifacts.get("profiler_export"), str
    ):
        raise ProductionAdapterError("profile result has no profiler_export")
    profiler_path = Path(artifacts["profiler_export"])
    report = load_profiler_export(profiler_path)
    raw = read_json(profiler_path)
    rows = raw.get("rows") if isinstance(raw, Mapping) else None
    if not isinstance(rows, list):
        raise ProductionAdapterError("profiler export rows must be a list")
    correlated = correlate_discovery_report(rows, report)
    report_path = run_dir / "stages" / "discover" / "discovery.json"
    write_discovery_report(correlated, report_path)
    ranked = rank_regions(
        correlated.regions,
        total_cuda_time_us=correlated.total_cuda_time_us,
    )
    candidates = [item.as_dict() for item in ranked if item.search_worthy]
    extra: dict[str, Any] = {}
    if not candidates:
        extra["recommendation"] = "no_worthwhile_candidate"
    return _result(
        "discover",
        (
            f"{len(candidates)} search-worthy captured region(s)"
            if candidates
            else "no search-worthy captured regions above the impact floor"
        ),
        metrics={
            "total_cuda_time_us": correlated.total_cuda_time_us,
            "regions": len(correlated.regions),
            "ranked_regions": len(ranked),
            "search_worthy_regions": len(candidates),
            "graph_breaks": len(correlated.graph_breaks),
        },
        candidates=candidates,
        artifacts={"discovery_report": str(report_path)},
        **extra,
    )


def _specgen(run_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    del config
    discover = _prior_result(run_dir, "discover")
    candidates = discover.get("candidates")
    artifacts = discover.get("artifacts")
    if not isinstance(candidates, list):
        raise ProductionAdapterError("discover result candidates must be a list")
    if not isinstance(artifacts, Mapping) or not isinstance(
        artifacts.get("discovery_report"), str
    ):
        raise ProductionAdapterError("discover result has no discovery_report")
    if not candidates:
        return _result(
            "specgen",
            "no discovered candidates require specifications",
            metrics={"specs_generated": 0},
            candidates=[],
            recommendation="no_worthwhile_candidate",
        )

    generated: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ProductionAdapterError("discovery candidate must be an object")
        fingerprint = candidate.get("fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ProductionAdapterError("discovery candidate has no fingerprint")
        output = run_dir / "candidates" / fingerprint
        paths = write_generated_artifacts(
            artifacts["discovery_report"], output, fingerprint=fingerprint
        )
        generated.append(
            {
                **dict(candidate),
                "generated": {name: str(path) for name, path in paths.items()},
            }
        )
    return _result(
        "specgen",
        f"generated {len(generated)} graph-derived specification(s)",
        metrics={"specs_generated": len(generated)},
        candidates=generated,
    )


def _package(run_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    isolated = _prior_result(run_dir, "isolated_validate")
    requests = isolated.get("package_requests")
    if not isinstance(requests, list) or not requests:
        raise ProductionAdapterError(
            "isolated_validate must provide a non-empty package_requests list "
            "containing measured artifact sections"
        )
    artifact_root = run_dir / str(config.get("artifact_dir_name") or "artifacts")
    packaged: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        if not isinstance(request, Mapping):
            raise ProductionAdapterError(f"package_requests[{index}] must be an object")
        source_value = request.get("source_dir")
        sections = request.get("sections")
        if not isinstance(source_value, str) or not isinstance(sections, Mapping):
            raise ProductionAdapterError(
                f"package_requests[{index}] requires source_dir and sections"
            )
        artifact_id = sections.get("artifact_id")
        if (
            not isinstance(artifact_id, str)
            or Path(artifact_id).name != artifact_id
            or artifact_id in {".", ".."}
        ):
            raise ProductionAdapterError(
                f"package_requests[{index}].sections.artifact_id is not a safe id"
            )
        promotion = sections.get("promotion")
        evidence = sections.get("evidence")
        if (
            not isinstance(promotion, Mapping)
            or promotion.get("decision") != "quarantined"
        ):
            raise ProductionAdapterError(
                "pre-validation package promotion must be 'quarantined'"
            )
        if not isinstance(evidence, Mapping):
            raise ProductionAdapterError("package evidence must be an object")
        benchmark = evidence.get("benchmark")
        generation = evidence.get("generation")
        if not isinstance(benchmark, Mapping) or benchmark.get("passed") is not True:
            raise ProductionAdapterError(
                "package requires passing measured isolated benchmark evidence"
            )
        if not isinstance(generation, Mapping) or generation.get("passed") is not False:
            raise ProductionAdapterError(
                "pre-validation package generation evidence must remain pending/failed"
            )
        source = Path(source_value)
        if not source.is_absolute():
            source = run_dir / source
        output = artifact_root / artifact_id
        manifest = package_artifact(
            source,
            output,
            sections,
            overwrite=True,
        )
        packaged.append(
            {
                "artifact_id": manifest.artifact_id,
                "bundle_dir": str(output),
                "manifest": str(output / "artifact.json"),
            }
        )
    return _result(
        "package",
        f"packaged {len(packaged)} quarantined artifact bundle(s)",
        metrics={"artifacts_packaged": len(packaged)},
        artifacts={"root": str(artifact_root), "bundles": packaged},
    )


def _dispatch_selected(path: Path) -> tuple[bool, dict[str, Any]]:
    try:
        payload = read_json(path)
    except Exception as exc:
        raise ProductionAdapterError(
            f"cannot read dispatch diagnostics {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ProductionAdapterError("dispatch diagnostics must be an object")
    dispatch = payload.get("dispatch")
    counts = dispatch.get("reason_counts") if isinstance(dispatch, Mapping) else None
    selected = counts.get("artifact_selected", 0) if isinstance(counts, Mapping) else 0
    return isinstance(selected, int) and selected > 0, payload


def _end_to_end_validate(run_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    packaged = _prior_result(run_dir, "package")
    artifacts = packaged.get("artifacts")
    artifact_root = artifacts.get("root") if isinstance(artifacts, Mapping) else None
    if not isinstance(artifact_root, str):
        raise ProductionAdapterError("package result has no artifact root")

    generation_dir = run_dir / "generation"
    native_path = generation_dir / "native_result.json"
    native = load_generation_result(native_path)
    workload = load_workload(_config_path(config, "workload"))
    diagnostics_path = run_dir / "stages" / "end_to_end_validate" / "dispatch.json"
    candidate_env = {
        **(workload.mode_env.for_mode("candidate") if workload.mode_env else {}),
        "FASTVIDEO_OPTIMIZATION_ARTIFACT_DIR": artifact_root,
        "FASTVIDEO_OPTIMIZATION_ARTIFACT_MODEL_ID": str(config["model"]),
        "FASTVIDEO_OPTIMIZATION_ARTIFACT_VALIDATION": "1",
        "FASTVIDEO_OPTIMIZATION_ARTIFACT_DIAGNOSTICS": str(diagnostics_path),
    }
    run_mode(
        fastvideo_checkout=_config_path(config, "fastvideo_checkout"),
        workload=_config_path(config, "workload"),
        mode="candidate",
        output_dir=generation_dir,
        model_override=str(config["model"]),
        env=candidate_env,
    )
    candidate_path = generation_dir / "candidate_result.json"
    candidate = load_generation_result(candidate_path)
    configured_threshold = _finite(
        config.get("min_e2e_speedup", 1.01), "min_e2e_speedup"
    )
    workload_threshold = (
        workload.performance.min_end_to_end_speedup
        if workload.performance is not None
        else 1.01
    )
    threshold = max(configured_threshold, workload_threshold)
    memory_limit = (
        workload.performance.max_peak_memory_regression
        if workload.performance is not None
        else 0.05
    )
    performance = classify_end_to_end(
        native,
        candidate,
        min_speedup=threshold,
        max_peak_memory_regression=memory_limit,
    )
    parity = compare_frame_outputs(
        native.frames_path,
        candidate.frames_path,
        policy=workload.parity.policy if workload.parity else "byte_equal",
        atol=(
            workload.parity.atol
            if workload.parity and workload.parity.atol is not None
            else 0.0
        ),
        rtol=(
            workload.parity.rtol
            if workload.parity and workload.parity.rtol is not None
            else 0.0
        ),
    )
    selected, diagnostics = _dispatch_selected(diagnostics_path)
    classification = performance.get("classification")
    if not parity.get("passed") or not selected:
        classification = "failed"
        recommendation = "failed"
        message = "end-to-end validation failed: " + (
            "output parity failed"
            if not parity.get("passed")
            else "artifact was not selected"
        )
    elif classification == "improved":
        recommendation = "promoted"
        message = "artifact passed full-generation parity and speedup gates"
    elif classification in {"neutral", "regressed"}:
        recommendation = "no_worthwhile_candidate"
        message = "artifact passed parity but not the end-to-end speedup gate"
    else:
        recommendation = "failed"
        message = "end-to-end measurement could not be classified"
    metrics = {
        **performance,
        "classification": classification,
        "parity_passed": bool(parity.get("passed")),
        "artifact_selected": selected,
    }
    return _result(
        "end_to_end_validate",
        message,
        metrics=metrics,
        recommendation=recommendation,
        artifacts={
            "native_result": str(native_path),
            "candidate_result": str(candidate_path),
            "dispatch_diagnostics": str(diagnostics_path),
        },
        parity=parity,
        dispatch=diagnostics,
    )


_ADAPTERS: dict[str, Callable[[Path, Mapping[str, Any]], dict[str, Any]]] = {
    "baseline": _baseline,
    "profile": _profile,
    "discover": _discover,
    "specgen": _specgen,
    "package": _package,
    "end_to_end_validate": _end_to_end_validate,
}


def run_production_stage(stage: str, run_dir: str | Path) -> dict[str, Any]:
    """Run a built-in production adapter and return a stage-result payload."""
    adapter = _ADAPTERS.get(stage)
    if adapter is None:
        raise ProductionAdapterError(
            f"stage {stage!r} has no built-in production adapter; "
            "configure it with --stage-commands"
        )
    root = Path(run_dir)
    stage_input = _stage_input(root, stage)
    try:
        return adapter(root, stage_input["config"])
    except ProductionAdapterError:
        raise
    except (ArtifactError, DiscoveryError, SpecGenerationError, WorkloadError) as exc:
        raise ProductionAdapterError(str(exc)) from exc
