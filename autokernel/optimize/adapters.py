"""Production adapters for the built-in optimize stages.

The optimize runner owns process isolation and the versioned ``result.json``
envelope. This module translates that envelope to the existing FastVideo and
MotionKernel APIs, including autonomous search and independently measured
isolated validation.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from autokernel.artifact import (
    ArtifactError,
    GenerationOutcome,
    finalize_bundle,
    package_artifact,
    verify_bundle,
)
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

from .search import BuiltinSearchError, search_candidates, validate_candidates
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


def _search(run_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    specified = _prior_result(run_dir, "specgen")
    candidates = specified.get("candidates")
    if not isinstance(candidates, list):
        raise ProductionAdapterError("specgen result candidates must be a list")
    outcome = search_candidates(run_dir, candidates, config)
    searched = outcome["candidates"]
    return _result(
        "search",
        (
            f"autonomous search produced {len(searched)} faster candidate(s)"
            if searched
            else "autonomous search found no candidate faster than the reference"
        ),
        metrics={
            "candidates_searched": len(candidates),
            "faster_candidates": len(searched),
            "failures": len(outcome.get("failures") or []),
        },
        **outcome,
    )


def _isolated_validate(run_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    search = _prior_result(run_dir, "search")
    candidates = search.get("candidates")
    if not isinstance(candidates, list):
        raise ProductionAdapterError("search result candidates must be a list")
    outcome = validate_candidates(run_dir, candidates, config)
    validated = outcome["candidates"]
    speedups = [
        float(item["validation"]["speedup"])
        for item in validated
        if isinstance(item, Mapping)
        and isinstance(item.get("validation"), Mapping)
    ]
    return _result(
        "isolated_validate",
        (
            f"independently validated {len(validated)} faster candidate(s)"
            if validated
            else "no searched candidate passed isolated correctness and speed gates"
        ),
        metrics={
            "isolated_correct": bool(validated),
            "isolated_speedup": max(speedups, default=0.0),
            "candidates_validated": len(validated),
            "failures": len(outcome.get("failures") or []),
        },
        **outcome,
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


def _selected_artifact_ids(path: Path, payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Artifact ids FastVideo actually dispatched, or fail closed.

    ``reason_counts`` says *how many* selections happened and ``decisions``
    says *which* artifacts they were. The two must agree exactly: a count with
    no matching decision record (or the reverse) means the diagnostics cannot
    identify what ran, and finalizing on that would promote a bundle that was
    never exercised.
    """
    dispatch = payload.get("dispatch")
    if not isinstance(dispatch, Mapping):
        raise ProductionAdapterError(
            f"dispatch diagnostics {path} have no dispatch object"
        )
    counts = dispatch.get("reason_counts")
    if not isinstance(counts, Mapping):
        raise ProductionAdapterError(
            f"dispatch diagnostics {path} have no reason_counts object"
        )
    expected = counts.get("artifact_selected", 0)
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
        raise ProductionAdapterError(
            "dispatch reason_counts.artifact_selected must be a non-negative integer"
        )
    decisions = payload.get("decisions")
    if decisions is None:
        if expected:
            raise ProductionAdapterError(
                f"dispatch diagnostics {path} report {expected} selection(s) but "
                "carry no decisions list naming the selected artifact(s)"
            )
        return ()
    if not isinstance(decisions, list):
        raise ProductionAdapterError("dispatch decisions must be a list")

    selected: list[str] = []
    matched = 0
    for index, decision in enumerate(decisions):
        if not isinstance(decision, Mapping):
            raise ProductionAdapterError(
                f"dispatch decisions[{index}] must be an object"
            )
        if decision.get("reason") != "artifact_selected":
            continue
        matched += 1
        artifact_id = decision.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ProductionAdapterError(
                f"dispatch decisions[{index}] selected an artifact without an id"
            )
        if artifact_id not in selected:
            selected.append(artifact_id)
    if matched != expected:
        raise ProductionAdapterError(
            f"dispatch diagnostics {path} are ambiguous: reason_counts report "
            f"{expected} selection(s) but decisions contain {matched}"
        )
    return tuple(selected)


def _speedup_threshold(config: Mapping[str, Any], workload: Any) -> float:
    """The stricter of the campaign and workload end-to-end speedup gates."""
    configured = _finite(config.get("min_e2e_speedup", 1.01), "min_e2e_speedup")
    declared = (
        workload.performance.min_end_to_end_speedup
        if getattr(workload, "performance", None) is not None
        else 1.01
    )
    return max(configured, float(declared))


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
    threshold = _speedup_threshold(config, workload)
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


def _packaged_bundles(packaged: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    artifacts = packaged.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ProductionAdapterError("package result has no artifacts object")
    root = artifacts.get("root")
    bundles = artifacts.get("bundles")
    if not isinstance(root, str) or not root:
        raise ProductionAdapterError("package result has no artifact root")
    if not isinstance(bundles, list) or not bundles:
        raise ProductionAdapterError("package result has no packaged bundles")
    result: list[dict[str, Any]] = []
    for index, bundle in enumerate(bundles):
        if not isinstance(bundle, Mapping):
            raise ProductionAdapterError(f"package bundles[{index}] must be an object")
        artifact_id = bundle.get("artifact_id")
        bundle_dir = bundle.get("bundle_dir")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ProductionAdapterError(f"package bundles[{index}] has no artifact_id")
        if not isinstance(bundle_dir, str) or not bundle_dir:
            raise ProductionAdapterError(f"package bundles[{index}] has no bundle_dir")
        result.append({"artifact_id": artifact_id, "bundle_dir": bundle_dir})
    return root, result


def _finalize(run_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    packaged = _prior_result(run_dir, "package")
    validated = _prior_result(run_dir, "end_to_end_validate")
    root, bundles = _packaged_bundles(packaged)

    artifacts = validated.get("artifacts")
    if not isinstance(artifacts, Mapping) or not isinstance(
        artifacts.get("dispatch_diagnostics"), str
    ):
        raise ProductionAdapterError(
            "end_to_end_validate result has no dispatch_diagnostics path"
        )
    diagnostics_path = Path(artifacts["dispatch_diagnostics"])
    _, diagnostics = _dispatch_selected(diagnostics_path)
    selected_ids = _selected_artifact_ids(diagnostics_path, diagnostics)

    metrics = validated.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ProductionAdapterError("end_to_end_validate result has no metrics object")
    parity = validated.get("parity")
    parity_policy = ""
    if isinstance(parity, Mapping) and isinstance(parity.get("policy"), str):
        parity_policy = parity["policy"]

    workload = load_workload(_config_path(config, "workload"))
    speedup = metrics.get("end_to_end_speedup")
    outcome = GenerationOutcome(
        workload_id=workload.workload_id,
        steps=workload.sampling.num_inference_steps,
        parity_passed=metrics.get("parity_passed") is True,
        artifact_selected=(
            bool(selected_ids) and metrics.get("artifact_selected") is True
        ),
        classification=str(metrics.get("classification") or ""),
        min_speedup=_speedup_threshold(config, workload),
        speedup=(
            float(speedup)
            if isinstance(speedup, (int, float))
            and not isinstance(speedup, bool)
            and math.isfinite(float(speedup))
            else None
        ),
        # Whether the stage *ran*, not whether its verdict passed. Collapsing
        # a failing verdict into "failed" here made every quarantine report
        # "the end-to-end validation stage did not complete", which in r4 was
        # untrue -- the stage completed and returned a definite negative -- and
        # suppressed the specific reason. parity_passed, artifact_selected,
        # classification and speedup are all populated above; decide() reads
        # them and names the actual cause.
        stage_status="ok" if validated.get("status") == "ok" else "failed",
        parity_policy=parity_policy,
        baseline_ref=str(artifacts.get("native_result") or ""),
        candidate_ref=str(artifacts.get("candidate_result") or ""),
    )

    known = {bundle["artifact_id"] for bundle in bundles}
    missing = sorted(set(selected_ids) - known)
    if missing:
        raise ProductionAdapterError(
            f"dispatch selected artifact(s) {missing} that the package stage did "
            "not produce"
        )

    finalized: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for bundle in bundles:
        artifact_id = bundle["artifact_id"]
        bundle_dir = Path(bundle["bundle_dir"])
        if artifact_id not in selected_ids:
            # Never exercised by this run: verify it is intact and leave the
            # packaged quarantine decision exactly as it is.
            manifest = verify_bundle(bundle_dir)
            quarantined.append(
                {
                    "artifact_id": manifest.artifact_id,
                    "bundle_dir": str(bundle_dir),
                    "manifest": str(bundle_dir / "artifact.json"),
                    "decision": manifest.promotion.decision,
                    "reason": "artifact was not selected during full generation",
                    "changed": False,
                }
            )
            continue
        result = finalize_bundle(bundle_dir, outcome)
        (finalized if result.decision != "quarantined" else quarantined).append(
            result.as_dict()
        )

    promoted = [item for item in finalized if item["decision"] == "promoted"]
    rejected = [item for item in finalized if item["decision"] == "rejected"]
    if promoted:
        recommendation = "promoted"
        message = f"finalized {len(promoted)} promoted artifact bundle(s)"
    elif rejected:
        recommendation = "no_worthwhile_candidate"
        message = f"rejected {len(rejected)} measured artifact bundle(s)"
    else:
        recommendation = "failed"
        message = "no artifact could be finalized; every bundle remains quarantined"
    return _result(
        "finalize",
        message,
        metrics={
            "artifacts_promoted": len(promoted),
            "artifacts_rejected": len(rejected),
            "artifacts_quarantined": len(quarantined),
            "artifacts_selected": len(selected_ids),
        },
        recommendation=recommendation,
        artifacts={
            "root": root,
            "finalized": finalized,
            "quarantined": quarantined,
        },
        decisions=[
            {"artifact_id": item["artifact_id"], "decision": item["decision"]}
            for item in [*finalized, *quarantined]
        ],
    )


_ADAPTERS: dict[str, Callable[[Path, Mapping[str, Any]], dict[str, Any]]] = {
    "baseline": _baseline,
    "profile": _profile,
    "discover": _discover,
    "specgen": _specgen,
    "search": _search,
    "isolated_validate": _isolated_validate,
    "package": _package,
    "end_to_end_validate": _end_to_end_validate,
    "finalize": _finalize,
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
    except (
        ArtifactError,
        BuiltinSearchError,
        DiscoveryError,
        SpecGenerationError,
        WorkloadError,
    ) as exc:
        raise ProductionAdapterError(str(exc)) from exc
