"""Built-in autonomous search and independent isolated validation.

The search agent may edit only a generated candidate's ``kernel.py``. The
validator then runs the fixed MotionKernel harness in a separate process and
derives the artifact request exclusively from measured JSON and the validated
graph manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autokernel.specgen import (
    build_dispatch_contract,
    spec_from_manifest,
    write_runtime_adapter,
)
from autokernel.workload import load_workload


class BuiltinSearchError(RuntimeError):
    """Search or isolated validation could not satisfy its contract."""


def _remaining(deadline: float | None) -> float | None:
    return None if deadline is None else deadline - time.monotonic()


def _generated(candidate: Mapping[str, Any]) -> dict[str, Path]:
    raw = candidate.get("generated")
    if not isinstance(raw, Mapping):
        raise BuiltinSearchError("specified candidate has no generated artifacts")
    required = ("manifest", "spec", "kernel", "corpus")
    paths: dict[str, Path] = {}
    for name in required:
        value = raw.get(name)
        if not isinstance(value, str) or not value:
            raise BuiltinSearchError(f"generated candidate has no {name!r} path")
        path = Path(value).resolve()
        if not path.is_file():
            raise BuiltinSearchError(f"generated candidate file is missing: {path}")
        paths[name] = path
    return paths


def _benchmark_command(
    repo_root: Path,
    generated: Mapping[str, Path],
    result_path: Path,
    *,
    baseline: str,
    quick: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(repo_root / "bench.py"),
        "--spec",
        f"{generated['spec']}:SPEC",
        "--shape-corpus",
        str(generated["corpus"]),
        "--shape-corpus-only",
        "--baseline",
        baseline,
        "--result-json",
        str(result_path),
    ]
    if quick:
        command.append("--quick")
    return command


def _run_benchmark(
    repo_root: Path,
    generated: Mapping[str, Path],
    result_path: Path,
    log_path: Path,
    *,
    baseline: str,
    quick: bool,
    timeout: float | None = None,
) -> dict[str, Any]:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    command = _benchmark_command(
        repo_root,
        generated,
        result_path,
        baseline=baseline,
        quick=quick,
    )
    try:
        completed = subprocess.run(
            command,
            cwd=generated["kernel"].parent,
            text=True,
            capture_output=True,
            check=False,
            timeout=max(timeout, 0.001) if timeout is not None else None,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        log_path.write_text(stdout + stderr, encoding="utf-8")
        raise BuiltinSearchError(
            "per-candidate budget exhausted during benchmark"
        ) from exc
    log_path.write_text(
        (completed.stdout or "") + (completed.stderr or ""), encoding="utf-8"
    )
    if completed.returncode != 0 or not result_path.is_file():
        raise BuiltinSearchError(
            f"benchmark failed with exit {completed.returncode}; see {log_path}"
        )
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BuiltinSearchError(f"invalid benchmark result {result_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise BuiltinSearchError("benchmark result must use schema_version 2")
    return payload


def _primary(payload: Mapping[str, Any]) -> dict[str, Any]:
    performance = payload.get("performance")
    primary = performance.get("primary") if isinstance(performance, Mapping) else None
    if not isinstance(primary, dict):
        raise BuiltinSearchError("benchmark produced no primary measurement")
    for name in ("pytorch_latency_us", "kernel_latency_us", "speedup_vs_pytorch"):
        value = primary.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise BuiltinSearchError(f"benchmark primary {name} is missing")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise BuiltinSearchError(f"benchmark primary {name} must be positive")
    return primary


def _agent_prompt(
    candidate: Mapping[str, Any],
    generated: Mapping[str, Path],
    benchmark: Sequence[str],
) -> str:
    return f"""\
Optimize one graph-derived CUDA kernel autonomously. Do not ask questions.

Target fingerprint: {candidate.get('fingerprint')}
Measured optimistic model impact: {candidate.get('estimated_max_e2e_improvement_pct')}%
Editable file: {generated['kernel']}
Read-only semantic inputs: {generated['manifest']}, {generated['spec']}, {generated['corpus']}

Run this exact benchmark after every meaningful experiment:
{' '.join(benchmark)}

You may change only kernel.py in the candidate directory. Never edit the
manifest, spec, corpus, benchmark harness, references, tolerances, or verifier.
Keep only changes that pass correctness and improve speedup_vs_pytorch. Explore
Triton, torch.compile, and fused PyTorch implementations as appropriate for the
captured operations and observed shapes. Leave the fastest passing candidate in
kernel.py. If no implementation beats the reference, restore the best passing
version and say so in the final message.
"""


def _agent_command(
    configured: Sequence[str] | None,
    *,
    repo_root: Path,
    run_dir: Path,
    candidate_dir: Path,
    prompt_path: Path,
    last_message: Path,
) -> list[str]:
    replacements = {
        "{repo_root}": str(repo_root),
        "{run_dir}": str(run_dir),
        "{candidate_dir}": str(candidate_dir),
        "{prompt_file}": str(prompt_path),
        "{last_message}": str(last_message),
    }
    if configured:
        command = []
        for raw in configured:
            value = str(raw)
            for placeholder, replacement in replacements.items():
                value = value.replace(placeholder, replacement)
            command.append(value)
        return command

    executable = shutil.which("codex")
    if executable is None:
        raise BuiltinSearchError(
            "autonomous search requires the Codex CLI or "
            "--search-agent-command"
        )
    return [
        executable,
        "exec",
        "-C",
        str(candidate_dir),
        "-s",
        "workspace-write",
        "--output-last-message",
        str(last_message),
        prompt_path.read_text(encoding="utf-8"),
    ]


def search_candidates(
    run_dir: Path,
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the configured coding agent, then measure every resulting kernel."""
    repo_root = Path(str(config.get("repo_root") or Path(__file__).parents[2])).resolve()
    baseline = str(config.get("baseline") or "eager")
    configured = config.get("search_agent_command")
    if configured is not None and not isinstance(configured, list):
        raise BuiltinSearchError("search_agent_command must be an argv list")
    budget_value = config.get("per_candidate_budget_seconds")
    per_candidate_budget = (
        float(budget_value) if budget_value is not None else None
    )

    stage_dir = run_dir / "stages" / "search"
    searched: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    measured_count = 0
    for candidate in candidates:
        deadline = (
            time.monotonic() + per_candidate_budget
            if per_candidate_budget is not None
            else None
        )

        generated = _generated(candidate)
        fingerprint = str(candidate.get("fingerprint") or "unknown")
        work = stage_dir / fingerprint
        work.mkdir(parents=True, exist_ok=True)
        quick_result = work / "search_benchmark.json"
        benchmark = _benchmark_command(
            repo_root,
            generated,
            quick_result,
            baseline=baseline,
            quick=True,
        )
        prompt_path = work / "prompt.md"
        prompt_path.write_text(
            _agent_prompt(candidate, generated, benchmark), encoding="utf-8"
        )
        last_message = work / "agent_last_message.md"
        log_path = work / "agent.log"
        try:
            command = _agent_command(
                configured,
                repo_root=repo_root,
                run_dir=run_dir,
                candidate_dir=generated["kernel"].parent,
                prompt_path=prompt_path,
                last_message=last_message,
            )
            try:
                completed = subprocess.run(
                    command,
                    cwd=repo_root,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=(
                        max(_remaining(deadline), 0.001)
                        if _remaining(deadline) is not None
                        else None
                    ),
                )
            except subprocess.TimeoutExpired as exc:
                stdout = (
                    exc.stdout.decode(errors="replace")
                    if isinstance(exc.stdout, bytes)
                    else (exc.stdout or "")
                )
                stderr = (
                    exc.stderr.decode(errors="replace")
                    if isinstance(exc.stderr, bytes)
                    else (exc.stderr or "")
                )
                log_path.write_text(stdout + stderr, encoding="utf-8")
                raise BuiltinSearchError(
                    "per-candidate budget exhausted during agent search"
                ) from exc
            log_path.write_text(
                (completed.stdout or "") + (completed.stderr or ""),
                encoding="utf-8",
            )
            if completed.returncode != 0:
                raise BuiltinSearchError(
                    f"search agent exited {completed.returncode}; see {log_path}"
                )
            payload = _run_benchmark(
                repo_root,
                generated,
                quick_result,
                work / "benchmark.log",
                baseline=baseline,
                quick=True,
                timeout=_remaining(deadline),
            )
            measured_count += 1
            primary = _primary(payload)
            forward = payload.get("forward")
            correct = (
                isinstance(forward, Mapping)
                and forward.get("correctness") == "PASS"
            )
            speedup = float(primary["speedup_vs_pytorch"])
            if not correct or speedup <= 1.0:
                failures.append(
                    {
                        "fingerprint": fingerprint,
                        "reason": "candidate did not beat the isolated reference",
                    }
                )
                continue
            searched.append(
                {
                    **dict(candidate),
                    "search": {
                        "kernel": str(generated["kernel"]),
                        "result": str(quick_result),
                        "speedup": speedup,
                        "agent_log": str(log_path),
                        "agent_summary": str(last_message),
                    },
                }
            )
        except BuiltinSearchError as exc:
            failures.append({"fingerprint": fingerprint, "reason": str(exc)})

    if not searched and not measured_count and failures:
        summary = "; ".join(item["reason"] for item in failures[:3])
        raise BuiltinSearchError(f"autonomous search produced no measurement: {summary}")
    if not searched:
        return {
            "candidates": [],
            "failures": failures,
            "recommendation": "no_worthwhile_candidate",
        }
    return {"candidates": searched, "failures": failures}


def _gpu_architecture(payload: Mapping[str, Any]) -> str:
    gpu = payload.get("gpu")
    capability = gpu.get("compute_capability") if isinstance(gpu, Mapping) else None
    if (
        not isinstance(capability, list | tuple)
        or len(capability) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in capability)
    ):
        raise BuiltinSearchError("benchmark result has no GPU compute capability")
    major, minor = capability
    if major <= 0 or minor < 0:
        raise BuiltinSearchError("isolated validation requires a CUDA GPU")
    return f"sm{major}{minor}"


def _error_maxima(forward: Mapping[str, Any]) -> tuple[float, float]:
    absolute: list[float] = []
    relative: list[float] = []
    details = forward.get("leaf_details")
    if isinstance(details, list):
        for item in details:
            if not isinstance(item, Mapping):
                continue
            for name, output in (
                ("max_abs_error", absolute),
                ("max_rel_error", relative),
            ):
                value = item.get(name)
                if isinstance(value, int | float) and not isinstance(value, bool):
                    number = float(value)
                    if math.isfinite(number) and number >= 0:
                        output.append(number)
    return max(absolute, default=0.0), max(relative, default=0.0)


def validate_candidates(
    run_dir: Path,
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently validate searched kernels and build package requests."""
    repo_root = Path(str(config.get("repo_root") or Path(__file__).parents[2])).resolve()
    baseline = str(config.get("baseline") or "eager")
    workload = load_workload(Path(str(config["workload"])))
    budget_value = config.get("per_candidate_budget_seconds")
    per_candidate_budget = (
        float(budget_value) if budget_value is not None else None
    )
    stage_dir = run_dir / "stages" / "isolated_validate"
    validated: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    measured_count = 0

    for candidate in candidates:
        generated = _generated(candidate)
        fingerprint = str(candidate.get("fingerprint") or "unknown")
        work = stage_dir / fingerprint
        work.mkdir(parents=True, exist_ok=True)
        result_path = work / "benchmark.json"
        try:
            payload = _run_benchmark(
                repo_root,
                generated,
                result_path,
                work / "benchmark.log",
                baseline=baseline,
                quick=False,
                timeout=per_candidate_budget,
            )
            measured_count += 1
            forward = payload.get("forward")
            if not isinstance(forward, Mapping) or forward.get("correctness") != "PASS":
                raise BuiltinSearchError("candidate failed full isolated correctness")
            primary = _primary(payload)
            speedup = float(primary["speedup_vs_pytorch"])
            if speedup <= 1.0:
                raise BuiltinSearchError("candidate did not beat the isolated reference")
            architecture = _gpu_architecture(payload)
            manifest_value = json.loads(
                generated["manifest"].read_text(encoding="utf-8")
            )
            contract = build_dispatch_contract(manifest_value)
            spec = spec_from_manifest(generated["manifest"])
            tolerance = spec.tolerance_for(spec.dtypes[0])
            max_abs, max_rel = _error_maxima(forward)

            kernel_digest = hashlib.sha256(generated["kernel"].read_bytes()).hexdigest()
            artifact_id = (
                f"mk-{fingerprint[:16]}-{kernel_digest[:8]}-{architecture}"
            )
            payload_dir = work / "payload"
            payload_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(generated["kernel"], payload_dir / "candidate.py")
            shutil.copyfile(generated["manifest"], payload_dir / "manifest.json")
            write_runtime_adapter(
                manifest_value,
                payload_dir / "entry.py",
                candidate_file="candidate.py",
            )

            now = datetime.now(timezone.utc).isoformat()
            sections = {
                "artifact_id": artifact_id,
                **contract,
                "entry_point": {"file": "entry.py", "symbol": "fused_subgraph"},
                "compatibility": {
                    "model_id": str(config["model"]),
                    "model_revision": getattr(workload.model, "revision", None) or "*",
                    "gpu_architectures": [architecture],
                    "torch": {},
                    "cuda": {},
                    "triton": {},
                    "execution_modes": ["inference"],
                    "distributed_modes": ["single"],
                },
                "evidence": {
                    "benchmark": {
                        "harness": "motionkernel-bench",
                        "device": "cuda:0",
                        "samples": 100,
                        "baseline_us": float(primary["pytorch_latency_us"]),
                        "candidate_us": float(primary["kernel_latency_us"]),
                        "speedup": speedup,
                        "max_abs_error": max_abs,
                        "max_rel_error": max_rel,
                        "atol": tolerance.atol,
                        "rtol": tolerance.rtol,
                        "passed": True,
                        "result_ref": str(result_path),
                    },
                    "generation": {
                        "workload_id": workload.workload_id,
                        "steps": workload.sampling.num_inference_steps,
                        "metric": "pending_full_generation_validation",
                        "value": 0.0,
                        "threshold": 0.0,
                        "passed": False,
                        "baseline_ref": str(
                            run_dir / "generation" / "native_result.json"
                        ),
                        "candidate_ref": str(
                            run_dir / "generation" / "candidate_result.json"
                        ),
                    },
                },
                "promotion": {
                    "decision": "quarantined",
                    "reason": "awaiting full-generation A/B validation",
                    "decided_at": now,
                    "campaign": {
                        "campaign_id": run_dir.name or "motionkernel-optimize",
                        "source": "motionkernel-optimize",
                        "target_name": str(manifest_value.get("name") or fingerprint),
                    },
                },
            }
            validated.append(
                {
                    **dict(candidate),
                    "validation": {
                        "result": str(result_path),
                        "speedup": speedup,
                        "artifact_id": artifact_id,
                    },
                }
            )
            requests.append({"source_dir": str(payload_dir), "sections": sections})
        except (BuiltinSearchError, OSError, json.JSONDecodeError) as exc:
            failures.append({"fingerprint": fingerprint, "reason": str(exc)})

    if not validated and not measured_count and failures:
        summary = "; ".join(item["reason"] for item in failures[:3])
        raise BuiltinSearchError(
            f"isolated validation produced no measurement: {summary}"
        )
    if not validated:
        return {
            "candidates": [],
            "package_requests": [],
            "failures": failures,
            "recommendation": "no_worthwhile_candidate",
        }
    return {
        "candidates": validated,
        "package_requests": requests,
        "failures": failures,
    }
