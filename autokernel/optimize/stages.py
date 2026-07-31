"""Stage subprocess execution with JSON/file contracts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .state import (
    OptimizeError,
    budget_remaining_seconds,
    ensure_budget,
    mark_stage,
    run_dir_layout,
    save_state,
    stage_is_complete,
    utc_now,
    write_json_atomic,
)
from .types import (
    CANDIDATE_STAGE_STATUS,
    PIPELINE_STAGES,
    STAGE_RESULT_SCHEMA_VERSION,
    OptimizeConfig,
    StageRecord,
)


def stage_dir(run_root: Path, stage: str) -> Path:
    return run_root / "stages" / stage


def default_stage_command(
    stage: str,
    *,
    config: OptimizeConfig,
    run_root: Path,
) -> list[str]:
    """Built-in driver invoked as an isolated subprocess."""
    repo = config.repo_root or Path(__file__).resolve().parents[2]
    return [
        sys.executable,
        "-m",
        "autokernel.optimize.stage_driver",
        "--stage",
        stage,
        "--run-dir",
        str(run_root),
        "--repo-root",
        str(repo),
    ]


def resolve_stage_command(
    stage: str,
    *,
    config: OptimizeConfig,
    run_root: Path,
) -> list[str]:
    if config.stage_commands and stage in config.stage_commands:
        replacements = {
            "{stage}": stage,
            "{run_dir}": str(run_root),
            "{repo_root}": str(config.repo_root or Path(__file__).resolve().parents[2]),
            "{fastvideo_checkout}": str(config.fastvideo_checkout),
            "{workload}": str(config.workload),
            "{model}": config.model,
            "{baseline}": config.baseline,
            "{artifact_dir}": str(run_root / config.artifact_dir_name),
        }
        command: list[str] = []
        for raw_part in config.stage_commands[stage]:
            part = str(raw_part)
            for placeholder, value in replacements.items():
                part = part.replace(placeholder, value)
            command.append(part)
        return command
    return default_stage_command(stage, config=config, run_root=run_root)


def _load_stage_result(path: Path, *, expected_stage: str) -> dict[str, Any]:
    if not path.is_file():
        raise OptimizeError(f"stage result missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OptimizeError(f"invalid stage result JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OptimizeError(f"stage result must be an object: {path}")
    if payload.get("schema_version") != STAGE_RESULT_SCHEMA_VERSION:
        raise OptimizeError(
            f"unsupported stage result schema_version "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("stage") != expected_stage:
        raise OptimizeError(
            f"stage result identity mismatch: expected {expected_stage!r}, "
            f"got {payload.get('stage')!r}"
        )
    if payload.get("status") not in {"ok", "skipped", "failed"}:
        raise OptimizeError(
            f"invalid stage result status {payload.get('status')!r}: {path}"
        )
    if "metrics" in payload and not isinstance(payload["metrics"], dict):
        raise OptimizeError(f"stage result metrics must be an object: {path}")
    if "candidates" in payload and not isinstance(payload["candidates"], list):
        raise OptimizeError(f"stage result candidates must be a list: {path}")
    return payload


def _advance_candidates(
    state: dict[str, Any], stage: str, payload: Mapping[str, Any]
) -> None:
    """Preserve candidates across stages and record their durable lifecycle."""
    if "candidates" in payload:
        candidates = list(payload.get("candidates") or [])
    else:
        candidates = list(state.get("candidates") or [])
    candidate_status = CANDIDATE_STAGE_STATUS.get(stage)
    if candidate_status:
        candidates = [
            {**dict(candidate), "status": candidate_status}
            if isinstance(candidate, Mapping)
            else candidate
            for candidate in candidates
        ]
    state["candidates"] = candidates


def run_stage(
    stage: str,
    *,
    config: OptimizeConfig,
    state: dict[str, Any],
    state_path: Path,
) -> dict[str, Any]:
    """Run one pipeline stage as a subprocess unless already complete."""
    if stage not in PIPELINE_STAGES:
        raise OptimizeError(f"unknown stage {stage!r}")

    layout = run_dir_layout(config.output)
    sdir = stage_dir(layout["root"], stage)
    sdir.mkdir(parents=True, exist_ok=True)
    result_path = sdir / "result.json"
    stdout_path = sdir / "stdout.log"
    stderr_path = sdir / "stderr.log"
    command_path = layout["commands"] / f"{stage}.json"
    layout["commands"].mkdir(parents=True, exist_ok=True)

    if config.resume and stage_is_complete(state, stage):
        if result_path.is_file():
            return _load_stage_result(result_path, expected_stage=stage)
        # Completed flag without result — re-run.
        completed = list(state.get("completed_stages") or [])
        state["completed_stages"] = [s for s in completed if s != stage]

    ensure_budget(state)

    command = resolve_stage_command(stage, config=config, run_root=layout["root"])
    write_json_atomic(
        command_path,
        {
            "stage": stage,
            "command": command,
            "cwd": str(config.repo_root or layout["root"]),
            "recorded_at": utc_now(),
        },
    )
    # Contract inputs for the child.
    write_json_atomic(
        sdir / "input.json",
        {
            "stage": stage,
            "run_dir": str(layout["root"]),
            "config": config.as_dict(),
            "state_snapshot": {
                "completed_stages": list(state.get("completed_stages") or []),
                "candidates": list(state.get("candidates") or []),
            },
        },
    )

    started = utc_now()
    env = os.environ.copy()
    env["MOTIONKERNEL_RUN_DIR"] = str(layout["root"])
    env["MOTIONKERNEL_STAGE"] = stage
    env["MOTIONKERNEL_BASELINE"] = config.baseline
    env["MOTIONKERNEL_FASTVIDEO_CHECKOUT"] = str(config.fastvideo_checkout)
    env["MOTIONKERNEL_WORKLOAD"] = str(config.workload)
    env["MOTIONKERNEL_MODEL"] = config.model
    env["MOTIONKERNEL_ARTIFACT_DIR"] = str(
        layout["root"] / config.artifact_dir_name
    )
    env["PYTHONPATH"] = (
        str(config.repo_root or Path(__file__).resolve().parents[2])
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    started_mono = time.monotonic()
    campaign_remaining = budget_remaining_seconds(state)
    timeout = campaign_remaining
    timeout_source = "campaign"
    if (
        stage in {"search", "isolated_validate"}
        and config.per_candidate_budget_seconds is not None
        and (timeout is None or config.per_candidate_budget_seconds < timeout)
    ):
        timeout = config.per_candidate_budget_seconds
        timeout_source = "per-candidate"
    try:
        completed = subprocess.run(
            command,
            cwd=str(config.repo_root) if config.repo_root else None,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=max(timeout, 0.001) if timeout is not None else None,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        message = f"{timeout_source} budget exhausted while running {stage}"
        record = StageRecord(
            name=stage,
            status="failed",
            started_at=started,
            finished_at=utc_now(),
            exit_code=124,
            message=message,
            metrics={"elapsed_seconds": time.monotonic() - started_mono},
        )
        mark_stage(state, record)
        save_state(state_path, state)
        raise OptimizeError(message) from exc
    except OSError as exc:
        record = StageRecord(
            name=stage,
            status="failed",
            started_at=started,
            finished_at=utc_now(),
            exit_code=127,
            message=f"failed to spawn stage: {exc}",
        )
        mark_stage(state, record)
        save_state(state_path, state)
        raise OptimizeError(record.message) from exc

    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    elapsed = time.monotonic() - started_mono

    if completed.returncode != 0:
        # Allow driver to still write a result.json explaining failure.
        if result_path.is_file():
            try:
                payload = _load_stage_result(result_path, expected_stage=stage)
            except OptimizeError:
                payload = {
                    "schema_version": STAGE_RESULT_SCHEMA_VERSION,
                    "stage": stage,
                    "status": "failed",
                    "message": (
                        f"stage exit {completed.returncode}; "
                        f"stderr={ (completed.stderr or '')[:500] }"
                    ),
                }
                write_json_atomic(result_path, payload)
        else:
            payload = {
                "schema_version": STAGE_RESULT_SCHEMA_VERSION,
                "stage": stage,
                "status": "failed",
                "message": (
                    f"stage exit {completed.returncode}; "
                    f"stderr={ (completed.stderr or '')[:500] }"
                ),
            }
            write_json_atomic(result_path, payload)

        record = StageRecord(
            name=stage,
            status="failed",
            started_at=started,
            finished_at=utc_now(),
            exit_code=completed.returncode,
            result_path=str(result_path),
            message=str(payload.get("message") or "stage failed"),
            metrics={"elapsed_seconds": elapsed},
        )
        mark_stage(state, record)
        save_state(state_path, state)
        raise OptimizeError(record.message)

    payload = _load_stage_result(result_path, expected_stage=stage)
    status = str(payload.get("status") or "ok")
    if status not in {"ok", "skipped"}:
        record = StageRecord(
            name=stage,
            status="failed",
            started_at=started,
            finished_at=utc_now(),
            exit_code=completed.returncode,
            result_path=str(result_path),
            message=str(payload.get("message") or f"stage status {status}"),
            metrics=dict(payload.get("metrics") or {}),
        )
        mark_stage(state, record)
        save_state(state_path, state)
        raise OptimizeError(record.message)

    _advance_candidates(state, stage, payload)

    record = StageRecord(
        name=stage,
        status="ok",
        started_at=started,
        finished_at=utc_now(),
        exit_code=0,
        result_path=str(result_path),
        message=str(payload.get("message") or ""),
        metrics={
            **dict(payload.get("metrics") or {}),
            "elapsed_seconds": elapsed,
        },
    )
    mark_stage(state, record)
    save_state(state_path, state)
    return payload
