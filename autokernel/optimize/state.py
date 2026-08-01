"""Durable campaign state, atomic JSON, and terminal receipts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autokernel._io import write_json_atomic

from .types import (
    CAMPAIGN_STATE_SCHEMA_VERSION,
    PIPELINE_STAGES,
    RECEIPT_SCHEMA_VERSION,
    OptimizeConfig,
    StageRecord,
)


class OptimizeError(ValueError):
    """Raised when the optimize control plane cannot proceed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_dir_layout(output: Path) -> dict[str, Path]:
    """Standard run directory paths."""
    root = output
    return {
        "root": root,
        "config": root / "config.json",
        "preflight": root / "preflight.json",
        "run_contract": root / "run_contract.json",
        "state": root / "state.json",
        "receipt": root / "receipt.json",
        "morning_report": root / "morning_report.md",
        "stages": root / "stages",
        "logs": root / "logs",
        "candidates": root / "candidates",
        "artifacts": root / "artifacts",
        "commands": root / "commands",
    }


def initial_state(config: OptimizeConfig) -> dict[str, Any]:
    return {
        "schema_version": CAMPAIGN_STATE_SCHEMA_VERSION,
        "status": "running",
        "terminal": None,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "budget_hours": config.budget_hours,
        "budget_deadline_epoch": None,
        "completed_stages": [],
        "failed_stages": {},
        "stage_records": {},
        "candidates": [],
        "active_candidate": None,
        "baseline_mode": config.baseline,
        "min_e2e_speedup": config.min_e2e_speedup,
        "messages": [],
    }


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise OptimizeError(f"campaign state not found: {path}")
    try:
        state = read_json(path)
    except json.JSONDecodeError as exc:
        raise OptimizeError(f"campaign state corrupt: {exc}") from exc
    if not isinstance(state, dict):
        raise OptimizeError("campaign state must be a JSON object")
    if state.get("schema_version") != CAMPAIGN_STATE_SCHEMA_VERSION:
        raise OptimizeError(
            f"unsupported campaign state schema_version "
            f"{state.get('schema_version')!r}"
        )
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    state = dict(state)
    state["updated_at"] = utc_now()
    write_json_atomic(path, state)


def stage_is_complete(state: Mapping[str, Any], stage: str) -> bool:
    completed = state.get("completed_stages") or []
    if stage not in completed:
        return False
    records = state.get("stage_records") or {}
    record = records.get(stage) or {}
    return record.get("status") == "ok"


def mark_stage(
    state: dict[str, Any],
    record: StageRecord,
) -> None:
    records = dict(state.get("stage_records") or {})
    records[record.name] = record.as_dict()
    state["stage_records"] = records
    completed = list(state.get("completed_stages") or [])
    failed = dict(state.get("failed_stages") or {})
    if record.status == "ok":
        if record.name not in completed:
            completed.append(record.name)
        failed.pop(record.name, None)
    elif record.status == "failed":
        failed[record.name] = record.message or "failed"
        if record.name in completed:
            completed = [s for s in completed if s != record.name]
    state["completed_stages"] = completed
    state["failed_stages"] = failed


def budget_remaining_seconds(state: Mapping[str, Any]) -> float | None:
    deadline = state.get("budget_deadline_epoch")
    if deadline is None:
        return None
    import time

    return float(deadline) - time.time()


def ensure_budget(state: Mapping[str, Any]) -> None:
    remaining = budget_remaining_seconds(state)
    if remaining is not None and remaining <= 0:
        raise OptimizeError("campaign budget exhausted")


def build_receipt(
    state: Mapping[str, Any],
    config: OptimizeConfig,
    *,
    terminal: str,
    message: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "terminal": terminal,
        "status": terminal,
        "message": message,
        "started_at": state.get("started_at"),
        "finished_at": utc_now(),
        "budget_hours": config.budget_hours,
        "model": config.model,
        "workload": str(config.workload),
        "fastvideo_checkout": str(config.fastvideo_checkout),
        "output": str(config.output),
        "baseline_mode": config.baseline,
        "min_e2e_speedup": config.min_e2e_speedup,
        "completed_stages": list(state.get("completed_stages") or []),
        "failed_stages": dict(state.get("failed_stages") or {}),
        "stage_records": dict(state.get("stage_records") or {}),
        "candidates": list(state.get("candidates") or []),
        "pipeline_stages": list(PIPELINE_STAGES),
    }
