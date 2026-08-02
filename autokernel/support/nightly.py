"""Nightly discovery-only campaigns that feed the support matrix.

One night = one immutable directory per workload under the nightly root, a
discovery-only campaign (baseline -> profile -> discover, then stop), and one
distilled run record per workload committed under the records directory.

Discipline, per the track-E brief:

- **Idempotent and resumable.** Campaigns resume through the optimize
  control plane; a rerun of a finished night returns the terminal receipt
  instead of re-running stages.
- **Immutable evidence directories.** Each night writes a fresh
  ``<root>/<date>/<workload_id>``; no night's directory is ever rewritten.
- **A failure to capture is data.** A failed campaign becomes a
  ``capture_blocked`` record carrying the receipt's real message; a
  budget-exhausted (preempted) campaign writes no record, because it
  produced no verdict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from autokernel._io import write_json_atomic
from autokernel.optimize import OptimizeConfig, OptimizeError, run_optimize
from autokernel.workload import load_workload

from .evidence import (
    RunRecord,
    SupportEvidenceError,
    record_filename,
    record_from_receipt,
    write_run_record,
)


class NightlyError(RuntimeError):
    """The nightly could not start (bad inputs, not a campaign failure)."""


@dataclass(frozen=True)
class NightlyTarget:
    """One workload to discover on one architecture."""

    workload: Path
    model_override: str | None = None


@dataclass(frozen=True)
class NightlyConfig:
    fastvideo_checkout: Path
    targets: tuple[NightlyTarget, ...]
    nightly_root: Path
    records_dir: Path
    arch: str
    budget_hours: float = 2.0
    date: str | None = None
    repo_root: Path | None = None


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _family_of(workload: Path) -> str:
    manifest = load_workload(workload)
    return manifest.tags[0] if manifest.tags else manifest.workload_id


def run_nightly_target(
    *,
    target: NightlyTarget,
    config: NightlyConfig,
    night_dir: Path,
) -> dict[str, Any]:
    """Run one discovery-only campaign and distill its run record."""
    manifest = load_workload(target.workload)
    campaign_dir = night_dir / manifest.workload_id
    model = target.model_override or manifest.model.model_id
    receipt = run_optimize(
        OptimizeConfig(
            fastvideo_checkout=config.fastvideo_checkout,
            model=model,
            workload=target.workload,
            output=campaign_dir,
            budget_hours=config.budget_hours,
            resume=True,
            repo_root=config.repo_root,
            stop_after_stage="discover",
        )
    )
    record = record_from_receipt(
        receipt,
        workload_id=manifest.workload_id,
        model_id=model,
        family=_family_of(target.workload),
        arch=config.arch,
        evidence=str(campaign_dir),
        recorded_utc=datetime.now(timezone.utc).isoformat(),
    )
    written: str | None = None
    if record is not None:
        config.records_dir.mkdir(parents=True, exist_ok=True)
        path = config.records_dir / record_filename(manifest.workload_id, config.arch)
        write_run_record(record, path)
        written = str(path)
    return {
        "workload_id": manifest.workload_id,
        "model_id": model,
        "terminal": receipt.get("terminal"),
        "record_written": written,
        "campaign_dir": str(campaign_dir),
    }


def run_nightly(config: NightlyConfig) -> dict[str, Any]:
    """Run all targets for one night; write one report per night root."""
    if not config.targets:
        raise NightlyError("nightly requires at least one workload target")
    if not config.arch.startswith("sm"):
        raise NightlyError(f"arch must look like sm100 / sm90; got {config.arch!r}")
    night = config.date or _today()
    night_dir = Path(config.nightly_root) / night
    night_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for target in config.targets:
        try:
            results.append(
                run_nightly_target(target=target, config=config, night_dir=night_dir)
            )
        except (OptimizeError, SupportEvidenceError) as exc:
            # A campaign that cannot run at all still produces data: the
            # capture_blocked record carries the real reason.
            manifest = load_workload(target.workload)
            model = target.model_override or manifest.model.model_id
            record = RunRecord.from_dict(
                {
                    "schema": "motionkernel.support-run",
                    "schema_version": 1,
                    "workload_id": manifest.workload_id,
                    "model_id": model,
                    "family": _family_of(target.workload),
                    "arch": config.arch,
                    "outcome": "capture_blocked",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "recorded_utc": datetime.now(timezone.utc).isoformat(),
                    "evidence": str(night_dir / manifest.workload_id),
                }
            )
            config.records_dir.mkdir(parents=True, exist_ok=True)
            path = config.records_dir / record_filename(manifest.workload_id, config.arch)
            write_run_record(record, path)
            results.append(
                {
                    "workload_id": manifest.workload_id,
                    "model_id": model,
                    "terminal": "failed",
                    "record_written": str(path),
                    "campaign_dir": str(night_dir / manifest.workload_id),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    report = {
        "schema": "motionkernel.support-nightly",
        "schema_version": 1,
        "date": night,
        "arch": config.arch,
        "nightly_root": str(config.nightly_root),
        "records_dir": str(config.records_dir),
        "results": results,
    }
    write_json_atomic(night_dir / "nightly_report.json", report)
    return report
