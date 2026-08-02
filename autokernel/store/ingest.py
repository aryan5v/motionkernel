"""Readers that turn each evidence shape into an :class:`ExperimentRow`.

One reader per producer. Each is deliberately tolerant: it extracts what it
recognizes and leaves the rest to the preserved ``raw`` column, so a source
schema that gains a field does not break ingest and a source that loses one
degrades to a null rather than an exception.

The tolerance stops at provenance. A document with no identifiable source path
is refused, and a document that matches no reader is reported rather than
silently skipped -- a backfill that quietly ingests half the evidence is worse
than one that fails, because the store then looks complete.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .schema import ExperimentRow

__all__ = ["ingest_path", "read_document", "readers"]

_JOB_RE = re.compile(r"(?:^|[-_/])(\d{3,7})(?:[-_/.]|$)")


def _slurm_ids(*values: Any) -> tuple[str, ...]:
    """Pull SLURM job ids out of paths and free text.

    Job ids are how a published number is traced back to the run that produced
    it, and they are usually embedded in a directory name rather than recorded
    as a field.
    """
    found: list[str] = []
    for value in values:
        if not value:
            continue
        for match in _JOB_RE.finditer(str(value)):
            job = match.group(1)
            if job not in found:
                found.append(job)
    return tuple(found)


def _f(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def read_dispatch_measurement(doc: dict, path: str) -> ExperimentRow | None:
    if doc.get("schema") != "motionkernel.dispatch-measurement":
        return None
    variance = doc.get("variance") or {}
    e2e = doc.get("e2e") or {}
    return ExperimentRow(
        kind="dispatch_measurement",
        source_path=path,
        raw=doc,
        workload_id=doc.get("workload_id"),
        model_id=doc.get("model_id"),
        arch=doc.get("arch"),
        speedup_median=_f(e2e.get("speedup_median")),
        speedup_min_to_min=_f(e2e.get("speedup_min_to_min")),
        native_cv=_f(variance.get("native_cv")),
        candidate_cv=_f(variance.get("candidate_cv")),
        valid_for_gating=variance.get("valid_for_gating"),
        recorded_utc=doc.get("created_utc"),
        slurm_job_ids=_slurm_ids(path, doc.get("artifact_root")),
    )


def read_support_run(doc: dict, path: str) -> ExperimentRow | None:
    if doc.get("schema") != "motionkernel.support-run":
        return None
    return ExperimentRow(
        kind="support_run",
        source_path=path,
        raw=doc,
        workload_id=doc.get("workload_id"),
        model_id=doc.get("model_id"),
        family=doc.get("family"),
        arch=doc.get("arch"),
        verdict=doc.get("outcome"),
        recorded_utc=doc.get("recorded_utc"),
        slurm_job_ids=_slurm_ids(doc.get("evidence"), doc.get("reason")),
    )


def read_attention_campaign(doc: dict, path: str) -> ExperimentRow | None:
    """The attention A/B receipt written by scripts/attention_ab_campaign.py."""
    if "arms" not in doc or "workload_id" not in doc:
        return None
    if not isinstance(doc.get("arms"), dict):
        return None
    fidelity = doc.get("fidelity") or {}
    verdict_block = fidelity.get("verdict") or {}
    evidence = fidelity.get("evidence") or {}
    budget = doc.get("budget") or {}
    native = (doc["arms"].get("native") or {})
    candidate = (doc["arms"].get("optimized") or {})

    def cv(arm: dict) -> float | None:
        median, stdev = _f(arm.get("median")), _f(arm.get("stdev"))
        return (stdev / median) if median else None

    speedup = _f(doc.get("speedup_median"))
    gate = _f(doc.get("min_end_to_end_speedup"))
    passed_speed = speedup is not None and gate is not None and speedup >= gate
    passed_quality = bool(verdict_block.get("passed"))
    return ExperimentRow(
        kind="attention_campaign",
        source_path=path,
        raw=doc,
        workload_id=doc.get("workload_id"),
        arch="sm100",
        verdict=("promoted" if (passed_speed and passed_quality) else "rejected"),
        speedup_median=speedup,
        speedup_min_to_min=_f(doc.get("speedup_min_to_min")),
        native_cv=cv(native),
        candidate_cv=cv(candidate),
        ssim=_f(evidence.get("ssim")),
        lpips=_f(evidence.get("lpips")),
        fidelity_tier=budget.get("tier"),
        fidelity_passed=verdict_block.get("passed"),
        slurm_job_ids=_slurm_ids(path),
    )


def read_fidelity_verdict(doc: dict, path: str) -> ExperimentRow | None:
    """A standalone fidelity.json written by the frame scorer."""
    if set(doc) != {"evidence", "verdict"}:
        return None
    evidence, verdict = doc["evidence"], doc["verdict"]
    return ExperimentRow(
        kind="fidelity_verdict",
        source_path=path,
        raw=doc,
        workload_id=evidence.get("frame_set"),
        ssim=_f(evidence.get("ssim")),
        lpips=_f(evidence.get("lpips")),
        fidelity_tier=verdict.get("tier"),
        fidelity_passed=verdict.get("passed"),
        verdict="passed" if verdict.get("passed") else "failed",
        slurm_job_ids=_slurm_ids(path),
    )


def read_artifact_manifest(doc: dict, path: str) -> ExperimentRow | None:
    """A packaged artifact bundle manifest -- the candidate lineage."""
    operation = doc.get("operation")
    promotion = doc.get("promotion")
    if not isinstance(operation, dict) or not isinstance(promotion, dict):
        return None
    compat = doc.get("compatibility") or {}
    payload = doc.get("payload") or {}
    files = payload.get("files") if isinstance(payload, dict) else None
    source_sha = None
    if isinstance(files, list):
        for entry in files:
            if isinstance(entry, dict) and str(entry.get("path", "")).endswith(".py"):
                source_sha = entry.get("sha256")
                break
    arches = compat.get("gpu_architectures") or []
    return ExperimentRow(
        kind="artifact_manifest",
        source_path=path,
        raw=doc,
        workload_id=(doc.get("evidence", {}).get("generation", {}) or {}).get("workload_id"),
        model_id=compat.get("model_id"),
        arch=arches[0] if arches else None,
        verdict=promotion.get("decision"),
        graph_fingerprint=operation.get("graph_fingerprint"),
        kernel_source_sha=source_sha,
        recorded_utc=promotion.get("decided_at"),
        slurm_job_ids=_slurm_ids(path),
    )


def readers():
    """Every reader, in the order ingest tries them."""
    return (
        read_dispatch_measurement,
        read_support_run,
        read_artifact_manifest,
        read_attention_campaign,
        read_fidelity_verdict,
    )


def read_document(path: Path) -> ExperimentRow | None:
    """Parse one JSON document into a row, or None when nothing recognizes it."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict):
        return None
    for reader in readers():
        row = reader(doc, str(path))
        if row is not None:
            return row
    return None


def ingest_path(root: Path) -> Iterator[tuple[Path, ExperimentRow | None]]:
    """Walk ``root`` yielding every JSON document and what it parsed to.

    Unrecognized documents are yielded with ``None`` rather than dropped, so a
    backfill can report how much it did not understand. A store that silently
    ingests half the evidence looks complete and is not.
    """
    for path in sorted(root.rglob("*.json")):
        yield path, read_document(path)
