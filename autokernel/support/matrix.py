"""Generate the support matrix page from committed run records.

The page is mechanical: this module reads the workload manifests (the rows),
the committed run records (the cells), and emits a Markdown page plus a JSON
sidecar. It is never hand-edited, and CI regenerates and diffs it so it
cannot quietly drift back into being a claim.

Cell rules, enforced here rather than by reviewer vigilance:

- a cell with no run record is ``not_attempted`` -- never blank;
- every other cell links to its evidence and carries the record's date;
- ``capture_blocked`` always carries its reason (the record schema refuses
  to store one without);
- a record older than the staleness window is visually marked.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from autokernel.workload import load_workload

from .evidence import OUTCOMES, RunRecord, load_run_record

MATRIX_SCHEMA = "motionkernel.support-matrix"
MATRIX_SCHEMA_VERSION = 1

DEFAULT_ARCHES = ("sm100", "sm90")
DEFAULT_STALE_DAYS = 30

_OUTCOME_MEANINGS = {
    "promoted": "an artifact was promoted by the real finalizer, with e2e evidence",
    "candidates_found": "discovery ran and found search-worthy candidates; not yet searched",
    "no_worthwhile_candidate": "discovery ran, found nothing above threshold",
    "capture_blocked": "capture failed; the reason is recorded with the cell",
    "not_attempted": "no run exists",
}


class MatrixError(ValueError):
    """The matrix inputs are inconsistent."""


@dataclass(frozen=True)
class MatrixRow:
    family: str
    workload_id: str
    model_id: str


@dataclass(frozen=True)
class MatrixCell:
    outcome: str
    record: RunRecord | None
    stale: bool


def _parse_record_date(recorded_utc: str) -> date:
    return datetime.fromisoformat(recorded_utc.replace("Z", "+00:00")).date()


def load_rows(workloads_dir: str | Path) -> tuple[MatrixRow, ...]:
    """Matrix rows from the committed workload manifests."""
    root = Path(workloads_dir)
    if not root.is_dir():
        raise MatrixError(f"workloads directory not found: {root}")
    rows = []
    for path in sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml")) + sorted(root.glob("*.json")):
        manifest = load_workload(path)
        family = manifest.tags[0] if manifest.tags else manifest.workload_id
        rows.append(
            MatrixRow(
                family=family,
                workload_id=manifest.workload_id,
                model_id=manifest.model.model_id,
            )
        )
    if not rows:
        raise MatrixError(f"no workload manifests found in {root}")
    rows.sort(key=lambda row: (row.family, row.workload_id))
    return tuple(rows)


def load_records(evidence_dir: str | Path) -> dict[tuple[str, str], RunRecord]:
    """Committed run records keyed by (workload_id, arch); latest wins."""
    root = Path(evidence_dir)
    records: dict[tuple[str, str], RunRecord] = {}
    if not root.is_dir():
        return records
    for path in sorted(root.glob("*.json")):
        record = load_run_record(path)
        key = (record.workload_id, record.arch)
        existing = records.get(key)
        if existing is None or _parse_record_date(record.recorded_utc) >= _parse_record_date(
            existing.recorded_utc
        ):
            records[key] = record
    return records


def build_cells(
    rows: Sequence[MatrixRow],
    arches: Sequence[str],
    records: Mapping[tuple[str, str], RunRecord],
    *,
    today: date,
    stale_days: int = DEFAULT_STALE_DAYS,
) -> dict[tuple[str, str], MatrixCell]:
    cells: dict[tuple[str, str], MatrixCell] = {}
    for row in rows:
        for arch in arches:
            record = records.get((row.workload_id, arch))
            if record is None:
                cells[(row.workload_id, arch)] = MatrixCell(
                    outcome="not_attempted", record=None, stale=False
                )
                continue
            age = (today - _parse_record_date(record.recorded_utc)).days
            cells[(row.workload_id, arch)] = MatrixCell(
                outcome=record.outcome, record=record, stale=age > stale_days
            )
    return cells


def _render_cell(cell: MatrixCell) -> str:
    if cell.record is None:
        # Distinct from every tried-and-failed state, and never blank.
        return "*not_attempted*"
    record = cell.record
    day = record.recorded_utc[:10]
    text = f"[{cell.outcome}]({record.evidence}) {day}"
    if cell.outcome == "capture_blocked" and record.reason:
        text += f" -- {record.reason}"
    if cell.stale:
        text += " (stale)"
    return text


def render_markdown(
    rows: Sequence[MatrixRow],
    arches: Sequence[str],
    cells: Mapping[tuple[str, str], MatrixCell],
    *,
    stale_days: int,
    generated_on: date,
) -> str:
    lines = [
        "# Support matrix",
        "",
        "Model x workload x architecture, generated from committed run records.",
        "Every non-`not_attempted` cell links to its evidence and carries the",
        "date the run produced it. This page is generated by",
        "`python -m autokernel.support matrix` and checked by CI; do not",
        "hand-edit it.",
        "",
        f"Cells whose evidence is older than {stale_days} days are marked",
        "`(stale)`. Generated on "
        f"{generated_on.isoformat()}.",
        "",
        "| outcome | meaning |",
        "|---|---|",
    ]
    for outcome in OUTCOMES:
        lines.append(f"| `{outcome}` | {_OUTCOME_MEANINGS[outcome]} |")
    lines.append("")
    header = "| family | workload | " + " | ".join(arches) + " |"
    divider = "|---|---|---|" + "---|" * (len(arches) - 1)
    lines.append(header)
    lines.append(divider)
    for row in rows:
        rendered = [_render_cell(cells[(row.workload_id, arch)]) for arch in arches]
        lines.append(
            f"| {row.family} | `{row.workload_id}` | " + " | ".join(rendered) + " |"
        )
    lines.append("")
    lines.append(
        "Isolated operator results are not support claims; a cell only turns"
    )
    lines.append(
        "`promoted` when the real finalizer promotes an artifact with"
    )
    lines.append("end-to-end evidence on that architecture.")
    lines.append("")
    return "\n".join(lines)


def render_sidecar(
    rows: Sequence[MatrixRow],
    arches: Sequence[str],
    cells: Mapping[tuple[str, str], MatrixCell],
    *,
    stale_days: int,
    generated_on: date,
) -> dict[str, Any]:
    return {
        "schema": MATRIX_SCHEMA,
        "schema_version": MATRIX_SCHEMA_VERSION,
        "generated_on": generated_on.isoformat(),
        "stale_days": stale_days,
        "arches": list(arches),
        "rows": [
            {
                "family": row.family,
                "workload_id": row.workload_id,
                "model_id": row.model_id,
                "cells": {
                    arch: {
                        "outcome": cells[(row.workload_id, arch)].outcome,
                        "stale": cells[(row.workload_id, arch)].stale,
                        **(
                            {
                                "recorded_utc": cells[(row.workload_id, arch)].record.recorded_utc,
                                "evidence": cells[(row.workload_id, arch)].record.evidence,
                                **(
                                    {"reason": cells[(row.workload_id, arch)].record.reason}
                                    if cells[(row.workload_id, arch)].record.reason
                                    else {}
                                ),
                            }
                            if cells[(row.workload_id, arch)].record is not None
                            else {}
                        ),
                    }
                    for arch in arches
                },
            }
            for row in rows
        ],
    }


def generate_matrix(
    *,
    workloads_dir: str | Path,
    evidence_dir: str | Path,
    arches: Sequence[str] = DEFAULT_ARCHES,
    stale_days: int = DEFAULT_STALE_DAYS,
    today: date | None = None,
) -> tuple[str, dict[str, Any]]:
    """Generate (markdown, sidecar) deterministically for one date."""
    if today is None:
        today = datetime.now(timezone.utc).date()
    rows = load_rows(workloads_dir)
    records = load_records(evidence_dir)
    cells = build_cells(rows, arches, records, today=today, stale_days=stale_days)
    markdown = render_markdown(rows, arches, cells, stale_days=stale_days, generated_on=today)
    sidecar = render_sidecar(rows, arches, cells, stale_days=stale_days, generated_on=today)
    return markdown, sidecar
