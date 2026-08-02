"""The experiment store: one queryable row per measured result.

Every campaign this project runs produces evidence in a different shape -- a
dispatch measurement record, an attention A/B receipt, a fidelity verdict, a
support-run outcome, a nightly discovery receipt. They live in different
directories, under different schemas, on a filesystem, and answering "what have
we already measured for this fingerprint on sm100?" means knowing where to look
and how to parse each one.

This module is the boring answer: one SQLite table, one row per result,
normalized enough to query and denormalized enough to survive schema drift in
its sources. Retrieval into search prompts comes later; the point of starting
now is that a store which begins accumulating in six months contains six months
less.

Design choices worth stating:

* **SQLite, not a service.** It lives in the evidence root next to the runs it
  indexes, it is queryable with the stdlib, and it can be copied.
* **The raw record is kept verbatim** in a JSON column alongside the extracted
  columns. Extraction is lossy and this project's schemas are still moving; a
  row whose source is preserved can be re-extracted, one whose source was
  discarded cannot.
* **Ingest is idempotent.** Records are keyed by a content hash of the source
  document, so backfilling the same evidence twice does not duplicate it and a
  re-run that produces an identical record is a no-op rather than a conflict.
* **Provenance is mandatory.** A row without a source path is refused. The
  store exists to make claims traceable, and a row nobody can trace back is
  worse than a missing row because it looks like evidence.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "STORE_SCHEMA_VERSION",
    "ExperimentRow",
    "StoreError",
    "connect",
    "ingest_row",
    "row_digest",
]

STORE_SCHEMA_VERSION = 1


class StoreError(ValueError):
    """A row cannot be stored as given."""


_DDL = """
CREATE TABLE IF NOT EXISTS experiments (
    digest              TEXT PRIMARY KEY,
    schema_version      INTEGER NOT NULL,
    kind                TEXT NOT NULL,
    workload_id         TEXT,
    model_id            TEXT,
    family              TEXT,
    arch                TEXT,
    verdict             TEXT,
    speedup_median      REAL,
    speedup_min_to_min  REAL,
    native_cv           REAL,
    candidate_cv        REAL,
    valid_for_gating    INTEGER,
    ssim                REAL,
    lpips               REAL,
    fidelity_tier       TEXT,
    fidelity_passed     INTEGER,
    kernel_source_sha   TEXT,
    graph_fingerprint   TEXT,
    slurm_job_ids       TEXT,
    recorded_utc        TEXT,
    source_path         TEXT NOT NULL,
    raw                 TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_experiments_workload ON experiments(workload_id);
CREATE INDEX IF NOT EXISTS idx_experiments_arch     ON experiments(arch);
CREATE INDEX IF NOT EXISTS idx_experiments_kind     ON experiments(kind);
CREATE INDEX IF NOT EXISTS idx_experiments_fp       ON experiments(graph_fingerprint);
CREATE TABLE IF NOT EXISTS store_meta (
    key TEXT PRIMARY KEY, value TEXT NOT NULL
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (creating if needed) the store at ``path``."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(destination)
    connection.row_factory = sqlite3.Row
    connection.executescript(_DDL)
    connection.execute(
        "INSERT OR REPLACE INTO store_meta(key, value) VALUES('schema_version', ?)",
        (str(STORE_SCHEMA_VERSION),),
    )
    connection.commit()
    return connection


def row_digest(raw: Mapping[str, Any], source_path: str) -> str:
    """Content hash of a source document, used as the primary key.

    Includes the source path so the same record copied into two evidence
    directories is two rows -- they are two pieces of provenance, and
    collapsing them would hide that one of them moved.
    """
    payload = json.dumps(
        {"source": source_path, "raw": raw}, sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class ExperimentRow:
    """One measured result, whatever produced it.

    Fields are optional because the sources genuinely differ: a dispatch
    measurement has a CV and no SSIM, a fidelity verdict is the reverse. What
    is *not* optional is ``kind``, ``source_path`` and ``raw`` -- what this is,
    where it came from, and what it said.
    """

    kind: str
    source_path: str
    raw: Mapping[str, Any]
    workload_id: str | None = None
    model_id: str | None = None
    family: str | None = None
    arch: str | None = None
    verdict: str | None = None
    speedup_median: float | None = None
    speedup_min_to_min: float | None = None
    native_cv: float | None = None
    candidate_cv: float | None = None
    valid_for_gating: bool | None = None
    ssim: float | None = None
    lpips: float | None = None
    fidelity_tier: str | None = None
    fidelity_passed: bool | None = None
    kernel_source_sha: str | None = None
    graph_fingerprint: str | None = None
    slurm_job_ids: tuple[str, ...] = field(default_factory=tuple)
    recorded_utc: str | None = None

    def __post_init__(self) -> None:
        if not self.kind or not isinstance(self.kind, str):
            raise StoreError("row kind must be a non-empty string")
        if not self.source_path or not isinstance(self.source_path, str):
            raise StoreError(
                f"{self.kind}: source_path is required -- a row nobody can trace "
                f"back looks like evidence without being any"
            )
        if not isinstance(self.raw, Mapping):
            raise StoreError(f"{self.kind}: raw must be the source mapping")

    @property
    def digest(self) -> str:
        return row_digest(self.raw, self.source_path)

    def as_params(self) -> tuple[Any, ...]:
        return (
            self.digest,
            STORE_SCHEMA_VERSION,
            self.kind,
            self.workload_id,
            self.model_id,
            self.family,
            self.arch,
            self.verdict,
            self.speedup_median,
            self.speedup_min_to_min,
            self.native_cv,
            self.candidate_cv,
            None if self.valid_for_gating is None else int(self.valid_for_gating),
            self.ssim,
            self.lpips,
            self.fidelity_tier,
            None if self.fidelity_passed is None else int(self.fidelity_passed),
            self.kernel_source_sha,
            self.graph_fingerprint,
            ",".join(self.slurm_job_ids) if self.slurm_job_ids else None,
            self.recorded_utc,
            self.source_path,
            json.dumps(self.raw, sort_keys=True, default=str),
        )


_INSERT = """
INSERT OR REPLACE INTO experiments (
    digest, schema_version, kind, workload_id, model_id, family, arch, verdict,
    speedup_median, speedup_min_to_min, native_cv, candidate_cv,
    valid_for_gating, ssim, lpips, fidelity_tier, fidelity_passed,
    kernel_source_sha, graph_fingerprint, slurm_job_ids, recorded_utc,
    source_path, raw
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""


def ingest_row(connection: sqlite3.Connection, row: ExperimentRow) -> bool:
    """Insert ``row``. Returns True when it was new.

    ``INSERT OR REPLACE`` on a content digest makes re-ingest idempotent: the
    same document produces the same key, so a backfill can be re-run safely and
    a nightly can ingest without checking what it already sent.
    """
    existing = connection.execute(
        "SELECT 1 FROM experiments WHERE digest = ?", (row.digest,)
    ).fetchone()
    connection.execute(_INSERT, row.as_params())
    return existing is None
