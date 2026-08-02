"""Support-matrix run records: the distilled, committable unit of evidence.

One run record says: for this workload on this architecture, a real run
produced this outcome on this date, and the full evidence lives there. The
support matrix is generated from these records and nothing else -- a cell
that cannot trace to a record is ``not_attempted`` by construction.

Records are metadata only. They never contain tensor values, prompts, model
weights, or credentials; the same forbidden-key discipline as workload
manifests applies.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from autokernel._io import write_json_atomic
from autokernel.workload.types import FORBIDDEN_METADATA_KEYS

RECORD_SCHEMA = "motionkernel.support-run"
RECORD_SCHEMA_VERSION = 1

#: Cell vocabulary. The four claim states are exactly the track-E brief's:
#: "we never tried" stays visually distinct from "we tried and it failed".
#: ``candidates_found`` is the discovery-only nightly's positive result:
#: discovery ran and found search-worthy candidates, but a discovery-only
#: campaign never searches them, so none of the four claim states applies
#: and reporting one would be a lie about the run.
OUTCOMES = (
    "promoted",
    "candidates_found",
    "no_worthwhile_candidate",
    "capture_blocked",
    "not_attempted",
)

_TOP_LEVEL_FIELDS = {
    "schema",
    "schema_version",
    "workload_id",
    "model_id",
    "family",
    "arch",
    "outcome",
    "reason",
    "recorded_utc",
    "evidence",
    "source",
}

_ARCH_PATTERN = re.compile(r"^sm\d+$")
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}")


class SupportEvidenceError(ValueError):
    """A run record is malformed, unsafe, or internally inconsistent."""


def _fail(source: object, location: str, message: str) -> SupportEvidenceError:
    return SupportEvidenceError(f"support run record {source!r}: {location}: {message}")


def _text(value: Any, source: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(source, location, "must be a non-empty string")
    return value.strip()


def _check_forbidden(raw: Mapping[str, Any], source: object, location: str) -> None:
    for key in raw:
        if not isinstance(key, str) or not key:
            raise _fail(source, location, "keys must be non-empty strings")
        if key.lower() in FORBIDDEN_METADATA_KEYS:
            raise _fail(source, f"{location}.{key}", "content or secret fields are forbidden")


@dataclass(frozen=True)
class RunRecord:
    """One workload x architecture outcome with its evidence link."""

    workload_id: str
    model_id: str
    family: str
    arch: str
    outcome: str
    recorded_utc: str
    evidence: str
    reason: str | None = None
    source: Mapping[str, Any] | None = None

    @classmethod
    def from_dict(cls, raw_value: Any, *, source: object = "<memory>") -> "RunRecord":
        if not isinstance(raw_value, Mapping):
            raise _fail(source, "top level", "must be an object")
        _check_forbidden(raw_value, source, "top level")
        unknown = sorted(set(raw_value) - _TOP_LEVEL_FIELDS)
        if unknown:
            raise _fail(source, "top level", f"unknown field(s) {unknown}")
        schema = raw_value.get("schema")
        if schema != RECORD_SCHEMA:
            raise _fail(source, "schema", f"must be {RECORD_SCHEMA!r}")
        version = raw_value.get("schema_version")
        if version != RECORD_SCHEMA_VERSION:
            raise _fail(
                source,
                "schema_version",
                f"unsupported version {version!r}; expected {RECORD_SCHEMA_VERSION}",
            )
        outcome = _text(raw_value.get("outcome"), source, "outcome")
        if outcome not in OUTCOMES:
            raise _fail(source, "outcome", f"must be one of {sorted(OUTCOMES)}")
        reason = raw_value.get("reason")
        if reason is not None:
            reason = _text(reason, source, "reason")
        if outcome == "capture_blocked" and reason is None:
            # A blocked cell with no reason is worse than useless: it is the
            # R4 section 5 failure mode, where a vague reason suppressed the
            # real cause.
            raise _fail(source, "reason", "capture_blocked must carry the reason")
        arch = _text(raw_value.get("arch"), source, "arch")
        if not _ARCH_PATTERN.fullmatch(arch):
            raise _fail(source, "arch", "must look like sm100 / sm90")
        recorded = _text(raw_value.get("recorded_utc"), source, "recorded_utc")
        if not _DATE_PATTERN.match(recorded):
            raise _fail(source, "recorded_utc", "must start with an ISO date")
        raw_source = raw_value.get("source")
        if raw_source is not None:
            if not isinstance(raw_source, Mapping):
                raise _fail(source, "source", "must be an object")
            _check_forbidden(raw_source, source, "source")
        return cls(
            workload_id=_text(raw_value.get("workload_id"), source, "workload_id"),
            model_id=_text(raw_value.get("model_id"), source, "model_id"),
            family=_text(raw_value.get("family"), source, "family"),
            arch=arch,
            outcome=outcome,
            recorded_utc=recorded,
            evidence=_text(raw_value.get("evidence"), source, "evidence"),
            reason=reason,
            source=raw_source,
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": RECORD_SCHEMA,
            "schema_version": RECORD_SCHEMA_VERSION,
            "workload_id": self.workload_id,
            "model_id": self.model_id,
            "family": self.family,
            "arch": self.arch,
            "outcome": self.outcome,
            "recorded_utc": self.recorded_utc,
            "evidence": self.evidence,
        }
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.source is not None:
            payload["source"] = dict(self.source)
        return payload


def load_run_record(path: str | Path) -> RunRecord:
    file_path = Path(path)
    if not file_path.is_file():
        raise SupportEvidenceError(f"support run record {file_path!s}: file: not found")
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SupportEvidenceError(f"support run record {file_path!s}: invalid JSON: {exc}") from exc
    return RunRecord.from_dict(raw, source=str(file_path))


def write_run_record(record: RunRecord, path: str | Path) -> Path:
    return write_json_atomic(Path(path), record.as_dict())


def record_filename(workload_id: str, arch: str) -> str:
    return f"{workload_id}--{arch}.json"


def record_from_receipt(
    receipt: Mapping[str, Any],
    *,
    workload_id: str,
    model_id: str,
    family: str,
    arch: str,
    evidence: str,
    recorded_utc: str,
) -> RunRecord | None:
    """Distill an optimize-campaign receipt into a run record.

    The receipt's terminal verdict maps onto the cell vocabulary:

    - ``promoted`` -> ``promoted`` (the real finalizer, with e2e evidence);
    - ``no_worthwhile_candidate`` -> ``no_worthwhile_candidate``;
    - ``discovery_complete`` -> discovery ran; candidates found or not, the
      cell says what discovery said;
    - ``failed`` -> ``capture_blocked`` carrying the receipt's actual
      failure message, never a paraphrase;
    - ``budget_exhausted`` -> ``None``: preemption is not a capture verdict,
      and a cell must not claim one. The nightly logs it and the cell keeps
      its previous state.
    """
    terminal = str(receipt.get("terminal") or receipt.get("status") or "")
    message = str(receipt.get("message") or "")
    candidates = receipt.get("candidates") or []
    if terminal == "promoted":
        outcome, reason = "promoted", None
    elif terminal == "no_worthwhile_candidate":
        outcome, reason = "no_worthwhile_candidate", message or None
    elif terminal == "discovery_complete":
        if candidates:
            outcome = "candidates_found"
            reason = (
                f"discovery found {len(candidates)} candidate(s); "
                "search not run (discovery-only campaign)"
            )
        else:
            outcome, reason = "no_worthwhile_candidate", message or None
    elif terminal == "failed":
        outcome = "capture_blocked"
        reason = message or "campaign failed with no message"
    elif terminal == "budget_exhausted":
        return None
    else:
        raise SupportEvidenceError(
            f"cannot distill a run record from terminal {terminal!r}"
        )
    return RunRecord.from_dict(
        {
            "schema": RECORD_SCHEMA,
            "schema_version": RECORD_SCHEMA_VERSION,
            "workload_id": workload_id,
            "model_id": model_id,
            "family": family,
            "arch": arch,
            "outcome": outcome,
            "recorded_utc": recorded_utc,
            "evidence": evidence,
            **({"reason": reason} if reason is not None else {}),
            "source": {
                "receipt_terminal": terminal,
                "completed_stages": list(receipt.get("completed_stages") or []),
            },
        }
    )
