"""Finalize a quarantined artifact bundle with real full-generation evidence.

A bundle is packaged *before* the model has ever run with it: its generation
evidence is a pending placeholder and its promotion decision is
``quarantined``. Finalization is the one step allowed to rewrite those two
sections, and only from measurements taken by the end-to-end validation stage.

The rules this module enforces are deliberately narrow:

* the bundle is verified, hash for hash, before anything is changed;
* payload bytes and isolated benchmark evidence are carried over untouched;
* only ``promoted`` and ``rejected`` rewrite the manifest -- an incomplete or
  failed validation leaves the bundle exactly as packaged;
* the manifest is written atomically and re-verified the way a runtime would;
* an already finalized bundle is never rewritten, so a resumed campaign cannot
  weaken a decision that has already shipped.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import (
    MANIFEST_FILENAME,
    ArtifactError,
    ArtifactManifest,
)
from .validator import verify_bundle

#: Decisions that mean finalization already happened. They are write-once.
FINALIZED_DECISIONS = ("promoted", "rejected")

#: End-to-end classifications that represent a completed measurement. Anything
#: else (notably ``failed``) is treated as missing evidence.
MEASURED_CLASSIFICATIONS = ("improved", "neutral", "regressed")

#: The metric recorded in generation evidence. Fixed so downstream readers can
#: compare ``value`` against ``threshold`` without parsing free text.
GENERATION_METRIC = "end_to_end_speedup"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class GenerationOutcome:
    """Everything measured about one full-generation validation run.

    This is the only input finalization trusts. It carries measurements, not
    conclusions: :meth:`decide` derives the promotion decision from them so the
    gate lives in one place and can be tested without touching a filesystem.
    """

    workload_id: str
    steps: int
    parity_passed: bool
    artifact_selected: bool
    classification: str
    min_speedup: float
    speedup: float | None = None
    stage_status: str = "ok"
    parity_policy: str = ""
    baseline_ref: str = ""
    candidate_ref: str = ""

    def __post_init__(self) -> None:
        source = "generation outcome"
        if not isinstance(self.workload_id, str) or not self.workload_id.strip():
            raise ArtifactError(f"{source}: workload_id must be a non-empty string")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int):
            raise ArtifactError(f"{source}: steps must be a positive integer")
        if self.steps <= 0:
            raise ArtifactError(f"{source}: steps must be a positive integer")
        for name in ("parity_passed", "artifact_selected"):
            if not isinstance(getattr(self, name), bool):
                raise ArtifactError(f"{source}: {name} must be a bool")
        if not isinstance(self.classification, str):
            raise ArtifactError(f"{source}: classification must be a string")
        if self.stage_status not in {"ok", "failed"}:
            raise ArtifactError(f"{source}: stage_status must be 'ok' or 'failed'")
        if (
            isinstance(self.min_speedup, bool)
            or not isinstance(self.min_speedup, (int, float))
            or not math.isfinite(float(self.min_speedup))
            or float(self.min_speedup) <= 0
        ):
            raise ArtifactError(f"{source}: min_speedup must be finite and positive")
        if self.speedup is not None and (
            isinstance(self.speedup, bool)
            or not isinstance(self.speedup, (int, float))
            or not math.isfinite(float(self.speedup))
            or float(self.speedup) < 0
        ):
            raise ArtifactError(
                f"{source}: speedup must be a finite non-negative number or None"
            )
        for name in ("parity_policy", "baseline_ref", "candidate_ref"):
            if not isinstance(getattr(self, name), str):
                raise ArtifactError(f"{source}: {name} must be a string")

    @property
    def measured(self) -> bool:
        """Whether the run produced a usable end-to-end measurement."""
        return (
            self.stage_status == "ok"
            and self.speedup is not None
            and self.classification in MEASURED_CLASSIFICATIONS
        )

    @property
    def speedup_passed(self) -> bool:
        return self.speedup is not None and float(self.speedup) >= float(
            self.min_speedup
        )

    def decide(self) -> tuple[str, str]:
        """Return the ``(decision, reason)`` this measurement supports."""
        parity = self.parity_policy or "configured"
        if self.stage_status != "ok":
            return (
                "quarantined",
                "held: the end-to-end validation stage did not complete",
            )
        if not self.parity_passed:
            return (
                "quarantined",
                f"held: full-generation output parity ({parity}) failed",
            )
        if not self.artifact_selected:
            return (
                "quarantined",
                "held: runtime dispatch did not select this artifact",
            )
        if not self.measured:
            return (
                "quarantined",
                (
                    "held: end-to-end evidence is incomplete "
                    f"(classification={self.classification!r}, "
                    f"speedup={self.speedup!r})"
                ),
            )
        measurement = (
            f"parity ({parity}) passed, dispatch selected the artifact, "
            f"classification={self.classification}, end-to-end speedup "
            f"{float(self.speedup):.4f} vs threshold {float(self.min_speedup):.4f}"
        )
        if self.classification == "improved" and self.speedup_passed:
            return ("promoted", f"promoted: {measurement}")
        return ("rejected", f"rejected: {measurement}")

    def generation_evidence(self, *, passed: bool) -> dict[str, Any]:
        """The manifest ``evidence.generation`` object for this measurement."""
        return {
            "workload_id": self.workload_id,
            "steps": self.steps,
            "metric": GENERATION_METRIC,
            "value": float(self.speedup) if self.speedup is not None else 0.0,
            "threshold": float(self.min_speedup),
            "passed": passed,
            "baseline_ref": self.baseline_ref,
            "candidate_ref": self.candidate_ref,
        }


@dataclass(frozen=True)
class FinalizationResult:
    """What finalization did to one bundle."""

    artifact_id: str
    bundle_dir: str
    manifest_path: str
    decision: str
    reason: str
    changed: bool
    manifest: ArtifactManifest

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "bundle_dir": self.bundle_dir,
            "manifest": self.manifest_path,
            "decision": self.decision,
            "reason": self.reason,
            "changed": self.changed,
        }


def _dump(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_manifest_atomic(bundle_dir: Path, document: dict[str, Any]) -> None:
    """Replace a bundle's manifest in one step, leaving no debris behind.

    The temporary file is created in the bundle's *parent* directory: a crash
    mid-write must not leave an undeclared file inside the bundle, which would
    make an otherwise intact bundle fail verification.
    """
    destination = bundle_dir / MANIFEST_FILENAME
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=bundle_dir.parent,
            prefix=f".{bundle_dir.name}.{MANIFEST_FILENAME}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(_dump(document))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def finalize_bundle(
    bundle_dir: str | Path,
    outcome: GenerationOutcome,
    *,
    decided_at: str | None = None,
) -> FinalizationResult:
    """Finalize one packaged bundle from measured generation evidence.

    The bundle is verified before and after the rewrite. If the rewrite or the
    re-verification fails for any reason the original manifest is restored, so
    an interrupted finalization always leaves a loadable bundle behind.
    """
    directory = Path(bundle_dir)
    manifest = verify_bundle(directory)
    source = str(directory)
    decision, reason = outcome.decide()

    current = manifest.promotion.decision
    if current in FINALIZED_DECISIONS:
        # Write-once: a resumed campaign re-derives the same decision and must
        # not rewrite it; a *different* decision means the run that produced
        # this evidence disagrees with the shipped bundle, which is never safe
        # to resolve silently.
        if current != decision:
            raise ArtifactError(
                f"artifact bundle {source!r}: promotion: already finalized as "
                f"{current!r}; refusing to re-decide it as {decision!r}"
            )
        return FinalizationResult(
            artifact_id=manifest.artifact_id,
            bundle_dir=source,
            manifest_path=str(directory / MANIFEST_FILENAME),
            decision=current,
            reason=manifest.promotion.reason,
            changed=False,
            manifest=manifest,
        )

    if decision == "quarantined":
        # Nothing measured is worth recording, and rewriting a quarantined
        # bundle would only churn its hashes.
        return FinalizationResult(
            artifact_id=manifest.artifact_id,
            bundle_dir=source,
            manifest_path=str(directory / MANIFEST_FILENAME),
            decision="quarantined",
            reason=reason,
            changed=False,
            manifest=manifest,
        )

    document = manifest.as_dict()
    document["evidence"]["generation"] = outcome.generation_evidence(
        passed=decision == "promoted"
    )
    document["promotion"] = {
        **manifest.promotion.as_dict(),
        "decision": decision,
        "reason": reason,
        "decided_at": decided_at or _now(),
    }
    # Parse before writing: a manifest that cannot be validated never reaches
    # disk, finalized or not.
    candidate = ArtifactManifest.from_dict(document, source=source)
    if candidate.files != manifest.files:
        raise ArtifactError(
            f"artifact bundle {source!r}: files: finalization must not change "
            "the packaged payload"
        )
    if candidate.evidence.benchmark != manifest.evidence.benchmark:
        raise ArtifactError(
            f"artifact bundle {source!r}: evidence.benchmark: finalization must "
            "not change isolated benchmark evidence"
        )

    original = (directory / MANIFEST_FILENAME).read_bytes()
    try:
        _write_manifest_atomic(directory, candidate.as_dict())
        finalized = verify_bundle(directory)
    except Exception:
        # Restore the packaged manifest so the bundle stays exactly as valid as
        # it was before this call.
        _write_manifest_atomic_bytes(directory, original)
        raise

    return FinalizationResult(
        artifact_id=finalized.artifact_id,
        bundle_dir=source,
        manifest_path=str(directory / MANIFEST_FILENAME),
        decision=finalized.promotion.decision,
        reason=finalized.promotion.reason,
        changed=True,
        manifest=finalized,
    )


def _write_manifest_atomic_bytes(bundle_dir: Path, payload: bytes) -> None:
    """Restore verbatim manifest bytes, used only on the rollback path."""
    destination = bundle_dir / MANIFEST_FILENAME
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=bundle_dir.parent,
            prefix=f".{bundle_dir.name}.{MANIFEST_FILENAME}.",
            suffix=".restore",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
