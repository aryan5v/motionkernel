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

from ..attention.identity import (
    AttentionFallbackError,
    AttentionIdentityError,
    backend_identity,
    verify_effective_backend,
)
from ..verification.fidelity import (
    EXACT,
    FidelityBudget,
    FidelityVerdict,
    PerceptualEvidence,
    evaluate_fidelity,
)
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
    fidelity: FidelityBudget | None = None
    perceptual: PerceptualEvidence | None = None
    #: Attention artifacts only. ``declared`` is the backend the artifact was
    #: measured with; ``effective`` is the one the runtime actually resolved.
    #: They are compared because FastVideo substitutes FlashAttention silently
    #: when an optional backend cannot be imported, and a run that measured the
    #: substitute must not be credited to the declared backend.
    attention_declared: str | None = None
    attention_effective: str | None = None

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
        if self.fidelity is not None and not isinstance(self.fidelity, FidelityBudget):
            raise ArtifactError(f"{source}: fidelity must be a FidelityBudget or None")
        if self.perceptual is not None and not isinstance(
            self.perceptual, PerceptualEvidence
        ):
            raise ArtifactError(
                f"{source}: perceptual must be a PerceptualEvidence or None"
            )
        for name in ("attention_declared", "attention_effective"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ArtifactError(f"{source}: {name} must be a string or None")
        if self.attention_declared is not None:
            try:
                backend_identity(self.attention_declared)
            except AttentionIdentityError as error:
                raise ArtifactError(f"{source}: {error}") from error

    @property
    def budget(self) -> FidelityBudget:
        """The declared contract; an absent one is ``exact``, never permissive."""
        return self.fidelity if self.fidelity is not None else FidelityBudget()

    def fidelity_verdict(self) -> FidelityVerdict:
        """Check this run's output evidence against the workload's budget."""
        return evaluate_fidelity(
            self.budget, self.perceptual, parity_passed=self.parity_passed
        )

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
        """Return the ``(decision, reason)`` this measurement supports.

        The output contract is checked through the workload's fidelity budget
        rather than through ``parity_passed`` directly. At tier 1 the two are
        the same question and the behaviour is unchanged. Above it they are
        not: a SageAttention2 backend or a TeaCache schedule transform fails
        ``byte_equal`` *by construction*, so gating on parity first would
        quarantine every tier-2 candidate before its perceptual evidence was
        ever read. The budget decides which contract applies; nothing here
        weakens tier 1, because an ``exact`` budget consults exactly the same
        boolean this method used to.
        """
        parity = self.parity_policy or "configured"
        if self.stage_status != "ok":
            return (
                "quarantined",
                "held: the end-to-end validation stage did not run to completion",
            )

        # An attention artifact is only evidence about the backend that
        # actually ran. FastVideo falls back to FlashAttention silently when an
        # optional backend cannot be imported, so a campaign can otherwise
        # measure the fallback and record it under the requested backend's
        # name. Checked before anything else: if the wrong implementation ran,
        # every downstream number describes a different system.
        if self.attention_declared is not None:
            try:
                identity = verify_effective_backend(
                    self.attention_declared, self.attention_effective
                )
            except AttentionFallbackError as error:
                return ("quarantined", f"held: {error}")
            if identity.exact is False and self.budget.tier == EXACT:
                return (
                    "quarantined",
                    (
                        f"held: attention backend {identity.name!r} "
                        f"({identity.notes}) cannot satisfy a tier 1 (exact) "
                        f"fidelity budget; declare tier 2 (perceptual) or use "
                        f"a bit-exact backend"
                    ),
                )

        budget = self.budget
        verdict = self.fidelity_verdict()
        if not verdict.passed:
            if budget.tier == EXACT:
                # Preserve the original wording so existing quarantine reasons,
                # dashboards and tests keep reading the same string.
                return (
                    "quarantined",
                    f"held: full-generation output parity ({parity}) failed",
                )
            return ("quarantined", verdict.reason)
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
        contract = (
            f"parity ({parity}) passed"
            if budget.tier == EXACT
            else verdict.reason
        )
        measurement = (
            f"{contract}, dispatch selected the artifact, "
            f"classification={self.classification}, end-to-end speedup "
            f"{float(self.speedup):.4f} vs threshold {float(self.min_speedup):.4f}"
        )
        if self.classification == "improved" and self.speedup_passed:
            return ("promoted", f"promoted: {measurement}")
        return ("rejected", f"rejected: {measurement}")

    def generation_evidence(self, *, passed: bool) -> dict[str, Any]:
        """The manifest ``evidence.generation`` object for this measurement.

        Above tier 1 the fidelity block is recorded whether the artifact passed
        or failed, with the signed margin for every gated metric. An artifact
        promoted on perceptual grounds must carry the numbers that justified
        it: "SSIM 0.9912 against a 0.98 floor" is auditable a year later,
        "parity passed" would not even be true.
        """
        evidence: dict[str, Any] = {
            "workload_id": self.workload_id,
            "steps": self.steps,
            "metric": GENERATION_METRIC,
            "value": float(self.speedup) if self.speedup is not None else 0.0,
            "threshold": float(self.min_speedup),
            "passed": passed,
            "baseline_ref": self.baseline_ref,
            "candidate_ref": self.candidate_ref,
        }
        budget = self.budget
        if budget.tier != EXACT:
            evidence["fidelity"] = {
                "budget": budget.as_dict(),
                "verdict": self.fidelity_verdict().as_dict(),
            }
            if self.perceptual is not None:
                evidence["fidelity"]["evidence"] = self.perceptual.as_dict()
        return evidence


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

    # The bundle, not the caller, decides whether a backend check is required.
    # GenerationOutcome.attention_declared skips the check when it is None, so
    # an adapter that simply forgot to populate it would promote an attention
    # artifact with no verification at all -- the exact fail-open the check
    # exists to close, one level up. Cross-check against the manifest instead
    # of trusting the caller to have remembered.
    declared_on_bundle = getattr(manifest.operation, "attention_backend", None)
    if declared_on_bundle is not None:
        if outcome.attention_declared is None:
            raise ArtifactError(
                f"artifact bundle {source!r}: operation declares attention "
                f"backend {declared_on_bundle!r}, but the generation outcome "
                f"carries none; refusing to finalize an attention artifact "
                f"without verifying which backend actually ran"
            )
        if outcome.attention_declared != declared_on_bundle:
            raise ArtifactError(
                f"artifact bundle {source!r}: operation declares attention "
                f"backend {declared_on_bundle!r} but the generation outcome "
                f"was measured against {outcome.attention_declared!r}"
            )

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
            reason=manifest.promotion.reason,
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
