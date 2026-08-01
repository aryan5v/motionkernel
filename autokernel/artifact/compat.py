"""Compatibility matching between artifact bundles and a live runtime.

Matching is driven entirely by data recorded in the manifest: the graph
fingerprint, the tensor signatures, and the declared environment window. There
is no per-model branch here and there must never be one -- a new model is
supported by publishing a bundle, not by editing this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .types import (
    ANY,
    ArtifactManifest,
    TensorSignature,
)

#: Structured rejection reason codes. Runtimes log these verbatim, so they are
#: part of the contract and must stay stable across schema versions.
REASON_FINGERPRINT_MISMATCH = "fingerprint_mismatch"
REASON_INPUT_SIGNATURE_MISMATCH = "input_signature_mismatch"
REASON_OUTPUT_SIGNATURE_MISMATCH = "output_signature_mismatch"
REASON_MODEL_MISMATCH = "model_mismatch"
REASON_REVISION_MISMATCH = "model_revision_mismatch"
REASON_ARCHITECTURE_MISMATCH = "gpu_architecture_mismatch"
REASON_TORCH_VERSION = "torch_version_unsupported"
REASON_CUDA_VERSION = "cuda_version_unsupported"
REASON_TRITON_VERSION = "triton_version_unsupported"
REASON_EXECUTION_MODE = "execution_mode_unsupported"
REASON_DISTRIBUTED_MODE = "distributed_mode_unsupported"
REASON_NOT_PROMOTED = "not_promoted"
REASON_EVIDENCE_INCOMPLETE = "evidence_incomplete"
REASON_NOT_SELECTED = "not_selected"

REJECTION_REASONS = (
    REASON_FINGERPRINT_MISMATCH,
    REASON_INPUT_SIGNATURE_MISMATCH,
    REASON_OUTPUT_SIGNATURE_MISMATCH,
    REASON_MODEL_MISMATCH,
    REASON_REVISION_MISMATCH,
    REASON_ARCHITECTURE_MISMATCH,
    REASON_TORCH_VERSION,
    REASON_CUDA_VERSION,
    REASON_TRITON_VERSION,
    REASON_EXECUTION_MODE,
    REASON_DISTRIBUTED_MODE,
    REASON_NOT_PROMOTED,
    REASON_EVIDENCE_INCOMPLETE,
    REASON_NOT_SELECTED,
)


@dataclass(frozen=True)
class RuntimeProfile:
    """The environment a runtime is actually executing in.

    ``cuda_version`` and ``triton_version`` are ``None`` on a CPU-only host;
    a bundle that declares a bound on either is then rejected rather than
    optimistically accepted.
    """

    model_id: str
    model_revision: str
    gpu_architecture: str
    torch_version: str
    cuda_version: str | None = None
    triton_version: str | None = None
    execution_mode: str = "inference"
    distributed_mode: str = "single"

    @classmethod
    def detect(
        cls,
        *,
        model_id: str,
        model_revision: str = ANY,
        execution_mode: str = "inference",
        distributed_mode: str = "single",
    ) -> RuntimeProfile:
        """Read torch/CUDA/Triton identity from the running process."""
        import torch

        architecture = "cpu"
        cuda_version = None
        if torch.cuda.is_available():  # pragma: no cover - needs a GPU host
            index = torch.cuda.current_device()
            major, minor = torch.cuda.get_device_capability(index)
            architecture = f"sm{major}{minor}"
            cuda_version = getattr(torch.version, "cuda", None)
        try:
            import triton

            triton_version = getattr(triton, "__version__", None)
        except Exception:  # noqa: BLE001 - Triton is optional everywhere
            triton_version = None
        return cls(
            model_id=model_id,
            model_revision=model_revision,
            gpu_architecture=architecture,
            torch_version=str(getattr(torch, "__version__", "")),
            cuda_version=cuda_version,
            triton_version=triton_version,
            execution_mode=execution_mode,
            distributed_mode=distributed_mode,
        )


@dataclass(frozen=True)
class DispatchRequest:
    """One graph invocation a runtime would like to replace."""

    graph_fingerprint: str
    inputs: tuple[TensorSignature, ...]
    outputs: tuple[TensorSignature, ...]
    runtime: RuntimeProfile

    def input_keys(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(item.match_key() for item in self.inputs)

    def output_keys(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(item.match_key() for item in self.outputs)


@dataclass(frozen=True)
class Rejection:
    """Why one artifact was not selected."""

    artifact_id: str
    reason: str
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class MatchResult:
    """The selected artifact, if any, plus why every other one was skipped."""

    manifest: ArtifactManifest | None
    rejections: tuple[Rejection, ...] = ()

    @property
    def matched(self) -> bool:
        return self.manifest is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.manifest.artifact_id if self.manifest else None,
            "rejections": [item.as_dict() for item in self.rejections],
        }


def _wildcard_equal(declared: str, observed: str) -> bool:
    return declared == ANY or declared == observed


def _describe_signatures(items: Sequence[TensorSignature]) -> str:
    return ", ".join(item.describe() for item in items) or "<none>"


def check_compatibility(
    manifest: ArtifactManifest,
    request: DispatchRequest,
) -> Rejection | None:
    """Return the first reason ``manifest`` cannot serve ``request``.

    Checks run cheapest-first so a large registry costs little per call: graph
    identity, then tensor layout, then the environment window.
    """
    identity = manifest.operation
    if identity.graph_fingerprint != request.graph_fingerprint:
        return Rejection(
            manifest.artifact_id,
            REASON_FINGERPRINT_MISMATCH,
            f"artifact {identity.graph_fingerprint} != "
            f"runtime {request.graph_fingerprint}",
        )

    if manifest.signature.input_keys() != request.input_keys():
        return Rejection(
            manifest.artifact_id,
            REASON_INPUT_SIGNATURE_MISMATCH,
            f"artifact [{_describe_signatures(manifest.signature.inputs)}] != "
            f"runtime [{_describe_signatures(request.inputs)}]",
        )
    if manifest.signature.output_keys() != request.output_keys():
        return Rejection(
            manifest.artifact_id,
            REASON_OUTPUT_SIGNATURE_MISMATCH,
            f"artifact [{_describe_signatures(manifest.signature.outputs)}] != "
            f"runtime [{_describe_signatures(request.outputs)}]",
        )

    runtime = request.runtime
    compatibility = manifest.compatibility
    if not _wildcard_equal(compatibility.model_id, runtime.model_id):
        return Rejection(
            manifest.artifact_id,
            REASON_MODEL_MISMATCH,
            f"artifact {compatibility.model_id!r} != "
            f"runtime {runtime.model_id!r}",
        )
    if not _wildcard_equal(compatibility.model_revision, runtime.model_revision):
        return Rejection(
            manifest.artifact_id,
            REASON_REVISION_MISMATCH,
            f"artifact {compatibility.model_revision!r} != "
            f"runtime {runtime.model_revision!r}",
        )
    if not any(
        _wildcard_equal(item, runtime.gpu_architecture)
        for item in compatibility.gpu_architectures
    ):
        return Rejection(
            manifest.artifact_id,
            REASON_ARCHITECTURE_MISMATCH,
            f"artifact {list(compatibility.gpu_architectures)} excludes "
            f"runtime {runtime.gpu_architecture!r}",
        )

    for range_, observed, reason, label in (
        (compatibility.torch, runtime.torch_version, REASON_TORCH_VERSION, "torch"),
        (compatibility.cuda, runtime.cuda_version, REASON_CUDA_VERSION, "cuda"),
        (
            compatibility.triton,
            runtime.triton_version,
            REASON_TRITON_VERSION,
            "triton",
        ),
    ):
        if not range_.contains(observed):
            return Rejection(
                manifest.artifact_id,
                reason,
                f"artifact requires {label} {range_.describe()}; "
                f"runtime has {observed!r}",
            )

    if runtime.execution_mode not in compatibility.execution_modes:
        return Rejection(
            manifest.artifact_id,
            REASON_EXECUTION_MODE,
            f"artifact {list(compatibility.execution_modes)} excludes "
            f"runtime {runtime.execution_mode!r}",
        )
    if runtime.distributed_mode not in compatibility.distributed_modes:
        return Rejection(
            manifest.artifact_id,
            REASON_DISTRIBUTED_MODE,
            f"artifact {list(compatibility.distributed_modes)} excludes "
            f"runtime {runtime.distributed_mode!r}",
        )

    if manifest.promotion.decision != "promoted":
        return Rejection(
            manifest.artifact_id,
            REASON_NOT_PROMOTED,
            f"promotion decision is {manifest.promotion.decision!r}",
        )
    evidence = manifest.evidence
    if not evidence.benchmark.passed or not evidence.generation.passed:
        return Rejection(
            manifest.artifact_id,
            REASON_EVIDENCE_INCOMPLETE,
            "benchmark and full-generation validation must both pass",
        )
    return None


def match_artifact(
    manifests: Iterable[ArtifactManifest],
    request: DispatchRequest,
) -> MatchResult:
    """Select the best compatible artifact for ``request``.

    When several bundles qualify the one with the strongest measured benchmark
    speedup wins; ties break on artifact id so selection is deterministic.
    """
    rejections: list[Rejection] = []
    candidates: list[ArtifactManifest] = []
    for manifest in manifests:
        rejection = check_compatibility(manifest, request)
        if rejection is None:
            candidates.append(manifest)
        else:
            rejections.append(rejection)
    if not candidates:
        return MatchResult(None, tuple(rejections))
    best = max(
        candidates,
        key=lambda item: (item.evidence.benchmark.speedup, item.artifact_id),
    )
    rejections.extend(
        Rejection(
            item.artifact_id,
            REASON_NOT_SELECTED,
            "a faster artifact was chosen",
        )
        for item in candidates
        if item is not best
    )
    return MatchResult(best, tuple(rejections))
