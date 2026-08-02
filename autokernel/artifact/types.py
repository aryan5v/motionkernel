"""Versioned artifact bundle manifest.

An artifact bundle is the packaging contract between MotionKernel and a model
runtime: an optimized graph kernel is packaged once, together with everything a
runtime needs to decide *safely* whether it may be executed, and is then loaded
without any model-specific branch on the runtime side.

The manifest records metadata only -- operation identity, tensor layout
signatures, file digests, environment compatibility, and measurement evidence.
It must never contain tensor values, model weights, prompts, or credentials.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..attention.identity import AttentionIdentityError, backend_identity
from .kinds import ATTENTION, validate_kind_fields

ARTIFACT_SCHEMA_VERSION = 1

MANIFEST_FILENAME = "artifact.json"

#: Promotion decisions. Only ``promoted`` bundles may be dispatched.
PROMOTION_DECISIONS = ("promoted", "quarantined", "rejected")

#: Execution modes a bundle may declare support for.
EXECUTION_MODES = ("inference", "training")

#: Distributed modes a bundle may declare support for. ``single`` means the
#: module runs unsharded in a single process.
DISTRIBUTED_MODES = (
    "single",
    "data_parallel",
    "tensor_parallel",
    "sequence_parallel",
    "pipeline_parallel",
)

#: Wildcard accepted by the string-valued compatibility fields.
ANY = "*"

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "artifact_id",
    "created_at",
    "producer",
    "operation",
    "signature",
    "entry_point",
    "files",
    "compatibility",
    "evidence",
    "promotion",
}
_OPERATION_FIELDS = {
    "name",
    "graph_fingerprint",
    "parent_module",
    "operations",
    "target_kind",
    "capture_mode",
    "selected_node_ids",
    "boundary_refs",
    "output_node_ids",
    "attention_backend",
    "attention_config",
}
#: Operation-identity fields every kind carries. Anything outside this set must
#: be claimed by a registered kind, so a field added to the schema and never
#: attached to a kind fails validation instead of quietly applying to all of
#: them.
_OPERATION_COMMON_FIELDS = frozenset(
    {"name", "graph_fingerprint", "parent_module", "operations", "target_kind"}
)
_SIGNATURE_FIELDS = {"inputs", "outputs"}
_TENSOR_FIELDS = {"name", "shape", "stride", "dtype", "device_type", "requires_grad"}
_ENTRY_POINT_FIELDS = {"file", "symbol"}
_FILE_FIELDS = {"path", "sha256", "bytes"}
_COMPATIBILITY_FIELDS = {
    "model_id",
    "model_revision",
    "gpu_architectures",
    "torch",
    "cuda",
    "triton",
    "execution_modes",
    "distributed_modes",
}
_VERSION_RANGE_FIELDS = {"min", "max_exclusive"}
_EVIDENCE_FIELDS = {"benchmark", "generation"}
_BENCHMARK_FIELDS = {
    "harness",
    "device",
    "samples",
    "baseline_us",
    "candidate_us",
    "speedup",
    "max_abs_error",
    "max_rel_error",
    "atol",
    "rtol",
    "passed",
    "result_ref",
}
_GENERATION_FIELDS = {
    "workload_id",
    "steps",
    "metric",
    "value",
    "threshold",
    "passed",
    "baseline_ref",
    "candidate_ref",
    "fidelity",
}
_PROMOTION_FIELDS = {"decision", "reason", "decided_at", "campaign"}
_CAMPAIGN_FIELDS = {"campaign_id", "source", "target_name"}
_PRODUCER_FIELDS = {"name", "version"}

#: Keys that must never appear anywhere in a manifest, at any depth. Mirrors
#: the discovery/campaign contracts so one leak check covers every hop.
FORBIDDEN_KEYS = frozenset(
    {
        "activations",
        "credential",
        "credentials",
        "data",
        "password",
        "prompt",
        "prompts",
        "secret",
        "secrets",
        "source_code",
        "tensor_values",
        "token",
        "values",
        "weights",
    }
)

# Artifact ids and operation names become directory names and log keys.
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SYMBOL_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
# Relative POSIX paths only: no absolute paths, no parent traversal, no
# backslashes that a Windows runtime would reinterpret.
_RELATIVE_FILE_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._/-]{0,255}$")
_VERSION_PATTERN = re.compile(r"^[0-9]+(\.[0-9]+)*[A-Za-z0-9.+_-]*$")
_IR_NODE_PATTERN = re.compile(r"^n[0-9]+$")
_IR_REF_PATTERN = re.compile(r"^[pn][0-9]+$")


class ArtifactError(ValueError):
    """Raised when an artifact bundle is malformed or unsafe to load."""


def _fail(source: object, location: str, message: str) -> ArtifactError:
    return ArtifactError(f"artifact bundle {source!r}: {location}: {message}")


def _mapping(
    value: Any,
    source: object,
    location: str,
    *,
    non_empty: bool = False,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or (non_empty and not value):
        qualifier = "non-empty " if non_empty else ""
        raise _fail(source, location, f"must be a {qualifier}object")
    for key in value:
        if not isinstance(key, str) or not key:
            raise _fail(source, location, "keys must be non-empty strings")
    return value


def _unknown_fields(
    raw: Mapping[str, Any],
    allowed: set[str],
    source: object,
    location: str,
) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise _fail(source, location, f"unknown field(s) {unknown}")


def _sequence(value: Any, source: object, location: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _fail(source, location, "must be a list")
    return value


def _text(value: Any, source: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(source, location, "must be a non-empty string")
    return value


def _pattern_text(
    value: Any,
    pattern: re.Pattern[str],
    source: object,
    location: str,
    description: str,
) -> str:
    text = _text(value, source, location)
    if not pattern.fullmatch(text):
        raise _fail(source, location, f"must be {description}")
    return text


def _bool(value: Any, source: object, location: str) -> bool:
    if not isinstance(value, bool):
        raise _fail(source, location, "must be a bool")
    return value


def _positive_int(value: Any, source: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _fail(source, location, "must be a positive integer")
    return value


def _finite_non_negative(value: Any, source: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(source, location, "must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise _fail(source, location, "must be a finite non-negative number")
    return result


def _string_choices(
    value: Any,
    allowed: Sequence[str],
    source: object,
    location: str,
) -> tuple[str, ...]:
    items = _sequence(value, source, location)
    if not items:
        raise _fail(source, location, "must be a non-empty list")
    result: list[str] = []
    for index, item in enumerate(items):
        text = _text(item, source, f"{location}[{index}]")
        if text not in allowed:
            raise _fail(
                source,
                f"{location}[{index}]",
                f"must be one of {sorted(allowed)}",
            )
        if text in result:
            raise _fail(source, location, f"contains duplicate {text!r}")
        result.append(text)
    return tuple(result)


def assert_no_forbidden_keys(value: Any, source: object, location: str) -> None:
    """Reject content/secret-shaped keys anywhere in a manifest."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key.lower() in FORBIDDEN_KEYS:
                raise _fail(
                    source,
                    f"{location}.{key}",
                    "content or secret fields are forbidden",
                )
            assert_no_forbidden_keys(item, source, f"{location}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            assert_no_forbidden_keys(item, source, f"{location}[{index}]")


def parse_version(text: str) -> tuple[int, ...] | None:
    """Parse a leading dotted-numeric version, ignoring local/pre-release tags.

    ``"2.12.0+cu128"`` and ``"2.12.0a0"`` both parse to ``(2, 12, 0)``. Returns
    ``None`` when no numeric prefix is present, which callers treat as "cannot
    compare" rather than "compatible".
    """
    parts: list[int] = []
    for chunk in str(text).strip().split("."):
        digits = ""
        for character in chunk:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or None


def _compare_versions(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    """Compare two parsed versions, zero-padding the shorter one."""
    width = max(len(left), len(right))
    padded_left = left + (0,) * (width - len(left))
    padded_right = right + (0,) * (width - len(right))
    if padded_left < padded_right:
        return -1
    return 0 if padded_left == padded_right else 1


@dataclass(frozen=True)
class VersionRange:
    """An inclusive-minimum, exclusive-maximum version window.

    Both bounds are optional; an unbounded range accepts any parseable version.
    """

    minimum: str | None = None
    maximum_exclusive: str | None = None

    @classmethod
    def from_dict(
        cls,
        raw_value: Any,
        *,
        source: object,
        location: str,
    ) -> "VersionRange":
        raw = _mapping(raw_value, source, location)
        _unknown_fields(raw, _VERSION_RANGE_FIELDS, source, location)
        minimum = raw.get("min")
        maximum = raw.get("max_exclusive")
        if minimum is not None:
            minimum = _pattern_text(
                minimum,
                _VERSION_PATTERN,
                source,
                f"{location}.min",
                "a dotted version string",
            )
        if maximum is not None:
            maximum = _pattern_text(
                maximum,
                _VERSION_PATTERN,
                source,
                f"{location}.max_exclusive",
                "a dotted version string",
            )
        if minimum is not None and maximum is not None:
            low = parse_version(minimum)
            high = parse_version(maximum)
            if low is not None and high is not None and _compare_versions(low, high) >= 0:
                raise _fail(
                    source,
                    location,
                    "min must be lower than max_exclusive",
                )
        return cls(minimum=minimum, maximum_exclusive=maximum)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.minimum is not None:
            result["min"] = self.minimum
        if self.maximum_exclusive is not None:
            result["max_exclusive"] = self.maximum_exclusive
        return result

    @property
    def unbounded(self) -> bool:
        return self.minimum is None and self.maximum_exclusive is None

    def contains(self, version: str | None) -> bool:
        """Whether ``version`` satisfies this range.

        An unparsable or missing version satisfies only an unbounded range:
        a bound that cannot be evaluated must never be assumed to hold.
        """
        if self.unbounded:
            return True
        if version is None:
            return False
        observed = parse_version(version)
        if observed is None:
            return False
        if self.minimum is not None:
            low = parse_version(self.minimum)
            if low is None or _compare_versions(observed, low) < 0:
                return False
        if self.maximum_exclusive is not None:
            high = parse_version(self.maximum_exclusive)
            if high is None or _compare_versions(observed, high) >= 0:
                return False
        return True

    def describe(self) -> str:
        if self.unbounded:
            return "any"
        low = self.minimum if self.minimum is not None else "any"
        high = self.maximum_exclusive if self.maximum_exclusive is not None else "any"
        return f">={low},<{high}"


@dataclass(frozen=True)
class TensorSignature:
    """One tensor's layout, with no tensor contents.

    ``name`` is informational: it is carried for diagnostics and deliberately
    excluded from :meth:`match_key`, so a runtime that names its arguments
    differently still matches a compatible bundle.
    """

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device_type: str
    requires_grad: bool = False
    name: str = ""

    @classmethod
    def from_dict(
        cls,
        raw_value: Any,
        *,
        source: object,
        location: str,
    ) -> "TensorSignature":
        raw = _mapping(raw_value, source, location, non_empty=True)
        _unknown_fields(raw, _TENSOR_FIELDS, source, location)
        shape = _dimensions(
            raw.get("shape"),
            source,
            f"{location}.shape",
            non_negative=True,
        )
        stride = _dimensions(
            raw.get("stride"),
            source,
            f"{location}.stride",
            non_negative=False,
        )
        if len(stride) != len(shape):
            raise _fail(
                source,
                f"{location}.stride",
                "must have the same length as shape",
            )
        name = raw.get("name", "")
        if not isinstance(name, str):
            raise _fail(source, f"{location}.name", "must be a string")
        return cls(
            shape=shape,
            stride=stride,
            dtype=_text(raw.get("dtype"), source, f"{location}.dtype"),
            device_type=_text(
                raw.get("device_type"), source, f"{location}.device_type"
            ),
            requires_grad=_bool(
                raw.get("requires_grad", False),
                source,
                f"{location}.requires_grad",
            ),
            name=name,
        )

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "shape": list(self.shape),
            "stride": list(self.stride),
            "dtype": self.dtype,
            "device_type": self.device_type,
            "requires_grad": self.requires_grad,
        }
        if self.name:
            result["name"] = self.name
        return result

    def match_key(self) -> tuple[Any, ...]:
        """The tuple compared during dispatch. Excludes ``name``."""
        return (
            self.shape,
            self.stride,
            self.dtype,
            self.device_type,
            self.requires_grad,
        )

    def describe(self) -> str:
        dims = "x".join(str(dim) for dim in self.shape)
        return f"{dims}:{self.dtype}:{self.device_type}"


def _dimensions(
    value: Any,
    source: object,
    location: str,
    *,
    non_negative: bool,
) -> tuple[int, ...]:
    items = _sequence(value, source, location)
    result: list[int] = []
    for index, item in enumerate(items):
        if isinstance(item, bool) or not isinstance(item, int):
            raise _fail(source, f"{location}[{index}]", "must be an integer")
        if non_negative and item < 0:
            raise _fail(
                source, f"{location}[{index}]", "must be a non-negative integer"
            )
        result.append(item)
    return tuple(result)


def _tensor_list(
    value: Any,
    source: object,
    location: str,
    *,
    non_empty: bool,
) -> tuple[TensorSignature, ...]:
    items = _sequence(value, source, location)
    if non_empty and not items:
        raise _fail(source, location, "must be a non-empty list")
    return tuple(
        TensorSignature.from_dict(
            item,
            source=source,
            location=f"{location}[{index}]",
        )
        for index, item in enumerate(items)
    )


@dataclass(frozen=True)
class OperationIdentity:
    """What graph this bundle replaces, independent of any model name."""

    name: str
    graph_fingerprint: str
    parent_module: str
    operations: tuple[str, ...]
    target_kind: str = "module"
    capture_mode: str | None = None
    selected_node_ids: tuple[str, ...] = ()
    boundary_refs: tuple[str, ...] = ()
    output_node_ids: tuple[str, ...] = ()
    #: Attention targets only: the AttentionBackendEnum member the artifact was
    #: measured with, and its configuration. Recorded so a run can be refused
    #: when a different backend actually executes -- FastVideo falls back to
    #: FlashAttention silently when an optional backend cannot be imported.
    attention_backend: str | None = None
    attention_config: Mapping[str, Any] | None = None

    @classmethod
    def from_dict(
        cls,
        raw_value: Any,
        *,
        source: object,
        location: str,
    ) -> "OperationIdentity":
        raw = _mapping(raw_value, source, location, non_empty=True)
        _unknown_fields(raw, _OPERATION_FIELDS, source, location)
        operations = _sequence(
            raw.get("operations"), source, f"{location}.operations"
        )
        if not operations:
            raise _fail(source, f"{location}.operations", "must be a non-empty list")
        target_kind = raw.get("target_kind", "module")
        # Kinds are registered rather than enumerated here, so adding one is a
        # registration instead of an edit to a conditional every other kind
        # also reads. See autokernel/artifact/kinds.py.
        try:
            validate_kind_fields(
                target_kind, raw, common=_OPERATION_COMMON_FIELDS
            )
        except ValueError as error:
            raise _fail(source, f"{location}.target_kind", str(error)) from error

        attention_backend: str | None = None
        attention_config: Mapping[str, Any] | None = None
        if target_kind == ATTENTION:
            attention_backend = _text(
                raw.get("attention_backend"),
                source,
                f"{location}.attention_backend",
            )
            try:
                backend_identity(attention_backend)
            except AttentionIdentityError as error:
                raise _fail(
                    source, f"{location}.attention_backend", str(error)
                ) from error
            raw_config = raw.get("attention_config")
            if raw_config is not None:
                attention_config = dict(
                    _mapping(
                        raw_config, source, f"{location}.attention_config"
                    )
                )

        capture_mode: str | None = None
        selected_node_ids: tuple[str, ...] = ()
        boundary_refs: tuple[str, ...] = ()
        output_node_ids: tuple[str, ...] = ()
        if target_kind == "subgraph":
            capture_mode = _text(
                raw.get("capture_mode"), source, f"{location}.capture_mode"
            )
            if capture_mode != "export":
                raise _fail(
                    source,
                    f"{location}.capture_mode",
                    "subgraph dispatch currently requires 'export'",
                )

            def refs(field: str, pattern: re.Pattern[str]) -> tuple[str, ...]:
                items = _sequence(raw.get(field), source, f"{location}.{field}")
                if not items:
                    raise _fail(
                        source, f"{location}.{field}", "must be a non-empty list"
                    )
                result = tuple(
                    _pattern_text(
                        item,
                        pattern,
                        source,
                        f"{location}.{field}[{index}]",
                        "a canonical executable-IR reference",
                    )
                    for index, item in enumerate(items)
                )
                if len(result) != len(set(result)):
                    raise _fail(
                        source, f"{location}.{field}", "must not contain duplicates"
                    )
                return result

            selected_node_ids = refs("selected_node_ids", _IR_NODE_PATTERN)
            boundary_refs = refs("boundary_refs", _IR_REF_PATTERN)
            output_node_ids = refs("output_node_ids", _IR_NODE_PATTERN)
            if not set(output_node_ids).issubset(selected_node_ids):
                raise _fail(
                    source,
                    f"{location}.output_node_ids",
                    "must be selected nodes",
                )

        return cls(
            name=_pattern_text(
                raw.get("name"),
                _NAME_PATTERN,
                source,
                f"{location}.name",
                "an alphanumeric name (dots, dashes and underscores allowed)",
            ),
            graph_fingerprint=_pattern_text(
                raw.get("graph_fingerprint"),
                _FINGERPRINT_PATTERN,
                source,
                f"{location}.graph_fingerprint",
                "32 lowercase hex characters",
            ),
            parent_module=_text(
                raw.get("parent_module"), source, f"{location}.parent_module"
            ),
            operations=tuple(
                _text(item, source, f"{location}.operations[{index}]")
                for index, item in enumerate(operations)
            ),
            target_kind=target_kind,
            capture_mode=capture_mode,
            selected_node_ids=selected_node_ids,
            boundary_refs=boundary_refs,
            output_node_ids=output_node_ids,
            attention_backend=attention_backend,
            attention_config=attention_config,
        )

    def as_dict(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "graph_fingerprint": self.graph_fingerprint,
            "parent_module": self.parent_module,
            "operations": list(self.operations),
        }
        if self.target_kind == ATTENTION:
            result["target_kind"] = self.target_kind
            result["attention_backend"] = self.attention_backend
            if self.attention_config is not None:
                result["attention_config"] = dict(self.attention_config)
        if self.target_kind == "subgraph":
            result.update(
                {
                    "target_kind": self.target_kind,
                    "capture_mode": self.capture_mode,
                    "selected_node_ids": list(self.selected_node_ids),
                    "boundary_refs": list(self.boundary_refs),
                    "output_node_ids": list(self.output_node_ids),
                }
            )
        return result


@dataclass(frozen=True)
class GraphSignature:
    """The input/output tensor layouts this bundle was built and proven for."""

    inputs: tuple[TensorSignature, ...]
    outputs: tuple[TensorSignature, ...]

    @classmethod
    def from_dict(
        cls,
        raw_value: Any,
        *,
        source: object,
        location: str,
    ) -> "GraphSignature":
        raw = _mapping(raw_value, source, location, non_empty=True)
        _unknown_fields(raw, _SIGNATURE_FIELDS, source, location)
        return cls(
            inputs=_tensor_list(
                raw.get("inputs"), source, f"{location}.inputs", non_empty=True
            ),
            outputs=_tensor_list(
                raw.get("outputs"), source, f"{location}.outputs", non_empty=True
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "inputs": [item.as_dict() for item in self.inputs],
            "outputs": [item.as_dict() for item in self.outputs],
        }

    def input_keys(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(item.match_key() for item in self.inputs)

    def output_keys(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(item.match_key() for item in self.outputs)


@dataclass(frozen=True)
class EntryPoint:
    """The candidate callable inside the bundle."""

    file: str
    symbol: str

    @classmethod
    def from_dict(
        cls,
        raw_value: Any,
        *,
        source: object,
        location: str,
    ) -> "EntryPoint":
        raw = _mapping(raw_value, source, location, non_empty=True)
        _unknown_fields(raw, _ENTRY_POINT_FIELDS, source, location)
        file = _relative_path(raw.get("file"), source, f"{location}.file")
        if not file.endswith(".py"):
            raise _fail(source, f"{location}.file", "must be a .py file")
        return cls(
            file=file,
            symbol=_pattern_text(
                raw.get("symbol"),
                _SYMBOL_PATTERN,
                source,
                f"{location}.symbol",
                "a Python identifier",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {"file": self.file, "symbol": self.symbol}


def _relative_path(value: Any, source: object, location: str) -> str:
    text = _pattern_text(
        value,
        _RELATIVE_FILE_PATTERN,
        source,
        location,
        "a relative POSIX path",
    )
    parts = text.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise _fail(source, location, "must not contain empty or relative segments")
    return text


@dataclass(frozen=True)
class FileDigest:
    """One bundled file, pinned by size and content hash."""

    path: str
    sha256: str
    bytes: int

    @classmethod
    def from_dict(
        cls,
        raw_value: Any,
        *,
        source: object,
        location: str,
    ) -> "FileDigest":
        raw = _mapping(raw_value, source, location, non_empty=True)
        _unknown_fields(raw, _FILE_FIELDS, source, location)
        size = raw.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise _fail(source, f"{location}.bytes", "must be a non-negative integer")
        return cls(
            path=_relative_path(raw.get("path"), source, f"{location}.path"),
            sha256=_pattern_text(
                raw.get("sha256"),
                _SHA256_PATTERN,
                source,
                f"{location}.sha256",
                "64 lowercase hex characters",
            ),
            bytes=size,
        )

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True)
class Compatibility:
    """The environments in which this bundle is allowed to run."""

    model_id: str
    model_revision: str
    gpu_architectures: tuple[str, ...]
    torch: VersionRange
    cuda: VersionRange
    triton: VersionRange
    execution_modes: tuple[str, ...]
    distributed_modes: tuple[str, ...]

    @classmethod
    def from_dict(
        cls,
        raw_value: Any,
        *,
        source: object,
        location: str,
    ) -> "Compatibility":
        raw = _mapping(raw_value, source, location, non_empty=True)
        _unknown_fields(raw, _COMPATIBILITY_FIELDS, source, location)
        architectures = _sequence(
            raw.get("gpu_architectures"), source, f"{location}.gpu_architectures"
        )
        if not architectures:
            raise _fail(
                source, f"{location}.gpu_architectures", "must be a non-empty list"
            )
        return cls(
            model_id=_text(raw.get("model_id"), source, f"{location}.model_id"),
            model_revision=_text(
                raw.get("model_revision"), source, f"{location}.model_revision"
            ),
            gpu_architectures=tuple(
                _text(item, source, f"{location}.gpu_architectures[{index}]")
                for index, item in enumerate(architectures)
            ),
            torch=VersionRange.from_dict(
                raw.get("torch", {}), source=source, location=f"{location}.torch"
            ),
            cuda=VersionRange.from_dict(
                raw.get("cuda", {}), source=source, location=f"{location}.cuda"
            ),
            triton=VersionRange.from_dict(
                raw.get("triton", {}), source=source, location=f"{location}.triton"
            ),
            execution_modes=_string_choices(
                raw.get("execution_modes"),
                EXECUTION_MODES,
                source,
                f"{location}.execution_modes",
            ),
            distributed_modes=_string_choices(
                raw.get("distributed_modes"),
                DISTRIBUTED_MODES,
                source,
                f"{location}.distributed_modes",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "gpu_architectures": list(self.gpu_architectures),
            "torch": self.torch.as_dict(),
            "cuda": self.cuda.as_dict(),
            "triton": self.triton.as_dict(),
            "execution_modes": list(self.execution_modes),
            "distributed_modes": list(self.distributed_modes),
        }


@dataclass(frozen=True)
class BenchmarkEvidence:
    """Isolated benchmark evidence produced by the kernel harness."""

    harness: str
    device: str
    samples: int
    baseline_us: float
    candidate_us: float
    speedup: float
    max_abs_error: float
    max_rel_error: float
    atol: float
    rtol: float
    passed: bool
    result_ref: str = ""

    @classmethod
    def from_dict(
        cls,
        raw_value: Any,
        *,
        source: object,
        location: str,
    ) -> "BenchmarkEvidence":
        raw = _mapping(raw_value, source, location, non_empty=True)
        _unknown_fields(raw, _BENCHMARK_FIELDS, source, location)
        result_ref = raw.get("result_ref", "")
        if not isinstance(result_ref, str):
            raise _fail(source, f"{location}.result_ref", "must be a string")
        return cls(
            harness=_text(raw.get("harness"), source, f"{location}.harness"),
            device=_text(raw.get("device"), source, f"{location}.device"),
            samples=_positive_int(raw.get("samples"), source, f"{location}.samples"),
            baseline_us=_finite_non_negative(
                raw.get("baseline_us"), source, f"{location}.baseline_us"
            ),
            candidate_us=_finite_non_negative(
                raw.get("candidate_us"), source, f"{location}.candidate_us"
            ),
            speedup=_finite_non_negative(
                raw.get("speedup"), source, f"{location}.speedup"
            ),
            max_abs_error=_finite_non_negative(
                raw.get("max_abs_error"), source, f"{location}.max_abs_error"
            ),
            max_rel_error=_finite_non_negative(
                raw.get("max_rel_error"), source, f"{location}.max_rel_error"
            ),
            atol=_finite_non_negative(raw.get("atol"), source, f"{location}.atol"),
            rtol=_finite_non_negative(raw.get("rtol"), source, f"{location}.rtol"),
            passed=_bool(raw.get("passed"), source, f"{location}.passed"),
            result_ref=result_ref,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "harness": self.harness,
            "device": self.device,
            "samples": self.samples,
            "baseline_us": self.baseline_us,
            "candidate_us": self.candidate_us,
            "speedup": self.speedup,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "atol": self.atol,
            "rtol": self.rtol,
            "passed": self.passed,
            "result_ref": self.result_ref,
        }


@dataclass(frozen=True)
class GenerationEvidence:
    """Full-generation validation evidence from an end-to-end model run."""

    workload_id: str
    steps: int
    metric: str
    value: float
    threshold: float
    passed: bool
    baseline_ref: str = ""
    candidate_ref: str = ""
    #: Tiered-fidelity record, present only above tier 1. Carries the declared
    #: budget, the verdict, and the signed margin for every gated metric, so a
    #: perceptual promotion can be audited from the manifest alone. Validated
    #: as a shape here; the authoritative rules live in
    #: :mod:`autokernel.verification.fidelity`, which owns the contract.
    fidelity: Mapping[str, Any] | None = None

    @classmethod
    def from_dict(
        cls,
        raw_value: Any,
        *,
        source: object,
        location: str,
    ) -> "GenerationEvidence":
        raw = _mapping(raw_value, source, location, non_empty=True)
        _unknown_fields(raw, _GENERATION_FIELDS, source, location)
        fidelity = raw.get("fidelity")
        if fidelity is not None:
            fidelity = dict(
                _mapping(fidelity, source, f"{location}.fidelity", non_empty=True)
            )
            for required in ("budget", "verdict"):
                if required not in fidelity:
                    raise _fail(
                        source,
                        f"{location}.fidelity",
                        f"must contain {required!r}",
                    )
        refs = {}
        for key in ("baseline_ref", "candidate_ref"):
            value = raw.get(key, "")
            if not isinstance(value, str):
                raise _fail(source, f"{location}.{key}", "must be a string")
            refs[key] = value
        return cls(
            workload_id=_text(
                raw.get("workload_id"), source, f"{location}.workload_id"
            ),
            steps=_positive_int(raw.get("steps"), source, f"{location}.steps"),
            metric=_text(raw.get("metric"), source, f"{location}.metric"),
            value=_finite_non_negative(raw.get("value"), source, f"{location}.value"),
            threshold=_finite_non_negative(
                raw.get("threshold"), source, f"{location}.threshold"
            ),
            passed=_bool(raw.get("passed"), source, f"{location}.passed"),
            fidelity=fidelity,
            **refs,
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "workload_id": self.workload_id,
            "steps": self.steps,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "passed": self.passed,
            "baseline_ref": self.baseline_ref,
            "candidate_ref": self.candidate_ref,
        }
        if self.fidelity is not None:
            payload["fidelity"] = dict(self.fidelity)
        return payload


@dataclass(frozen=True)
class Evidence:
    """Both measurement gates a bundle must carry to be promotable."""

    benchmark: BenchmarkEvidence
    generation: GenerationEvidence

    @classmethod
    def from_dict(
        cls,
        raw_value: Any,
        *,
        source: object,
        location: str,
    ) -> "Evidence":
        raw = _mapping(raw_value, source, location, non_empty=True)
        _unknown_fields(raw, _EVIDENCE_FIELDS, source, location)
        return cls(
            benchmark=BenchmarkEvidence.from_dict(
                raw.get("benchmark"), source=source, location=f"{location}.benchmark"
            ),
            generation=GenerationEvidence.from_dict(
                raw.get("generation"),
                source=source,
                location=f"{location}.generation",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark.as_dict(),
            "generation": self.generation.as_dict(),
        }


@dataclass(frozen=True)
class CampaignRef:
    """Which optimization campaign produced this candidate."""

    campaign_id: str
    source: str
    target_name: str

    @classmethod
    def from_dict(
        cls,
        raw_value: Any,
        *,
        source: object,
        location: str,
    ) -> "CampaignRef":
        raw = _mapping(raw_value, source, location, non_empty=True)
        _unknown_fields(raw, _CAMPAIGN_FIELDS, source, location)
        return cls(
            campaign_id=_text(
                raw.get("campaign_id"), source, f"{location}.campaign_id"
            ),
            source=_text(raw.get("source"), source, f"{location}.source"),
            target_name=_text(
                raw.get("target_name"), source, f"{location}.target_name"
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "source": self.source,
            "target_name": self.target_name,
        }


@dataclass(frozen=True)
class Promotion:
    """The recorded decision to ship, hold, or reject this candidate."""

    decision: str
    reason: str
    decided_at: str
    campaign: CampaignRef

    @classmethod
    def from_dict(
        cls,
        raw_value: Any,
        *,
        source: object,
        location: str,
    ) -> "Promotion":
        raw = _mapping(raw_value, source, location, non_empty=True)
        _unknown_fields(raw, _PROMOTION_FIELDS, source, location)
        decision = _text(raw.get("decision"), source, f"{location}.decision")
        if decision not in PROMOTION_DECISIONS:
            raise _fail(
                source,
                f"{location}.decision",
                f"must be one of {sorted(PROMOTION_DECISIONS)}",
            )
        return cls(
            decision=decision,
            reason=_text(raw.get("reason"), source, f"{location}.reason"),
            decided_at=_text(
                raw.get("decided_at"), source, f"{location}.decided_at"
            ),
            campaign=CampaignRef.from_dict(
                raw.get("campaign"), source=source, location=f"{location}.campaign"
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "decided_at": self.decided_at,
            "campaign": self.campaign.as_dict(),
        }


@dataclass(frozen=True)
class ArtifactManifest:
    """A complete, validated artifact bundle manifest."""

    artifact_id: str
    created_at: str
    producer: Mapping[str, Any]
    operation: OperationIdentity
    signature: GraphSignature
    entry_point: EntryPoint
    files: tuple[FileDigest, ...]
    compatibility: Compatibility
    evidence: Evidence
    promotion: Promotion
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    @classmethod
    def from_dict(
        cls,
        raw_value: Any,
        *,
        source: object = "<memory>",
    ) -> "ArtifactManifest":
        raw = _mapping(raw_value, source, "top level", non_empty=True)
        _unknown_fields(raw, _TOP_LEVEL_FIELDS, source, "top level")
        assert_no_forbidden_keys(raw, source, "manifest")
        version = raw.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise _fail(source, "schema_version", "must be an integer")
        if version != ARTIFACT_SCHEMA_VERSION:
            raise _fail(
                source,
                "schema_version",
                f"unsupported version {version}; "
                f"expected {ARTIFACT_SCHEMA_VERSION}",
            )
        producer = _mapping(
            raw.get("producer"), source, "producer", non_empty=True
        )
        _unknown_fields(producer, _PRODUCER_FIELDS, source, "producer")
        for key in _PRODUCER_FIELDS:
            _text(producer.get(key), source, f"producer.{key}")

        files = _sequence(raw.get("files"), source, "files")
        if not files:
            raise _fail(source, "files", "must be a non-empty list")
        digests = tuple(
            FileDigest.from_dict(item, source=source, location=f"files[{index}]")
            for index, item in enumerate(files)
        )
        paths = [item.path for item in digests]
        if len(paths) != len(set(paths)):
            raise _fail(source, "files", "contains duplicate paths")
        if MANIFEST_FILENAME in paths:
            raise _fail(
                source,
                "files",
                f"must not declare {MANIFEST_FILENAME}; the manifest cannot "
                "hash itself",
            )

        entry_point = EntryPoint.from_dict(
            raw.get("entry_point"), source=source, location="entry_point"
        )
        if entry_point.file not in paths:
            raise _fail(
                source,
                "entry_point.file",
                f"{entry_point.file!r} is not a declared file",
            )

        signature = GraphSignature.from_dict(
            raw.get("signature"), source=source, location="signature"
        )
        return cls(
            artifact_id=_pattern_text(
                raw.get("artifact_id"),
                _NAME_PATTERN,
                source,
                "artifact_id",
                "an alphanumeric id (dots, dashes and underscores allowed)",
            ),
            created_at=_text(raw.get("created_at"), source, "created_at"),
            producer={key: producer[key] for key in sorted(producer)},
            operation=OperationIdentity.from_dict(
                raw.get("operation"), source=source, location="operation"
            ),
            signature=signature,
            entry_point=entry_point,
            files=digests,
            compatibility=Compatibility.from_dict(
                raw.get("compatibility"), source=source, location="compatibility"
            ),
            evidence=Evidence.from_dict(
                raw.get("evidence"), source=source, location="evidence"
            ),
            promotion=Promotion.from_dict(
                raw.get("promotion"), source=source, location="promotion"
            ),
            schema_version=version,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "created_at": self.created_at,
            "producer": dict(self.producer),
            "operation": self.operation.as_dict(),
            "signature": self.signature.as_dict(),
            "entry_point": self.entry_point.as_dict(),
            "files": [item.as_dict() for item in self.files],
            "compatibility": self.compatibility.as_dict(),
            "evidence": self.evidence.as_dict(),
            "promotion": self.promotion.as_dict(),
        }

    @property
    def graph_fingerprint(self) -> str:
        return self.operation.graph_fingerprint

    def digest_for(self, path: str) -> FileDigest | None:
        for item in self.files:
            if item.path == path:
                return item
        return None

    def file_paths(self) -> tuple[str, ...]:
        return tuple(item.path for item in self.files)


def iter_manifest_files(manifest: ArtifactManifest) -> Iterable[FileDigest]:
    """Iterate declared files in a stable order."""
    return sorted(manifest.files, key=lambda item: item.path)
