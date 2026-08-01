"""Strict metadata-only executable IR for graph-derived kernel specs.

The discovery ``operations`` list is useful for ranking, but it cannot encode
operand order, constants, or outputs.  This module accepts a deliberately
small JSON IR that carries those facts without source code or tensor values.
Only the allowlisted targets in :data:`ALLOWED_TARGETS` may be executed.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

IR_SCHEMA_VERSION = 1
_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
_DTYPES = {
    "bool",
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
}

# Exact overload names are intentional. Adding an entry expands the executable
# trust boundary and must come with parity tests.
ALLOWED_TARGETS = frozenset(
    {
        "aten.add.Scalar",
        "aten.add.Tensor",
        "aten.sub.Scalar",
        "aten.sub.Tensor",
        "aten.mul.Scalar",
        "aten.mul.Tensor",
        "aten.div.Scalar",
        "aten.div.Tensor",
        "aten.neg.default",
        "aten.pow.Scalar",
        "aten.pow.Tensor_Scalar",
        "aten.mean.dim",
        "aten.rsqrt.default",
        "aten.silu.default",
        "aten.gelu.default",
        "aten.native_layer_norm.default",
        "aten.layer_norm.default",
        "aten._to_copy.default",
        "aten.to.dtype",
        "aten.type_as.default",
        "aten.reshape.default",
        "aten.view.default",
        "aten.flatten.using_ints",
        "aten.unflatten.int",
        "aten.transpose.int",
        "aten.permute.default",
        "aten.unsqueeze.default",
        "aten.squeeze.default",
        "aten.squeeze.dim",
        "aten.expand.default",
        "aten.select.int",
        "aten.slice.Tensor",
        "aten.chunk.default",
        "aten.unbind.int",
        "operator.getitem",
    }
)


class SpecGenerationError(ValueError):
    """A graph report cannot safely produce an executable specification."""


def _fail(location: str, message: str) -> SpecGenerationError:
    return SpecGenerationError(f"executable IR {location}: {message}")


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(location, "must be an object")
    for key in value:
        if not isinstance(key, str) or not key:
            raise _fail(location, "keys must be non-empty strings")
    return value


def _keys(raw: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise _fail(location, f"unknown field(s) {unknown}")


def _identifier(value: Any, location: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise _fail(location, "must be an identifier-like string")
    return value


def _shape(value: Any, location: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _fail(location, "must be a list")
    result = []
    for index, dim in enumerate(value):
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            raise _fail(f"{location}[{index}]", "must be a non-negative integer")
        result.append(dim)
    return tuple(result)


@dataclass(frozen=True)
class ValueMeta:
    """Tensor metadata attached to an input or node output."""

    shape: tuple[int, ...]
    dtype: str
    requires_grad: bool = False
    stride: tuple[int, ...] | None = None
    device_type: str | None = None

    @classmethod
    def from_dict(cls, value: Any, location: str) -> ValueMeta:
        raw = _mapping(value, location)
        _keys(
            raw,
            {"shape", "dtype", "requires_grad", "stride", "device_type"},
            location,
        )
        dtype = raw.get("dtype")
        if not isinstance(dtype, str) or dtype not in _DTYPES:
            raise _fail(f"{location}.dtype", f"unsupported dtype {dtype!r}")
        requires_grad = raw.get("requires_grad", False)
        if not isinstance(requires_grad, bool):
            raise _fail(f"{location}.requires_grad", "must be a bool")
        shape = _shape(raw.get("shape"), f"{location}.shape")
        stride_value = raw.get("stride")
        stride = None
        if stride_value is not None:
            stride = _shape(stride_value, f"{location}.stride")
            if len(stride) != len(shape):
                raise _fail(
                    f"{location}.stride", "must have the same length as shape"
                )
        device_type = raw.get("device_type")
        if device_type is not None and (
            not isinstance(device_type, str) or not device_type
        ):
            raise _fail(f"{location}.device_type", "must be a non-empty string")
        return cls(
            shape=shape,
            dtype=dtype,
            requires_grad=requires_grad,
            stride=stride,
            device_type=device_type,
        )

    def as_dict(self) -> dict[str, Any]:
        result = {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "requires_grad": self.requires_grad,
        }
        if self.stride is not None:
            result["stride"] = list(self.stride)
        if self.device_type is not None:
            result["device_type"] = self.device_type
        return result


def validate_expr(value: Any, location: str) -> None:
    """Validate one JSON argument expression without evaluating it."""
    if isinstance(value, Mapping):
        raw = _mapping(value, location)
        if set(raw) == {"ref"}:
            _identifier(raw["ref"], f"{location}.ref")
            return
        if set(raw) == {"const"}:
            constant = raw["const"]
            if constant is None or isinstance(constant, (str, bool, int)):
                return
            if isinstance(constant, float) and math.isfinite(constant):
                return
            raise _fail(f"{location}.const", "must be a finite JSON primitive")
        if set(raw) == {"dtype"}:
            if raw["dtype"] not in _DTYPES:
                raise _fail(f"{location}.dtype", "unsupported dtype")
            return
        if set(raw) == {"device"}:
            if raw["device"] != "runtime":
                raise _fail(f"{location}.device", "must be 'runtime'")
            return
        if set(raw) == {"unsupported"}:
            _identifier(raw["unsupported"], f"{location}.unsupported")
            return
        if set(raw) in ({"list"}, {"tuple"}):
            key = next(iter(raw))
            items = raw[key]
            if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
                raise _fail(f"{location}.{key}", "must be a list")
            for index, item in enumerate(items):
                validate_expr(item, f"{location}.{key}[{index}]")
            return
        raise _fail(
            location,
            "must be exactly one of ref, const, dtype, device, list, tuple, "
            "or unsupported",
        )
    raise _fail(location, "must be an expression object")


def iter_refs(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        if set(value) == {"ref"}:
            yield value["ref"]
            return
        for key in ("list", "tuple"):
            if set(value) == {key}:
                for item in value[key]:
                    yield from iter_refs(item)


def has_unsupported_expr(value: Any) -> bool:
    if isinstance(value, Mapping):
        if set(value) == {"unsupported"}:
            return True
        for key in ("list", "tuple"):
            if set(value) == {key}:
                return any(has_unsupported_expr(item) for item in value[key])
    return False


@dataclass(frozen=True)
class IRInput:
    id: str
    name: str
    kind: str
    meta: ValueMeta

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "meta": self.meta.as_dict(),
        }


@dataclass(frozen=True)
class IRNode:
    id: str
    target: str
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]
    meta: ValueMeta | None

    def refs(self) -> tuple[str, ...]:
        found: list[str] = []
        for value in self.args:
            found.extend(iter_refs(value))
        for value in self.kwargs.values():
            found.extend(iter_refs(value))
        return tuple(found)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "target": self.target,
            "args": list(self.args),
            "kwargs": dict(self.kwargs),
        }
        if self.meta is not None:
            payload["meta"] = self.meta.as_dict()
        return payload


@dataclass(frozen=True)
class ExecutableIR:
    inputs: tuple[IRInput, ...]
    nodes: tuple[IRNode, ...]
    outputs: tuple[Any, ...]
    schema_version: int = IR_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, value: Any) -> ExecutableIR:
        raw = _mapping(value, "top level")
        _keys(raw, {"schema_version", "inputs", "nodes", "outputs"}, "top level")
        if raw.get("schema_version") != IR_SCHEMA_VERSION:
            raise _fail(
                "top level.schema_version",
                f"must equal {IR_SCHEMA_VERSION}",
            )
        raw_inputs = raw.get("inputs")
        raw_nodes = raw.get("nodes")
        raw_outputs = raw.get("outputs")
        if not isinstance(raw_inputs, list) or not raw_inputs:
            raise _fail("top level.inputs", "must be a non-empty list")
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise _fail("top level.nodes", "must be a non-empty list")
        if not isinstance(raw_outputs, list) or not raw_outputs:
            raise _fail("top level.outputs", "must be a non-empty list")

        inputs: list[IRInput] = []
        identifiers: set[str] = set()
        for index, item in enumerate(raw_inputs):
            location = f"top level.inputs[{index}]"
            item_raw = _mapping(item, location)
            _keys(item_raw, {"id", "name", "kind", "meta"}, location)
            item_id = _identifier(item_raw.get("id"), f"{location}.id")
            if item_id in identifiers:
                raise _fail(f"{location}.id", f"duplicate id {item_id!r}")
            identifiers.add(item_id)
            kind = item_raw.get("kind")
            if kind not in {"runtime", "lifted"}:
                raise _fail(f"{location}.kind", "must be 'runtime' or 'lifted'")
            name = _identifier(item_raw.get("name"), f"{location}.name")
            inputs.append(
                IRInput(
                    id=item_id,
                    name=name,
                    kind=kind,
                    meta=ValueMeta.from_dict(item_raw.get("meta"), f"{location}.meta"),
                )
            )

        nodes: list[IRNode] = []
        for index, item in enumerate(raw_nodes):
            location = f"top level.nodes[{index}]"
            item_raw = _mapping(item, location)
            _keys(item_raw, {"id", "target", "args", "kwargs", "meta"}, location)
            item_id = _identifier(item_raw.get("id"), f"{location}.id")
            if item_id in identifiers:
                raise _fail(f"{location}.id", f"duplicate id {item_id!r}")
            target = item_raw.get("target")
            if not isinstance(target, str) or not target:
                raise _fail(f"{location}.target", "must be a non-empty string")
            args = item_raw.get("args")
            kwargs = item_raw.get("kwargs")
            if not isinstance(args, list):
                raise _fail(f"{location}.args", "must be a list")
            if not isinstance(kwargs, Mapping):
                raise _fail(f"{location}.kwargs", "must be an object")
            for arg_index, arg in enumerate(args):
                validate_expr(arg, f"{location}.args[{arg_index}]")
            for key, arg in kwargs.items():
                _identifier(key, f"{location}.kwargs key")
                validate_expr(arg, f"{location}.kwargs.{key}")
            node = IRNode(
                id=item_id,
                target=target,
                args=tuple(args),
                kwargs=dict(kwargs),
                meta=(
                    ValueMeta.from_dict(item_raw["meta"], f"{location}.meta")
                    if "meta" in item_raw
                    else None
                ),
            )
            missing = sorted(set(node.refs()) - identifiers)
            if missing:
                raise _fail(location, f"references unknown or forward id(s) {missing}")
            identifiers.add(item_id)
            nodes.append(node)

        outputs = []
        for index, output in enumerate(raw_outputs):
            validate_expr(output, f"top level.outputs[{index}]")
            missing = sorted(set(iter_refs(output)) - identifiers)
            if missing:
                raise _fail(
                    f"top level.outputs[{index}]",
                    f"references unknown id(s) {missing}",
                )
            outputs.append(output)
        return cls(inputs=tuple(inputs), nodes=tuple(nodes), outputs=tuple(outputs))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "inputs": [item.as_dict() for item in self.inputs],
            "nodes": [item.as_dict() for item in self.nodes],
            "outputs": list(self.outputs),
        }
