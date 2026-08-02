"""Runtime for validated graph-derived specifications."""

from __future__ import annotations

import json
import operator
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .ir import ALLOWED_TARGETS, ExecutableIR, SpecGenerationError


def _torch_target(name: str) -> Any:
    if name == "operator.getitem":
        return operator.getitem
    if name not in ALLOWED_TARGETS or not name.startswith("aten."):
        raise SpecGenerationError(f"target {name!r} is not executable")
    import torch

    parts = name.split(".")
    try:
        packet = getattr(torch.ops.aten, parts[1])
        return getattr(packet, ".".join(parts[2:]))
    except AttributeError as exc:
        raise SpecGenerationError(
            f"allowlisted target {name!r} is unavailable in this PyTorch build"
        ) from exc


def _evaluate(value: Any, env: Mapping[str, Any]) -> Any:
    if set(value) == {"ref"}:
        return env[value["ref"]]
    if set(value) == {"const"}:
        return value["const"]
    if set(value) == {"dtype"}:
        import torch

        return getattr(torch, value["dtype"])
    if set(value) == {"device"}:
        for item in env.values():
            if hasattr(item, "device"):
                return item.device
        raise SpecGenerationError("runtime device expression has no tensor input")
    if set(value) == {"list"}:
        return [_evaluate(item, env) for item in value["list"]]
    if set(value) == {"tuple"}:
        return tuple(_evaluate(item, env) for item in value["tuple"])
    if set(value) == {"unsupported"}:
        raise SpecGenerationError("unsupported expression reached runtime")
    raise SpecGenerationError("unvalidated argument expression reached runtime")


def execute_ir(ir: ExecutableIR, inputs: Mapping[str, Any]) -> Any:
    """Execute a previously validated, allowlisted IR."""
    expected = {item.name for item in ir.inputs}
    if set(inputs) != expected:
        raise SpecGenerationError(
            f"generated reference expected inputs {sorted(expected)}, "
            f"received {sorted(inputs)}"
        )
    env = {item.id: inputs[item.name] for item in ir.inputs}
    for node in ir.nodes:
        if node.target not in ALLOWED_TARGETS:
            raise SpecGenerationError(f"target {node.target!r} is not executable")
        fn = _torch_target(node.target)
        args = [_evaluate(value, env) for value in node.args]
        kwargs = {key: _evaluate(value, env) for key, value in node.kwargs.items()}
        env[node.id] = fn(*args, **kwargs)
    outputs = tuple(_evaluate(value, env) for value in ir.outputs)
    return outputs[0] if len(outputs) == 1 else outputs


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Read JSON and validate the embedded executable IR before returning it."""
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpecGenerationError(
            f"cannot read generated manifest {source}: {exc}"
        ) from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise SpecGenerationError("generated manifest must use schema_version 1")
    ExecutableIR.from_dict(raw.get("executable_ir"))
    return raw


def load_generated_reference(path: str | Path):
    """Create a reference callable from a generated manifest path."""
    manifest = load_manifest(path)
    ir = ExecutableIR.from_dict(manifest["executable_ir"])

    def reference_fn(**inputs: Any) -> Any:
        return execute_ir(ir, inputs)

    return reference_fn
