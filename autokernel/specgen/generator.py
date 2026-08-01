"""Derive safe, benchmarkable KernelSpecs from discovery graph reports."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autokernel.discovery import DiscoveryReport, GraphRegion, load_discovery_report
from autokernel.specs import KernelSpec, Tolerance

from .ir import (
    ALLOWED_TARGETS,
    ExecutableIR,
    IRInput,
    IRNode,
    SpecGenerationError,
    ValueMeta,
    has_unsupported_expr,
)
from .runtime import execute_ir, load_manifest

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_]+")
_LOW_PRECISION = {"bfloat16", "float16"}
_SAFE_FILE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}\.py$")


@dataclass(frozen=True)
class DerivedSubregion:
    """A selected allowlisted component and its parent timing provenance."""

    ir: ExecutableIR
    parent_name: str
    parent_module: str
    parent_fingerprint: str
    parent_cuda_time_us: float
    parent_self_cuda_time_us: float
    parent_calls: int
    selected_node_ids: tuple[str, ...]
    boundary_refs: tuple[str, ...]
    output_node_ids: tuple[str, ...]


def _meta_by_id(ir: ExecutableIR) -> dict[str, ValueMeta]:
    result = {item.id: item.meta for item in ir.inputs}
    result.update({node.id: node.meta for node in ir.nodes if node.meta is not None})
    return result


def _replace_refs(value: Any, aliases: Mapping[str, str]) -> Any:
    if set(value) == {"ref"}:
        return {"ref": aliases.get(value["ref"], value["ref"])}
    for key in ("list", "tuple"):
        if set(value) == {key}:
            return {key: [_replace_refs(item, aliases) for item in value[key]]}
    return value


def derive_safe_subregion(region: GraphRegion) -> DerivedSubregion:
    """Select the best connected allowlisted component from a captured region."""
    attributes = region.attributes or {}
    raw_ir = attributes.get("executable_ir")
    if raw_ir is None:
        raise SpecGenerationError(
            f"region {region.name!r} has no attributes.executable_ir; "
            "operation names alone cannot reconstruct executable semantics"
        )
    ir = ExecutableIR.from_dict(raw_ir)
    allowed = {
        node.id
        for node in ir.nodes
        if node.target in ALLOWED_TARGETS
        and not any(has_unsupported_expr(value) for value in node.args)
        and not any(has_unsupported_expr(value) for value in node.kwargs.values())
    }
    if not allowed:
        raise SpecGenerationError(
            f"region {region.name!r} has no nodes in the executable allowlist"
        )
    if len(ir.nodes) != len(region.operations):
        raise SpecGenerationError(
            f"region {region.name!r} executable_ir has {len(ir.nodes)} nodes "
            f"but operations has {len(region.operations)} entries"
        )
    for index, (node, operation) in enumerate(
        zip(ir.nodes, region.operations, strict=True)
    ):
        if node.target == "operator.getitem":
            expected = "aten::select"
        elif node.target.count(".") >= 2:
            namespace, operation_name, _overload = node.target.split(".", 2)
            expected = f"{namespace}::{operation_name}"
        else:
            expected = node.target
        if expected != operation:
            raise SpecGenerationError(
                f"region {region.name!r} node {index} target {node.target!r} "
                f"does not match operation {operation!r}"
            )

    # A replacement is one call at one point in the parent FX graph. Merely
    # being connected through data dependencies is insufficient: an
    # allowlisted component can span unsupported operations and require a
    # boundary value that is produced after an early selected result is
    # already consumed. Such scattered islands have no legal insertion point.
    # Consecutive allowlisted runs are topologically closed intervals: every
    # external input precedes the run and every external user follows it.
    components: list[set[str]] = []
    current: set[str] = set()
    for node in ir.nodes:
        if node.id in allowed:
            current.add(node.id)
            continue
        if current:
            components.append(current)
            current = set()
    if current:
        components.append(current)

    node_by_id = {node.id: node for node in ir.nodes}
    metadata = _meta_by_id(ir)

    def score(component: set[str]) -> tuple[int, int, int]:
        nodes = [node_by_id[item] for item in component]
        layer_norms = sum("layer_norm" in node.target for node in nodes)
        volume = max(
            (_numel(node.meta.shape) for node in nodes if node.meta is not None),
            default=0,
        )
        return (layer_norms, len(component), volume)

    component = max(components, key=score)
    selected = [node for node in ir.nodes if node.id in component]
    external_refs: list[str] = []
    for node in selected:
        for ref in node.refs():
            if ref not in component and ref not in external_refs:
                external_refs.append(ref)
    missing_meta = [ref for ref in external_refs if ref not in metadata]
    if missing_meta:
        raise SpecGenerationError(
            f"selected component has boundary value(s) without tensor metadata: "
            f"{missing_meta}"
        )

    aliases = {ref: f"boundary_{index}" for index, ref in enumerate(external_refs)}
    selected_inputs = tuple(
        IRInput(
            id=aliases[ref],
            name=f"input_{index}",
            kind="runtime",
            meta=metadata[ref],
        )
        for index, ref in enumerate(external_refs)
    )
    if not selected_inputs:
        raise SpecGenerationError(
            "selected component has no tensor boundary inputs; constant-only "
            "graphs are not optimization candidates"
        )
    selected_nodes = tuple(
        IRNode(
            id=node.id,
            target=node.target,
            args=tuple(_replace_refs(value, aliases) for value in node.args),
            kwargs={
                key: _replace_refs(value, aliases) for key, value in node.kwargs.items()
            },
            meta=node.meta,
        )
        for node in selected
    )

    consumed_inside = {
        ref for node in selected for ref in node.refs() if ref in component
    }
    parent_output_refs = [
        ref for output in ir.outputs for ref in _iter_refs(output) if ref in component
    ]
    external_user_refs = [
        ref
        for node in ir.nodes
        if node.id not in component
        for ref in node.refs()
        if ref in component
    ]
    # A selected value can feed both another selected node and an unselected
    # parent-graph node. Such a value is not a sink inside the component, but
    # it is still a required rewrite output. Omitting it leaves a live external
    # user behind when dispatch erases the selected nodes.
    terminal = list(dict.fromkeys([*parent_output_refs, *external_user_refs]))
    terminal.extend(
        node.id
        for node in selected
        if node.id not in consumed_inside and node.id not in terminal
    )
    if not terminal:
        terminal = [selected[-1].id]
    outputs = tuple({"ref": item} for item in terminal)
    selected_ir = ExecutableIR(
        inputs=selected_inputs,
        nodes=selected_nodes,
        outputs=outputs,
    )
    # Round-trip validation catches producer/generator schema drift.
    selected_ir = ExecutableIR.from_dict(selected_ir.as_dict())
    return DerivedSubregion(
        ir=selected_ir,
        parent_name=region.name,
        parent_module=region.parent_module or region.name,
        parent_fingerprint=region.fingerprint,
        parent_cuda_time_us=region.cuda_time_us,
        parent_self_cuda_time_us=region.self_cuda_time_us,
        parent_calls=region.calls,
        selected_node_ids=tuple(node.id for node in selected),
        boundary_refs=tuple(external_refs),
        output_node_ids=tuple(terminal),
    )


def _iter_refs(value: Any):
    if set(value) == {"ref"}:
        yield value["ref"]
    for key in ("list", "tuple"):
        if set(value) == {key}:
            for item in value[key]:
                yield from _iter_refs(item)


def _numel(shape: Sequence[int]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _dtype_tolerance(dtype: str) -> Tolerance:
    if dtype == "bfloat16":
        return Tolerance(atol=2e-2, rtol=2e-2)
    if dtype == "float16":
        return Tolerance(atol=3e-3, rtol=3e-3)
    return Tolerance(atol=1e-5, rtol=1e-5)


def _model_dtypes(ir: ExecutableIR) -> tuple[str, ...]:
    observed = []
    for item in ir.inputs:
        if item.meta.dtype in _LOW_PRECISION and item.meta.dtype not in observed:
            observed.append(item.meta.dtype)
    return tuple(observed or ["float32"])


def _tensor_dtype(meta_dtype: str, requested: str):
    import torch

    name = requested if meta_dtype in _LOW_PRECISION else meta_dtype
    return getattr(torch, name)


def _input_generator_for(ir: ExecutableIR):
    def generate(
        size_map: Mapping[str, int],
        dtype: Any,
        device: str,
        seed: int = 42,
    ) -> dict[str, Any]:
        import torch

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        requested = str(dtype).removeprefix("torch.")
        result: dict[str, Any] = {}
        for item in ir.inputs:
            tensor_dtype = _tensor_dtype(item.meta.dtype, requested)
            shape = item.meta.shape
            stride = item.meta.stride
            if stride is not None:
                tensor = torch.empty_strided(
                    shape,
                    stride,
                    device=device,
                    dtype=tensor_dtype,
                )
                if tensor_dtype == torch.bool:
                    values = torch.randint(
                        0,
                        2,
                        shape,
                        device=device,
                        generator=generator,
                        dtype=torch.int8,
                    ).bool()
                    tensor.copy_(values)
                elif not tensor_dtype.is_floating_point:
                    values = torch.randint(
                        0,
                        7,
                        shape,
                        device=device,
                        generator=generator,
                        dtype=tensor_dtype,
                    )
                    tensor.copy_(values)
                else:
                    tensor.normal_(generator=generator)
                tensor.requires_grad_(item.meta.requires_grad)
                result[item.name] = tensor
                continue
            if tensor_dtype == torch.bool:
                tensor = torch.randint(
                    0, 2, shape, device=device, generator=generator, dtype=torch.int8
                ).bool()
            elif not tensor_dtype.is_floating_point:
                tensor = torch.randint(
                    0, 7, shape, device=device, generator=generator, dtype=tensor_dtype
                )
            else:
                tensor = torch.randn(
                    shape,
                    device=device,
                    dtype=tensor_dtype,
                    generator=generator,
                    requires_grad=item.meta.requires_grad,
                )
            result[item.name] = tensor
        return result

    return generate


def spec_from_manifest(path: str | Path) -> KernelSpec:
    """Build a KernelSpec from a validated generated manifest."""
    manifest = load_manifest(path)
    ir = ExecutableIR.from_dict(manifest["executable_ir"])
    dtypes = tuple(manifest["dtypes"])

    def reference_fn(**inputs: Any) -> Any:
        return execute_ir(ir, inputs)

    return KernelSpec(
        name=manifest["name"],
        reference_fn=reference_fn,
        input_generator=_input_generator_for(ir),
        sizes={key: dict(value) for key, value in manifest["sizes"].items()},
        dtypes=dtypes,
        tolerances={dtype: _dtype_tolerance(dtype) for dtype in dtypes},
        flops_fn=lambda size: 0,
        bytes_fn=lambda size, dtype_bytes: 0,
        shape_keys=("case",),
        graph_fingerprint=manifest["parent"]["fingerprint"],
        speedup_estimate="unknown; graph-derived starter requires measurement",
    )


def _safe_operation_name(region: GraphRegion) -> str:
    stem = _SAFE_NAME.sub("_", region.name).strip("_").lower()
    if not stem or stem[0].isdigit():
        stem = f"region_{stem}"
    return f"generated_{stem}"[:120]


def build_manifest(region: GraphRegion) -> dict[str, Any]:
    derived = derive_safe_subregion(region)
    dtypes = _model_dtypes(derived.ir)
    # The first implementation preserves the exact observed metadata shape.
    # Additional production shapes are added once the producer supplies
    # per-node metadata for each shape-frequency case.
    sizes = {
        "small": {"case": 1},
        "medium": {"case": 1},
        "large": {"case": 1},
    }
    return {
        "schema_version": 1,
        "name": _safe_operation_name(region),
        "parent": {
            "name": derived.parent_name,
            "module": derived.parent_module,
            "fingerprint": derived.parent_fingerprint,
            "cuda_time_us": derived.parent_cuda_time_us,
            "self_cuda_time_us": derived.parent_self_cuda_time_us,
            "calls": derived.parent_calls,
            "timing_scope": "parent_region_not_selected_subregion",
            "capture_mode": str((region.attributes or {}).get("capture_mode", "")),
        },
        "selected_node_ids": list(derived.selected_node_ids),
        "boundary_refs": list(derived.boundary_refs),
        "output_node_ids": list(derived.output_node_ids),
        "dtypes": list(dtypes),
        "sizes": sizes,
        "executable_ir": derived.ir.as_dict(),
    }


def build_dispatch_contract(manifest_value: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the artifact operation/signature sections from generated IR.

    This is the bridge between graph-derived search and runtime rewriting. It
    preserves only canonical node references and tensor metadata; model
    parameter paths and values never enter the artifact.
    """
    manifest = dict(manifest_value)
    ir = ExecutableIR.from_dict(manifest.get("executable_ir"))
    parent = manifest.get("parent")
    if not isinstance(parent, Mapping):
        raise SpecGenerationError("generated manifest parent must be an object")
    if parent.get("capture_mode") != "export":
        raise SpecGenerationError("subgraph dispatch requires an export capture")
    parent_module = parent.get("module")
    if not isinstance(parent_module, str) or not parent_module:
        raise SpecGenerationError("generated manifest parent module is missing")
    selected = tuple(manifest.get("selected_node_ids") or ())
    boundaries = tuple(manifest.get("boundary_refs") or ())
    outputs = tuple(manifest.get("output_node_ids") or ())
    if not selected or not boundaries or not outputs:
        raise SpecGenerationError("generated manifest has an incomplete rewrite recipe")

    node_by_id = {node.id: node for node in ir.nodes}
    metadata = _meta_by_id(ir)

    def signature(ref: str, index: int, prefix: str) -> dict[str, Any]:
        # Generated boundary inputs use aliases. Their order is the canonical
        # boundary order, so use the corresponding IR input metadata.
        if prefix == "boundary":
            try:
                meta = ir.inputs[index].meta
            except IndexError as exc:
                raise SpecGenerationError("rewrite boundary metadata is missing") from exc
        else:
            meta = metadata.get(ref)
            if meta is None:
                raise SpecGenerationError(f"rewrite output {ref!r} has no tensor metadata")
        if meta.stride is None or meta.device_type is None:
            raise SpecGenerationError(
                f"rewrite {prefix} {ref!r} lacks stride/device metadata"
            )
        return {
            "name": f"{prefix}_{index}",
            "shape": list(meta.shape),
            "stride": list(meta.stride),
            "dtype": meta.dtype,
            "device_type": meta.device_type,
            "requires_grad": meta.requires_grad,
        }

    try:
        operations = [node_by_id[item].target for item in selected]
    except KeyError as exc:
        raise SpecGenerationError("rewrite recipe references an unknown selected node") from exc
    return {
        "operation": {
            "name": str(manifest.get("name", "")),
            "graph_fingerprint": str(parent.get("fingerprint", "")),
            "parent_module": parent_module,
            "operations": operations,
            "target_kind": "subgraph",
            "capture_mode": "export",
            "selected_node_ids": list(selected),
            "boundary_refs": list(boundaries),
            "output_node_ids": list(outputs),
        },
        "signature": {
            "inputs": [
                signature(ref, index, "boundary")
                for index, ref in enumerate(boundaries)
            ],
            "outputs": [
                signature(ref, index, "output")
                for index, ref in enumerate(outputs)
            ],
        },
    }


def write_runtime_adapter(
    manifest_value: Mapping[str, Any],
    output_path: str | Path,
    *,
    candidate_file: str = "candidate.py",
    candidate_symbol: str = "kernel_fn",
    entry_symbol: str = "fused_subgraph",
) -> Path:
    """Write the positional runtime adapter for a generated search candidate."""
    manifest = dict(manifest_value)
    ir = ExecutableIR.from_dict(manifest.get("executable_ir"))
    build_dispatch_contract(manifest)
    if not _SAFE_FILE.fullmatch(candidate_file):
        raise SpecGenerationError("candidate_file must be a safe Python basename")
    for value, name in (
        (candidate_symbol, "candidate_symbol"),
        (entry_symbol, "entry_symbol"),
    ):
        if not value.isidentifier():
            raise SpecGenerationError(f"{name} must be a Python identifier")
    input_names = tuple(item.name for item in ir.inputs)
    source = f'''"""Generated positional adapter for a graph-derived candidate."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_CANDIDATE_PATH = Path(__file__).with_name({candidate_file!r})
_SPEC = importlib.util.spec_from_file_location(
    "autokernel_runtime_candidate", _CANDIDATE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load candidate {{_CANDIDATE_PATH}}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_KERNEL = getattr(_MODULE, {candidate_symbol!r})
_INPUT_NAMES = {input_names!r}


def {entry_symbol}(module, *values):
    del module
    if len(values) != len(_INPUT_NAMES):
        raise TypeError(
            f"expected {{len(_INPUT_NAMES)}} boundary tensors, got {{len(values)}}"
        )
    return _KERNEL(**dict(zip(_INPUT_NAMES, values, strict=True)))
'''
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source, encoding="utf-8")
    return output
def select_region(
    report: DiscoveryReport, fingerprint: str | None = None
) -> GraphRegion:
    if fingerprint is not None:
        matches = [item for item in report.regions if item.fingerprint == fingerprint]
        if not matches:
            raise SpecGenerationError(
                f"no region with fingerprint {fingerprint!r} in report"
            )
        return matches[0]
    candidates = [
        item
        for item in report.regions
        if item.attributes and item.attributes.get("executable_ir") is not None
    ]
    if not candidates:
        raise SpecGenerationError("report has no region with executable_ir metadata")
    return max(candidates, key=lambda item: (item.self_cuda_time_us, item.cuda_time_us))


_SPEC_TEMPLATE = '''"""Generated graph-derived KernelSpec. Do not edit by hand."""\n\
from pathlib import Path\n\
from autokernel.specgen import spec_from_manifest\n\
SPEC = spec_from_manifest(Path(__file__).with_name("manifest.json"))\n'''

_KERNEL_TEMPLATE = '''"""Correct eager starter for a generated graph subregion."""\n\
from pathlib import Path\n\
from autokernel.specgen import load_generated_reference\n\
kernel_fn = load_generated_reference(Path(__file__).with_name("manifest.json"))\n'''


def write_generated_artifacts(
    report_path: str | Path,
    output_dir: str | Path,
    *,
    fingerprint: str | None = None,
) -> dict[str, Path]:
    report = load_discovery_report(report_path)
    region = select_region(report, fingerprint)
    manifest = build_manifest(region)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    spec_path = output / "spec.py"
    kernel_path = output / "kernel.py"
    corpus_path = output / "corpus.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    spec_path.write_text(_SPEC_TEMPLATE, encoding="utf-8")
    kernel_path.write_text(_KERNEL_TEMPLATE, encoding="utf-8")
    corpus = {
        "schema_version": 1,
        "operation": manifest["name"],
        "cases": [
            {
                "name": "observed",
                "size": {"case": 1},
                "dtype": manifest["dtypes"][0],
                "weight": max(1, region.calls),
                "tags": ["production", "graph-derived"],
            }
        ],
    }
    corpus_path.write_text(
        json.dumps(corpus, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Validate the just-written executable artifact before returning it.
    spec_from_manifest(manifest_path)
    return {
        "manifest": manifest_path,
        "spec": spec_path,
        "kernel": kernel_path,
        "corpus": corpus_path,
    }
