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


@dataclass(frozen=True)
class DerivedSubregion:
    """A selected allowlisted component and its parent timing provenance."""

    ir: ExecutableIR
    parent_name: str
    parent_fingerprint: str
    parent_cuda_time_us: float
    parent_self_cuda_time_us: float
    parent_calls: int
    selected_node_ids: tuple[str, ...]


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
        expected = (
            "aten::select"
            if node.target == "operator.getitem"
            else "aten::" + node.target.split(".", 2)[1]
            if node.target.startswith("aten.")
            else node.target
        )
        if expected != operation:
            raise SpecGenerationError(
                f"region {region.name!r} node {index} target {node.target!r} "
                f"does not match operation {operation!r}"
            )

    # Connected components over data dependencies among allowlisted nodes.
    adjacency = {node_id: set() for node_id in allowed}
    for node in ir.nodes:
        if node.id not in allowed:
            continue
        for ref in node.refs():
            if ref in allowed:
                adjacency[node.id].add(ref)
                adjacency[ref].add(node.id)
    components: list[set[str]] = []
    unseen = set(allowed)
    while unseen:
        root = min(unseen)
        stack = [root]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current] - component)
        unseen -= component
        components.append(component)

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
    terminal = list(dict.fromkeys(parent_output_refs))
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
        parent_fingerprint=region.fingerprint,
        parent_cuda_time_us=region.cuda_time_us,
        parent_self_cuda_time_us=region.self_cuda_time_us,
        parent_calls=region.calls,
        selected_node_ids=tuple(node.id for node in selected),
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
            "fingerprint": derived.parent_fingerprint,
            "cuda_time_us": derived.parent_cuda_time_us,
            "self_cuda_time_us": derived.parent_self_cuda_time_us,
            "calls": derived.parent_calls,
            "timing_scope": "parent_region_not_selected_subregion",
        },
        "selected_node_ids": list(derived.selected_node_ids),
        "dtypes": list(dtypes),
        "sizes": sizes,
        "executable_ir": derived.ir.as_dict(),
    }


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
