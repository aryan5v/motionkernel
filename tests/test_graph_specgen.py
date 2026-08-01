"""Tests for graph-derived specification generation."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from autokernel.discovery import (
    DiscoveryReport,
    GraphRegion,
    TensorMeta,
    write_discovery_report,
)
from autokernel.specgen import (
    ExecutableIR,
    SpecGenerationError,
    build_manifest,
    derive_safe_subregion,
    execute_ir,
    spec_from_manifest,
    write_generated_artifacts,
)
from autokernel.specs import KernelSpec, SpecValidationError
from conftest import spec_kwargs


@pytest.mark.parametrize(
    "fingerprint",
    (
        "",
        "0" * 31,
        "0" * 33,
        "G" * 32,
        "ABCDEF0123456789ABCDEF0123456789",
    ),
)
def test_kernel_spec_rejects_invalid_graph_fingerprint(fingerprint: str) -> None:
    with pytest.raises(SpecValidationError, match="graph_fingerprint"):
        KernelSpec(**spec_kwargs(graph_fingerprint=fingerprint))


def test_kernel_spec_accepts_graph_fingerprint() -> None:
    fingerprint = "0123456789abcdef0123456789abcdef"
    spec = KernelSpec(**spec_kwargs(graph_fingerprint=fingerprint))
    assert spec.graph_fingerprint == fingerprint


def _meta(shape: tuple[int, ...], dtype: str = "float32") -> dict:
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= max(dim, 1)
    return {
        "shape": list(shape),
        "stride": list(reversed(stride)),
        "dtype": dtype,
        "device_type": "cuda",
        "requires_grad": False,
    }


def _input(name: str, shape: tuple[int, ...], dtype: str = "float32") -> dict:
    return {"id": name, "name": name, "kind": "runtime", "meta": _meta(shape, dtype)}


def _ref(name: str) -> dict:
    return {"ref": name}


def _const(value) -> dict:
    return {"const": value}


def _dtype(name: str) -> dict:
    return {"dtype": name}


def _node(
    node_id: str,
    target: str,
    args: list,
    *,
    kwargs: dict | None = None,
    meta: dict | None = None,
) -> dict:
    result = {
        "id": node_id,
        "target": target,
        "args": args,
        "kwargs": kwargs or {},
    }
    if meta is not None:
        result["meta"] = meta
    return result


def _ir(inputs: list[dict], nodes: list[dict], outputs: list[dict]) -> ExecutableIR:
    return ExecutableIR.from_dict(
        {
            "schema_version": 1,
            "inputs": inputs,
            "nodes": nodes,
            "outputs": outputs,
        }
    )


def test_executable_ir_rejects_forward_reference() -> None:
    with pytest.raises(SpecGenerationError, match="forward id"):
        _ir(
            [_input("x", (2, 3))],
            [
                _node(
                    "add",
                    "aten.add.Tensor",
                    [_ref("x"), _ref("future")],
                    meta=_meta((2, 3)),
                ),
                _node(
                    "future",
                    "aten.neg.default",
                    [_ref("x")],
                    meta=_meta((2, 3)),
                ),
            ],
            [_ref("add")],
        )


def test_runtime_rejects_non_allowlisted_target(torch_mod) -> None:
    ir = _ir(
        [_input("x", (2, 3))],
        [_node("bad", "aten.mm.default", [_ref("x"), _ref("x")])],
        [_ref("bad")],
    )
    with pytest.raises(SpecGenerationError, match="not executable"):
        execute_ir(ir, {"x": torch_mod.randn(2, 3)})


def test_ltx_rmsnorm_silu_chain_is_allowlisted_and_matches_eager(torch_mod) -> None:
    """Exact overloads observed in the LTX transformer remain executable."""
    meta = _meta((2, 4))
    ir = _ir(
        [_input("x", (2, 4))],
        [
            _node(
                "square",
                "aten.pow.Tensor_Scalar",
                [_ref("x"), _const(2)],
                meta=meta,
            ),
            _node(
                "mean",
                "aten.mean.dim",
                [_ref("square"), {"list": [_const(-1)]}, _const(True)],
                meta=_meta((2, 1)),
            ),
            _node(
                "epsilon",
                "aten.add.Tensor",
                [_ref("mean"), _const(1e-6)],
                meta=_meta((2, 1)),
            ),
            _node(
                "inverse_rms",
                "aten.rsqrt.default",
                [_ref("epsilon")],
                meta=_meta((2, 1)),
            ),
            _node(
                "normalized",
                "aten.mul.Tensor",
                [_ref("x"), _ref("inverse_rms")],
                meta=meta,
            ),
            _node(
                "activated",
                "aten.silu.default",
                [_ref("normalized")],
                meta=meta,
            ),
        ],
        [_ref("activated")],
    )
    x = torch_mod.randn(2, 4)

    actual = execute_ir(ir, {"x": x})
    expected = torch_mod.nn.functional.silu(
        x * torch_mod.rsqrt(torch_mod.mean(x.pow(2), dim=-1, keepdim=True) + 1e-6)
    )

    torch_mod.testing.assert_close(actual, expected)
    derived = derive_safe_subregion(_region_for(ir, _operations_for(ir)))
    assert [node.target for node in derived.ir.nodes] == [
        "aten.pow.Tensor_Scalar",
        "aten.mean.dim",
        "aten.add.Tensor",
        "aten.rsqrt.default",
        "aten.mul.Tensor",
        "aten.silu.default",
    ]


def test_ltx_gelu_overload_is_allowlisted_and_matches_eager(torch_mod) -> None:
    ir = _ir(
        [_input("x", (2, 4))],
        [
            _node(
                "activated",
                "aten.gelu.default",
                [_ref("x")],
                kwargs={"approximate": _const("tanh")},
                meta=_meta((2, 4)),
            )
        ],
        [_ref("activated")],
    )
    x = torch_mod.randn(2, 4)

    actual = execute_ir(ir, {"x": x})
    expected = torch_mod.nn.functional.gelu(x, approximate="tanh")

    torch_mod.testing.assert_close(actual, expected)
    derived = derive_safe_subregion(_region_for(ir, _operations_for(ir)))
    assert [node.target for node in derived.ir.nodes] == ["aten.gelu.default"]


def _region_for(ir: ExecutableIR, operations: list[str]) -> GraphRegion:
    return GraphRegion.build(
        name="wan_transformer_blocks_shape",
        operations=operations,
        inputs=(
            TensorMeta(
                name="input_0",
                shape=(2, 3),
                stride=(3, 1),
                dtype="float32",
                device_type="cuda",
            ),
        ),
        parent_module="transformer.blocks",
        cuda_time_us=1000.0,
        self_cuda_time_us=900.0,
        calls=240,
        attributes={"executable_ir": ir.as_dict(), "capture_mode": "export"},
    )


def _operations_for(ir: ExecutableIR) -> list[str]:
    operations = []
    for node in ir.nodes:
        if node.target == "operator.getitem":
            operations.append("aten::select")
        elif node.target.count(".") >= 2:
            namespace, operation_name, _overload = node.target.split(".", 2)
            operations.append(f"{namespace}::{operation_name}")
        else:
            operations.append(node.target)
    return operations


def test_derivation_isolates_allowlisted_component_with_explicit_boundary() -> None:
    ir = _ir(
        [_input("x", (2, 3)), _input("weight", (3, 4))],
        [
            _node(
                "mm",
                "aten.mm.default",
                [_ref("x"), _ref("weight")],
                meta=_meta((2, 4)),
            ),
            _node(
                "scale",
                "aten.mul.Scalar",
                [_ref("mm"), _const(0.5)],
                meta=_meta((2, 4)),
            ),
            _node(
                "negative",
                "aten.neg.default",
                [_ref("scale")],
                meta=_meta((2, 4)),
            ),
        ],
        [_ref("negative")],
    )
    derived = derive_safe_subregion(
        _region_for(ir, ["aten::mm", "aten::mul", "aten::neg"])
    )
    assert [node.target for node in derived.ir.nodes] == [
        "aten.mul.Scalar",
        "aten.neg.default",
    ]
    assert [item.name for item in derived.ir.inputs] == ["input_0"]
    assert derived.ir.inputs[0].meta.shape == (2, 4)
    assert derived.parent_cuda_time_us == 1000.0
    assert derived.parent_module == "transformer.blocks"


def test_derivation_returns_selected_values_used_outside_component() -> None:
    ir = _ir(
        [_input("x", (2, 3)), _input("weight", (3, 3))],
        [
            _node(
                "negative",
                "aten.neg.default",
                [_ref("x")],
                meta=_meta((2, 3)),
            ),
            _node(
                "updated",
                "aten.add.Tensor",
                [_ref("negative"), _ref("x")],
                meta=_meta((2, 3)),
            ),
            _node(
                "unsupported",
                "aten.mm.default",
                [_ref("negative"), _ref("weight")],
                meta=_meta((2, 3)),
            ),
        ],
        [_ref("updated"), _ref("unsupported")],
    )

    derived = derive_safe_subregion(
        _region_for(ir, ["aten::neg", "aten::add", "aten::mm"])
    )

    assert derived.selected_node_ids == ("negative", "updated")
    assert derived.output_node_ids == ("updated", "negative")


def test_derivation_does_not_span_an_unsupported_topological_gap() -> None:
    ir = _ir(
        [_input("x", (2, 3)), _input("weight", (3, 3))],
        [
            _node(
                "early",
                "aten.neg.default",
                [_ref("x")],
                meta=_meta((2, 3)),
            ),
            _node(
                "gap",
                "aten.mm.default",
                [_ref("early"), _ref("weight")],
                meta=_meta((2, 3)),
            ),
            _node(
                "late",
                "aten.add.Tensor",
                [_ref("early"), _ref("gap")],
                meta=_meta((2, 3)),
            ),
        ],
        [_ref("late")],
    )

    derived = derive_safe_subregion(
        _region_for(ir, ["aten::neg", "aten::mm", "aten::add"])
    )

    assert derived.selected_node_ids == ("early",)
    assert derived.boundary_refs == ("x",)
    assert derived.output_node_ids == ("early",)


def test_derivation_preserves_custom_op_identity_but_excludes_it() -> None:
    ir = _ir(
        [_input("x", (2, 3))],
        [
            _node(
                "attention",
                "fastvideo._flash_attn_default_forward.default",
                [_ref("x")],
                meta=_meta((2, 3)),
            ),
            _node(
                "negative",
                "aten.neg.default",
                [_ref("attention")],
                meta=_meta((2, 3)),
            ),
        ],
        [_ref("negative")],
    )

    derived = derive_safe_subregion(_region_for(ir, _operations_for(ir)))

    assert [node.target for node in derived.ir.nodes] == ["aten.neg.default"]
    assert derived.boundary_refs == ("attention",)


def _to_fp32(node_id: str, source: str, shape: tuple[int, ...]) -> dict:
    return _node(
        node_id,
        "aten._to_copy.default",
        [_ref(source)],
        kwargs={"dtype": _dtype("float32")},
        meta=_meta(shape),
    )


def _to_bf16(node_id: str, source: str, shape: tuple[int, ...]) -> dict:
    return _node(
        node_id,
        "aten._to_copy.default",
        [_ref(source)],
        kwargs={"dtype": _dtype("bfloat16")},
        meta=_meta(shape, "bfloat16"),
    )


def _gated_residual_ir() -> ExecutableIR:
    activation = (2, 5, 7)
    vector = (2, 7)
    return _ir(
        [
            _input("residual", activation, "bfloat16"),
            _input("x", activation, "bfloat16"),
            _input("gate", vector),
        ],
        [
            _to_fp32("residual_fp32", "residual", activation),
            _to_fp32("x_fp32", "x", activation),
            _node(
                "gate_unsqueeze",
                "aten.unsqueeze.default",
                [_ref("gate"), _const(1)],
                meta=_meta((2, 1, 7)),
            ),
            _node(
                "gated",
                "aten.mul.Tensor",
                [_ref("x_fp32"), _ref("gate_unsqueeze")],
                meta=_meta(activation),
            ),
            _node(
                "updated",
                "aten.add.Tensor",
                [_ref("residual_fp32"), _ref("gated")],
                meta=_meta(activation),
            ),
            _to_bf16("output", "updated", activation),
        ],
        [_ref("output")],
    )


def _modulated_layer_norm_ir() -> ExecutableIR:
    activation = (2, 5, 7)
    vector = (2, 7)
    return _ir(
        [
            _input("x", activation, "bfloat16"),
            _input("scale", vector),
            _input("shift", vector),
        ],
        [
            _to_fp32("x_fp32", "x", activation),
            _node(
                "norm_tuple",
                "aten.native_layer_norm.default",
                [
                    _ref("x_fp32"),
                    {"list": [_const(7)]},
                    _const(None),
                    _const(None),
                    _const(1e-6),
                ],
            ),
            _node(
                "normalized",
                "operator.getitem",
                [_ref("norm_tuple"), _const(0)],
                meta=_meta(activation),
            ),
            _node(
                "scale_u",
                "aten.unsqueeze.default",
                [_ref("scale"), _const(1)],
                meta=_meta((2, 1, 7)),
            ),
            _node(
                "scale_plus_one",
                "aten.add.Scalar",
                [_ref("scale_u"), _const(1.0)],
                meta=_meta((2, 1, 7)),
            ),
            _node(
                "scaled",
                "aten.mul.Tensor",
                [_ref("normalized"), _ref("scale_plus_one")],
                meta=_meta(activation),
            ),
            _node(
                "shift_u",
                "aten.unsqueeze.default",
                [_ref("shift"), _const(1)],
                meta=_meta((2, 1, 7)),
            ),
            _node(
                "shifted",
                "aten.add.Tensor",
                [_ref("scaled"), _ref("shift_u")],
                meta=_meta(activation),
            ),
            _to_bf16("output", "shifted", activation),
        ],
        [_ref("output")],
    )


def _gated_residual_norm_ir() -> ExecutableIR:
    base = _gated_residual_ir().as_dict()
    activation = (2, 5, 7)
    base["inputs"].extend([_input("weight_ln", (7,)), _input("bias_ln", (7,))])
    base["nodes"][-1] = _to_bf16("updated_output", "updated", activation)
    base["nodes"].extend(
        [
            _node(
                "norm_tuple",
                "aten.native_layer_norm.default",
                [
                    _ref("updated"),
                    {"list": [_const(7)]},
                    _ref("weight_ln"),
                    _ref("bias_ln"),
                    _const(1e-6),
                ],
            ),
            _node(
                "normalized",
                "operator.getitem",
                [_ref("norm_tuple"), _const(0)],
                meta=_meta(activation),
            ),
            _to_bf16("norm_output", "normalized", activation),
        ]
    )
    base["outputs"] = [_ref("norm_output"), _ref("updated_output")]
    return ExecutableIR.from_dict(base)


@pytest.mark.parametrize(
    ("ir_factory", "reference_name"),
    (
        (_gated_residual_ir, "wan_gated_residual_ref"),
        (_modulated_layer_norm_ir, "wan_modulated_layer_norm_ref"),
        (_gated_residual_norm_ir, "wan_gated_residual_norm_ref"),
    ),
)
def test_generated_ir_matches_handwritten_wan_reference(
    ir_factory, reference_name: str, torch_mod, tmp_path
) -> None:
    from models import (
        wan_gated_residual,
        wan_gated_residual_norm,
        wan_modulated_layer_norm,
    )

    references = {
        "wan_gated_residual_ref": wan_gated_residual.wan_gated_residual_ref,
        "wan_modulated_layer_norm_ref": (
            wan_modulated_layer_norm.wan_modulated_layer_norm_ref
        ),
        "wan_gated_residual_norm_ref": (
            wan_gated_residual_norm.wan_gated_residual_norm_ref
        ),
    }
    ir = ir_factory()
    region = _region_for(ir, _operations_for(ir))
    manifest = build_manifest(region)
    manifest_path = tmp_path / f"{reference_name}.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    spec = spec_from_manifest(manifest_path)
    generator = torch_mod.Generator().manual_seed(123)
    original_inputs = {
        item.name: torch_mod.randn(
            item.meta.shape,
            dtype=getattr(torch_mod, item.meta.dtype),
            generator=generator,
        )
        for item in ir.inputs
    }
    generated_ir = ExecutableIR.from_dict(manifest["executable_ir"])
    generated_inputs = {
        generated.name: original_inputs[original.name]
        for generated, original in zip(
            generated_ir.inputs,
            ir.inputs,
            strict=True,
        )
    }
    actual = spec.reference_fn(**generated_inputs)
    expected = references[reference_name](
        **{
            {
                "weight_ln": "weight",
                "bias_ln": "bias",
            }.get(key, key): value
            for key, value in original_inputs.items()
        }
    )
    actual_items = actual if isinstance(actual, tuple) else (actual,)
    expected_items = expected if isinstance(expected, tuple) else (expected,)
    assert len(actual_items) == len(expected_items)
    for got, want in zip(actual_items, expected_items, strict=True):
        torch_mod.testing.assert_close(got, want, atol=0, rtol=0)
    assert spec.tolerance_for("bfloat16").atol == pytest.approx(2e-2)
    assert spec.tolerance_for("bfloat16").rtol == pytest.approx(2e-2)


def test_artifact_round_trip_carries_parent_provenance(tmp_path) -> None:
    ir = _gated_residual_ir()
    region = _region_for(
        ir,
        [
            "aten::_to_copy",
            "aten::_to_copy",
            "aten::unsqueeze",
            "aten::mul",
            "aten::add",
            "aten::_to_copy",
        ],
    )
    manifest = build_manifest(region)
    assert manifest["parent"]["fingerprint"] == region.fingerprint
    assert manifest["parent"]["timing_scope"] == (
        "parent_region_not_selected_subregion"
    )
    assert manifest["parent"]["capture_mode"] == "export"
    assert manifest["selected_node_ids"]
    assert manifest["boundary_refs"]
    assert manifest["output_node_ids"]
    report = DiscoveryReport(
        producer={"name": "test", "version": "1"},
        workload={"workload_id": "wan-test", "model_id": "wan"},
        environment={"device": "test"},
        total_cuda_time_us=1000.0,
        operators=(),
        regions=(region,),
    )
    report_path = tmp_path / "report.json"
    write_discovery_report(report, report_path)
    paths = write_generated_artifacts(report_path, tmp_path / "generated")
    loaded = spec_from_manifest(paths["manifest"])
    assert loaded.graph_fingerprint == region.fingerprint
    assert loaded.name == manifest["name"]
    corpus = json.loads(paths["corpus"].read_text())
    assert corpus["cases"][0]["weight"] == region.calls


def test_generated_manifest_builds_subgraph_dispatch_contract() -> None:
    from autokernel.specgen import build_dispatch_contract

    region = _region_for(_gated_residual_ir(), [
        "aten::_to_copy",
        "aten::_to_copy",
        "aten::unsqueeze",
        "aten::mul",
        "aten::add",
        "aten::_to_copy",
    ])
    manifest = build_manifest(region)
    contract = build_dispatch_contract(manifest)

    assert contract["operation"]["target_kind"] == "subgraph"
    assert contract["operation"]["graph_fingerprint"] == region.fingerprint
    assert contract["operation"]["parent_module"] == "transformer.blocks"
    assert contract["operation"]["boundary_refs"] == manifest["boundary_refs"]
    assert contract["signature"]["inputs"]
    assert contract["signature"]["outputs"]


def test_runtime_adapter_converts_boundary_positionals_to_search_kwargs(
    tmp_path, torch_mod
) -> None:
    import importlib.util

    from autokernel.specgen import write_runtime_adapter

    region = _region_for(_gated_residual_ir(), [
        "aten::_to_copy",
        "aten::_to_copy",
        "aten::unsqueeze",
        "aten::mul",
        "aten::add",
        "aten::_to_copy",
    ])
    manifest = build_manifest(region)
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        "def kernel_fn(**inputs):\n"
        "    values = list(inputs.values())\n"
        "    return values[0] + values[1]\n",
        encoding="utf-8",
    )
    adapter = write_runtime_adapter(manifest, tmp_path / "entry.py")
    spec = importlib.util.spec_from_file_location("test_runtime_adapter", adapter)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generated_ir = ExecutableIR.from_dict(manifest["executable_ir"])
    inputs = [torch_mod.ones(item.meta.shape) for item in generated_ir.inputs]

    actual = module.fused_subgraph(object(), *inputs)

    torch_mod.testing.assert_close(actual, inputs[0] + inputs[1])


def test_runtime_adapter_import_writes_no_candidate_bytecode(
    tmp_path, torch_mod
) -> None:
    """The generated adapter must not write bytecode into an immutable bundle.

    Runtimes import ``entry.py`` from inside a verified artifact bundle; a
    bytecode cache written next to ``candidate.py`` would be undeclared
    executable content and fail the bundle's next verification.
    """
    import importlib.util

    from autokernel.specgen import write_runtime_adapter

    region = _region_for(_gated_residual_ir(), [
        "aten::_to_copy",
        "aten::_to_copy",
        "aten::unsqueeze",
        "aten::mul",
        "aten::add",
        "aten::_to_copy",
    ])
    manifest = build_manifest(region)
    (tmp_path / "candidate.py").write_text(
        "def kernel_fn(**inputs):\n"
        "    values = list(inputs.values())\n"
        "    return values[0] + values[1]\n",
        encoding="utf-8",
    )
    adapter = write_runtime_adapter(manifest, tmp_path / "entry.py")

    # Simulate a runtime that imports the adapter through the standard source
    # loader without bytecode suppression of its own.
    spec = importlib.util.spec_from_file_location(
        "test_runtime_adapter_bytecode", adapter
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    stray = [
        path
        for path in tmp_path.rglob("*.pyc")
        if path.name.startswith("candidate")
    ]
    assert stray == []


def test_discovery_specgen_cli_writes_bench_artifacts(tmp_path, repo_root) -> None:
    ir = _gated_residual_ir()
    region = _region_for(ir, _operations_for(ir))
    report = DiscoveryReport(
        producer={"name": "test", "version": "1"},
        workload={"workload_id": "wan-test", "model_id": "wan"},
        environment={"device": "test"},
        total_cuda_time_us=1000.0,
        operators=(),
        regions=(region,),
    )
    report_path = tmp_path / "report.json"
    output = tmp_path / "generated"
    write_discovery_report(report, report_path)
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "discovery.py"),
            "specgen",
            str(report_path),
            "--region",
            region.fingerprint,
            "--output",
            str(output),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "SPECGEN: PASS" in result.stdout
    assert {path.name for path in output.iterdir()} == {
        "corpus.json",
        "kernel.py",
        "manifest.json",
        "spec.py",
    }
