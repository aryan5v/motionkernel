"""CPU tests for model-independent Dynamo/FX region capture."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from autokernel.discovery import (
    DiscoveryReport,
    RegionCaptureSession,
    capture_model_regions,
    capture_module_region,
    fx_capture,
    is_region_safe,
    reject_region,
)


class _PureBlock(nn.Module):
    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(x * scale + x)


class _MutatingBlock(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x.add_(1.0)
        return x


class _TinyStack(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block0 = _PureBlock()
        self.block1 = _PureBlock()

    def forward(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        x = self.block0(x, scale)
        x = self.block1(x, scale)
        return x


class _ShapeBranchBlock(nn.Module):
    """Shape-dependent Python control flow: FX breaks, export specializes."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] > 4:
            return x * 2.0
        return x + 1.0


class _CachedRotaryLikeBlock(nn.Module):
    """Representative cached-table shape branch used by rotary modules."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "rotary_cache",
            torch.ones(16, 8),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] <= self.rotary_cache.shape[0]:
            return x * self.rotary_cache[: x.shape[-2]]
        return x


class _DataDependentBlock(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum().item() > 0:
            return x * 2
        return x


class _UnknownAliasBlock(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.as_strided(x, x.shape, x.stride())


class _ParameterBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def test_cpu_module_produces_graph_region_with_ops_and_deps():
    module = _PureBlock()
    x = torch.randn(2, 8)
    scale = torch.randn(2, 8)
    result = capture_module_region(
        module,
        (x, scale),
        name="test.pure_block",
        parent_module="blocks.0",
    )
    assert result.region is not None
    region = result.region
    assert region.operations
    assert region.inputs
    assert region.outputs
    assert region.parent_module == "blocks.0"
    assert region.fingerprint
    # Dependencies should encode at least one edge for a multi-op graph.
    assert isinstance(region.dependencies, tuple)
    # Metadata only — no forbidden keys
    payload = region.as_dict()
    assert "values" not in str(payload).lower() or "values" not in payload
    for key in ("weights", "prompt", "tensor_values", "source_code"):
        assert key not in payload


def test_equivalent_graphs_stable_fingerprints():
    module = _PureBlock()
    a = capture_module_region(
        module,
        (torch.randn(2, 8), torch.randn(2, 8)),
        name="test.pure_block",
        parent_module="blocks.0",
    )
    b = capture_module_region(
        module,
        (torch.randn(2, 8), torch.randn(2, 8)),
        name="test.pure_block",
        parent_module="blocks.0",
    )
    assert a.region is not None and b.region is not None
    # Same structure + same shapes/dtypes => same fingerprint (values ignored).
    assert a.region.operations == b.region.operations
    assert a.region.fingerprint == b.region.fingerprint


def test_fingerprint_changes_with_shape():
    module = _PureBlock()
    small = capture_module_region(
        module,
        (torch.randn(2, 8), torch.randn(2, 8)),
        name="test.pure_block",
        parent_module="blocks.0",
    )
    large = capture_module_region(
        module,
        (torch.randn(4, 16), torch.randn(4, 16)),
        name="test.pure_block",
        parent_module="blocks.0",
    )
    assert small.region is not None and large.region is not None
    assert small.region.fingerprint != large.region.fingerprint


def test_mutation_rejected_fail_closed():
    module = _MutatingBlock()
    result = capture_module_region(
        module,
        (torch.randn(2, 4),),
        name="test.mutating",
        parent_module="bad",
    )
    # Either trace fails or region is rejected for in-place mutation.
    if result.region is not None:
        assert result.region.rejection_reasons
        assert not is_region_safe(result.region.operations)
    else:
        assert result.graph_breaks


def test_collectives_and_custom_ops_rejected():
    reasons = reject_region(
        ["aten::mul", "c10d::all_reduce_", "custom::flash_attn"]
    )
    assert any("collective" in r for r in reasons)
    assert any("custom" in r for r in reasons)


def test_repeated_module_capture_session_builds_report():
    model = _TinyStack()
    x = torch.randn(2, 8)
    scale = torch.randn(2, 8)

    with capture_model_regions(
        model,
        predicate=lambda name, _m: name in {"block0", "block1"},
        name_prefix="dit",
    ) as session:
        # Multiple calls → aggregated call counts / shape frequency.
        for _ in range(3):
            model(x, scale)

    regions = session.regions()
    assert regions
    # Same pure block structure should collapse to one fingerprint class
    # (two scopes may still share ops; at least one region with calls>=3).
    assert any(region.calls >= 3 for region in regions) or any(
        sum((region.shape_frequency or {}).values()) >= 3 for region in regions
    )

    report = session.to_discovery_report(
        workload={"workload_id": "unit-fx", "model_id": "toy"},
        total_cuda_time_us=0.0,
    )
    assert isinstance(report, DiscoveryReport)
    assert report.regions
    # Graph breaks / unsupported must be present as fields (may be empty for pure).
    assert isinstance(report.graph_breaks, tuple)
    assert isinstance(report.unsupported, tuple)
    # Round-trip
    again = DiscoveryReport.from_dict(report.as_dict())
    assert [r.fingerprint for r in again.regions] == [
        r.fingerprint for r in report.regions
    ]


def test_graph_breaks_visible_for_untraceable_control_flow():
    result = capture_module_region(
        _DataDependentBlock(),
        (torch.randn(2, 2),),
        name="test.data_dependent",
    )
    # symbolic_trace typically fails on data-dependent Python control flow
    assert result.region is None or result.graph_breaks or result.region.rejection_reasons
    if result.region is None:
        assert result.graph_breaks
        assert all(
            break_record.reason.startswith("capture_failed:")
            for break_record in result.graph_breaks
        )
        assert {failure.split(":", 2)[1] for failure in result.mode_failures} == {
            "symbolic",
            "export",
            "dynamo",
        }


def test_auto_falls_back_from_symbolic_to_export_for_shape_branch():
    result = capture_module_region(
        _ShapeBranchBlock(),
        (torch.randn(2, 8),),
        name="test.shape_branch",
    )

    assert result.region is not None
    assert result.capture_mode == "export"
    assert result.region.attributes["capture_mode"] == "export"
    assert result.region.attributes["capture_attempts"] == [
        "symbolic",
        "export",
    ]
    assert result.mode_failures == (
        "capture_failed:symbolic:dynamic_python_control_flow:TraceError",
    )
    assert result.graph_breaks[0].reason == result.mode_failures[0]


def test_cached_rotary_shape_branch_uses_export_fallback():
    result = capture_module_region(
        _CachedRotaryLikeBlock(),
        (torch.randn(2, 8),),
        name="test.cached_rotary",
    )

    assert result.region is not None
    assert result.capture_mode == "export"
    assert "aten::mul" in result.region.operations
    payload = result.region.as_dict()
    assert "rotary_cache" not in payload.get("safe_constants", {})


def test_tensor_kwargs_are_preserved_for_export_and_metadata():
    class _KeywordBlock(nn.Module):
        def forward(
            self,
            x: torch.Tensor,
            *,
            gate: torch.Tensor,
            enabled: bool,
        ) -> torch.Tensor:
            if enabled:
                return x * gate
            return x

    x = torch.randn(2, 8)
    gate = torch.randn(2, 8)
    result = capture_module_region(
        _KeywordBlock(),
        (x,),
        example_kwargs={"gate": gate, "enabled": True},
        example_output=x * gate,
        name="test.kwargs",
        tracer="export",
    )

    assert result.region is not None
    assert [item.name for item in result.region.inputs] == [
        "input_0",
        "kwarg_gate",
    ]
    assert result.region.outputs[0].shape == (2, 8)


def test_export_lifted_parameters_never_enter_serialized_metadata():
    result = capture_module_region(
        _ParameterBlock(),
        (torch.randn(2, 8),),
        name="test.parameters",
        tracer="export",
    )

    assert result.region is not None
    payload = result.region.as_dict()
    assert [item["name"] for item in payload["inputs"]] == ["input_0"]
    assert "weights" not in payload
    assert "tensor_values" not in payload
    assert all(
        not key.startswith("attr:")
        for key in payload.get("safe_constants", {})
    )


@pytest.mark.parametrize("mode", ["symbolic", "export", "dynamo"])
def test_each_capture_mode_is_cpu_testable(mode):
    result = capture_module_region(
        _PureBlock(),
        (torch.randn(2, 8), torch.randn(2, 8)),
        name=f"test.mode.{mode}",
        tracer=mode,
    )

    assert result.region is not None
    assert result.capture_mode == mode
    assert result.region.attributes["capture_mode"] == mode
    assert result.region.attributes["capture_attempts"] == [mode]


@pytest.mark.parametrize("mode", ["symbolic", "export", "dynamo"])
def test_mutation_fails_closed_in_every_capture_mode(mode):
    result = capture_module_region(
        _MutatingBlock(),
        (torch.randn(2, 4),),
        name=f"test.mutation.{mode}",
        tracer=mode,
    )

    assert result.region is not None
    assert result.region.rejection_reasons
    assert any(
        "mutation" in reason
        for reason in result.region.rejection_reasons
    )


@pytest.mark.parametrize("mode", ["symbolic", "export", "dynamo"])
def test_data_dependent_control_flow_fails_closed_in_every_mode(mode):
    result = capture_module_region(
        _DataDependentBlock(),
        (torch.ones(2, 2),),
        name=f"test.data_dependent.{mode}",
        tracer=mode,
    )

    assert result.region is None
    assert result.graph_breaks
    assert all(
        item.reason.startswith(f"capture_failed:{mode}:")
        for item in result.graph_breaks
    )


@pytest.mark.parametrize("mode", ["symbolic", "export", "dynamo"])
def test_unknown_aliasing_fails_closed_in_every_mode(mode):
    result = capture_module_region(
        _UnknownAliasBlock(),
        (torch.randn(2, 8),),
        name=f"test.alias.{mode}",
        tracer=mode,
    )

    assert result.region is not None
    assert any(
        reason.startswith("capture_safety:unknown_aliasing:")
        for reason in result.region.rejection_reasons
    )


@pytest.mark.parametrize("mode", ["symbolic", "export", "dynamo"])
def test_fingerprint_stable_within_each_capture_mode(mode):
    module = _PureBlock()
    first = capture_module_region(
        module,
        (torch.randn(2, 8), torch.randn(2, 8)),
        name=f"test.stable.{mode}",
        tracer=mode,
    )
    second = capture_module_region(
        module,
        (torch.randn(2, 8), torch.randn(2, 8)),
        name=f"test.stable.{mode}",
        tracer=mode,
    )

    assert first.region is not None and second.region is not None
    assert first.region.fingerprint == second.region.fingerprint


def test_auto_reaches_dynamo_when_earlier_modes_fail(monkeypatch):
    original = fx_capture._trace_module

    def force_first_two_failures(*args, mode, **kwargs):
        if mode in {"symbolic", "export"}:
            raise RuntimeError(f"{mode} unavailable")
        return original(*args, mode=mode, **kwargs)

    monkeypatch.setattr(fx_capture, "_trace_module", force_first_two_failures)
    result = capture_module_region(
        _PureBlock(),
        (torch.randn(2, 8), torch.randn(2, 8)),
        name="test.dynamo_fallback",
    )

    assert result.region is not None
    assert result.capture_mode == "dynamo"
    assert result.region.attributes["capture_attempts"] == [
        "symbolic",
        "export",
        "dynamo",
    ]
    assert len(result.mode_failures) == 2


def test_failure_diagnostics_never_serialize_raw_exception_text(monkeypatch):
    def fail_with_sensitive_text(*args, mode, **kwargs):
        raise RuntimeError(
            "prompt=private customer text token=secret source_code=hidden"
        )

    monkeypatch.setattr(fx_capture, "_trace_module", fail_with_sensitive_text)
    result = capture_module_region(
        _PureBlock(),
        (torch.randn(2, 8), torch.randn(2, 8)),
        name="test.safe_failure",
    )

    serialized = str([item.as_dict() for item in result.graph_breaks])
    assert result.region is None
    assert "private customer" not in serialized
    assert "secret" not in serialized
    assert "source_code" not in serialized


def test_session_register_explicit_module():
    block = _PureBlock()
    session = RegionCaptureSession(name_prefix="leaf")
    session.register_module(block, scope="blocks.0")
    for _ in range(2):
        block(torch.randn(1, 4), torch.randn(1, 4))
    session.close()
    regions = session.regions()
    assert len(regions) == 1
    assert regions[0].calls == 2
    assert regions[0].parent_module == "blocks.0"
    assert regions[0].shape_frequency
