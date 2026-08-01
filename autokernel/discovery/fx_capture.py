"""Model-independent Dynamo/FX region capture (metadata only).

Captures repeated module calls as executable FX tensor subgraphs for discovery
and ranking. Records only:

- ordered operations and dependencies
- input/output tensor signatures (shape, stride, dtype, device, requires_grad)
- safe scalar constants needed for semantics
- parent module scope, call counts, shape frequency
- graph breaks and unsupported operations

Never serializes tensor values, weights, prompts, model outputs, credentials,
or arbitrary Python source. Fail closed on mutation, collectives,
data-dependent control flow, unknown aliasing, and unsupported custom ops.

CUDA timings remain optional and are filled by a separate profiler path.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .ranking import classify_pattern_family
from .safety import normalize_op_name, reject_region
from .types import (
    DiscoveryReport,
    GraphBreakRecord,
    GraphRegion,
    TensorMeta,
    UnsupportedOpRecord,
)

_FORBIDDEN_SERIALIZED_KEYS = frozenset(
    {
        "activations",
        "credential",
        "credentials",
        "data",
        "password",
        "prompt",
        "secret",
        "secrets",
        "source",
        "source_code",
        "tensor_values",
        "token",
        "values",
        "weights",
    }
)

_SAFE_SCALAR_TYPES = (bool, int, float, str)
_CAPTURE_MODES = ("symbolic", "export", "dynamo")
_NO_OUTPUT = object()


@dataclass(frozen=True)
class CaptureResult:
    """Result of attempting to capture one module or callable."""

    region: GraphRegion | None
    graph_breaks: tuple[GraphBreakRecord, ...]
    unsupported: tuple[UnsupportedOpRecord, ...]
    operations: tuple[str, ...]
    capture_mode: str | None = None
    mode_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class _TraceResult:
    """Internal graph capture result, including fail-closed diagnostics."""

    graph: Any
    mode: str
    structural_rejections: tuple[str, ...] = ()


@dataclass
class _RegionAccumulator:
    """Mutable aggregation for repeated captures of the same fingerprint."""

    region: GraphRegion
    calls: int = 0
    shape_keys: dict[str, int] = field(default_factory=dict)
    breaks: list[GraphBreakRecord] = field(default_factory=list)
    unsupported: list[UnsupportedOpRecord] = field(default_factory=list)


def _tensor_meta_from_example(name: str, tensor: Any) -> TensorMeta:
    """Layout metadata only — never reads or stores tensor values."""
    shape = tuple(int(x) for x in tensor.shape)
    if hasattr(tensor, "stride"):
        stride = tuple(int(x) for x in tensor.stride())
    else:
        stride_list: list[int] = []
        running = 1
        for dim in reversed(shape):
            stride_list.append(running)
            running *= max(dim, 1)
        stride = tuple(reversed(stride_list))
    dtype = str(tensor.dtype).replace("torch.", "")
    device_type = "cpu"
    if hasattr(tensor, "device") and hasattr(tensor.device, "type"):
        device_type = tensor.device.type
    requires_grad = bool(getattr(tensor, "requires_grad", False))
    return TensorMeta(
        name=name,
        shape=shape,
        stride=stride,
        dtype=dtype,
        device_type=device_type,
        requires_grad=requires_grad,
    )


def _safe_tensor_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._-")
    if not cleaned or not re.match(r"^[A-Za-z0-9]", cleaned):
        cleaned = f"tensor_{cleaned}"
    return cleaned[:128]


def _flatten_tensor_examples(value: Any, *, prefix: str) -> list[tuple[str, Any]]:
    """Return tensor leaves with structural names, never tensor values."""
    import torch

    if isinstance(value, torch.Tensor):
        return [(_safe_tensor_name(prefix), value)]
    leaves: list[tuple[str, Any]] = []
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            leaves.extend(
                _flatten_tensor_examples(item, prefix=f"{prefix}_{index}")
            )
    elif isinstance(value, Mapping):
        for key, item in value.items():
            leaves.extend(
                _flatten_tensor_examples(
                    item,
                    prefix=f"{prefix}_{_safe_tensor_name(str(key))}",
                )
            )
    return leaves


def _input_tensor_examples(
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> tuple[tuple[str, Any], ...]:
    leaves: list[tuple[str, Any]] = []
    for index, value in enumerate(args):
        leaves.extend(
            _flatten_tensor_examples(value, prefix=f"input_{index}")
        )
    for key, value in kwargs.items():
        leaves.extend(
            _flatten_tensor_examples(
                value,
                prefix=f"kwarg_{_safe_tensor_name(str(key))}",
            )
        )
    return tuple(leaves)


def _output_tensor_examples(output: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(_flatten_tensor_examples(output, prefix="output"))


def _shape_frequency_key(inputs: Sequence[TensorMeta]) -> str:
    parts = []
    for item in inputs:
        shape = "x".join(str(d) for d in item.shape)
        parts.append(f"{item.name}:{shape}:{item.dtype}")
    return "|".join(parts)


def _function_to_op_key(target: Any) -> str:
    text = str(target)
    if "aten::" in text:
        return normalize_op_name(
            "aten::" + text.split("aten::", 1)[1].split(".")[0].split("(")[0]
        )
    if "aten." in text:
        return normalize_op_name("aten::" + text.split("aten.", 1)[1].split(".")[0])
    name = getattr(target, "__name__", None) or text
    module = getattr(target, "__module__", "") or ""
    if "torch" in module or name in {
        "add",
        "mul",
        "sub",
        "div",
        "silu",
        "gelu",
        "relu",
        "sigmoid",
        "layer_norm",
        "softmax",
        "tanh",
        "rsqrt",
        "sqrt",
        "pow",
        "mean",
        "var",
        "neg",
        "exp",
        "clone",
        "contiguous",
        "view",
        "reshape",
        "permute",
        "transpose",
        "unsqueeze",
        "squeeze",
        "cat",
        "stack",
        "expand",
        "type_as",
    }:
        return normalize_op_name(f"aten::{name}")
    return normalize_op_name(str(name))


def _is_safe_constant_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, _SAFE_SCALAR_TYPES):
        if isinstance(value, float):
            import math

            return math.isfinite(value)
        if isinstance(value, str):
            # Short enum-like tags only — never free-form source/prompts.
            return bool(re.fullmatch(r"[A-Za-z0-9_.:/-]{1,64}", value))
        return True
    if isinstance(value, (list, tuple)):
        if len(value) > 32:
            return False
        return all(_is_safe_constant_value(item) for item in value)
    return False


def _sanitize_safe_constants(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only finite scalars / short tags; drop anything else."""
    result: dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            continue
        if key.lower() in _FORBIDDEN_SERIALIZED_KEYS:
            continue
        if _is_safe_constant_value(value):
            result[key] = value
    return result


def _extract_graph_structure(graph: Any) -> tuple[list[str], list[str], dict[str, Any], list[str]]:
    """Return (operations, dependencies, safe_constants, structural_rejections)."""
    operations: list[str] = []
    dependencies: list[str] = []
    safe_constants: dict[str, Any] = {}
    structural: list[str] = []
    node_index: dict[str, int] = {}

    def node_names(value: Any) -> Iterator[str]:
        name = getattr(value, "name", None)
        if name is not None and hasattr(value, "op"):
            yield name
        elif isinstance(value, Mapping):
            for item in value.values():
                yield from node_names(item)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                yield from node_names(item)

    def record_dependencies(node: Any, index: int) -> None:
        seen: set[str] = set()
        for arg_name in node_names((node.args, node.kwargs or {})):
            if arg_name in node_index and arg_name not in seen:
                dependencies.append(f"{node_index[arg_name]}->{index}")
                seen.add(arg_name)

    def record_constants(node: Any, op_key: str) -> None:
        for arg_index, value in enumerate(node.args):
            if hasattr(value, "op"):
                continue
            if _is_safe_constant_value(value):
                safe_constants[f"{node.name}.arg{arg_index}"] = value
            elif value is not None and not tuple(node_names(value)):
                structural.append(
                    f"{op_key}: unsafe positional constant arg{arg_index}"
                )
        for key, value in (node.kwargs or {}).items():
            if hasattr(value, "op"):
                continue
            if _is_safe_constant_value(value):
                safe_constants[f"{node.name}.{key}"] = value
            elif value is not None and not tuple(node_names(value)):
                structural.append(
                    f"{op_key}: unsafe or non-scalar constant {key!r}"
                )

    for node in graph.nodes:
        if node.op in {"placeholder", "output"}:
            continue
        if node.op == "get_attr":
            # A graph attribute can be a parameter, buffer, tensor constant, or
            # Python value. Without reading it we cannot prove which; never
            # serialize it or claim the graph is self-contained.
            target = str(node.target)
            structural.append(
                f"get_attr:{target}: lifted attribute value not exported"
            )
            continue
        if node.op == "call_function":
            op_key = _function_to_op_key(node.target)
            idx = len(operations)
            node_index[node.name] = idx
            operations.append(op_key)
            record_dependencies(node, idx)
            record_constants(node, op_key)
        elif node.op == "call_method":
            op_key = normalize_op_name(f"aten::{node.target}")
            idx = len(operations)
            node_index[node.name] = idx
            operations.append(op_key)
            record_dependencies(node, idx)
            record_constants(node, op_key)
        elif node.op == "call_module":
            # Nested modules collapse to a single opaque node — fail closed for
            # leaf fusion search unless expanded.
            op_key = normalize_op_name(f"module::{node.target}")
            idx = len(operations)
            node_index[node.name] = idx
            operations.append(op_key)
            structural.append(f"{op_key}: nested module not expanded")
        else:
            structural.append(f"unknown_fx_op:{node.op}")

    # Unknown aliasing: multiple outputs writing overlapping views without
    # explicit clone is hard to prove; flag in-place method names already
    # covered by reject_region. Detect star-deps fan-in of getitem/scatter-like.
    for op in operations:
        if "scatter" in op or "index_put" in op or "copy_" in op:
            structural.append(f"{op}: potential aliasing mutation")
        if "as_strided" in op or op.endswith("::alias"):
            structural.append(f"capture_safety:unknown_aliasing:{op}")

    return operations, dependencies, safe_constants, structural


def _capture_failure_reason(mode: str, exc: Exception) -> str:
    """Classify a tracer failure without serializing its potentially sensitive text."""
    text = str(exc).lower()
    if any(
        marker in text
        for marker in (
            "data-dependent",
            "data dependent",
            "guardondatadependentsymnode",
            "could not guard on data-dependent",
            ".item()",
        )
    ):
        code = "data_dependent_control_flow"
    elif any(
        marker in text
        for marker in (
            "control flow",
            "proxy object",
            "symbolically traced variables",
            "cannot be iterated",
        )
    ):
        code = "dynamic_python_control_flow"
    elif "alias" in text:
        code = "unknown_aliasing"
    elif "unsupported" in text or "not supported" in text:
        code = "unsupported_graph"
    else:
        code = "trace_error"
    return f"capture_failed:{mode}:{code}:{type(exc).__name__}"


def _trace_module(
    module: Any,
    *,
    mode: str,
    example_inputs: Sequence[Any],
    example_kwargs: Mapping[str, Any],
) -> _TraceResult:
    """Capture one graph mode or raise; never silently changes mode."""
    import torch
    from torch import fx

    if mode == "symbolic":
        return _TraceResult(graph=fx.symbolic_trace(module), mode=mode)
    if mode == "export":
        exported_program = torch.export.export(
            module,
            tuple(example_inputs),
            dict(example_kwargs),
            strict=False,
        )
        structural: list[str] = []
        signature = getattr(exported_program, "graph_signature", None)
        for field_name in ("user_inputs_to_mutate", "buffers_to_mutate"):
            mutations = getattr(signature, field_name, None)
            if mutations:
                structural.append(
                    f"capture_safety:{mode}:mutation_signature:{field_name}"
                )
        return _TraceResult(
            graph=exported_program.graph_module,
            mode=mode,
            structural_rejections=tuple(structural),
        )
    if mode == "dynamo":
        exported = torch._dynamo.export(module, aten_graph=True)(
            *example_inputs,
            **dict(example_kwargs),
        )
        graph_module = getattr(exported, "graph_module", None)
        if graph_module is None and isinstance(exported, tuple):
            graph_module = exported[0]
        return _TraceResult(
            graph=graph_module if graph_module is not None else exported,
            mode=mode,
        )
    raise ValueError(
        f"unsupported capture mode {mode!r}; use auto, symbolic, export, or dynamo"
    )


def _capture_mode_order(tracer: str) -> tuple[str, ...]:
    if tracer in {"auto", "fallback"}:
        return _CAPTURE_MODES
    if tracer in _CAPTURE_MODES:
        return (tracer,)
    raise ValueError(
        f"unsupported tracer {tracer!r}; use auto, symbolic, export, or dynamo"
    )


def _assert_region_metadata_only(region: GraphRegion) -> None:
    """Hard privacy check on the produced region dict."""
    payload = region.as_dict()

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, Mapping):
            for key, value in obj.items():
                lower = str(key).lower()
                if lower in _FORBIDDEN_SERIALIZED_KEYS:
                    raise RuntimeError(
                        f"forbidden key {key!r} at {path} in captured region"
                    )
                walk(value, f"{path}.{key}")
        elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
            for index, item in enumerate(obj):
                walk(item, f"{path}[{index}]")

    walk(payload, "region")


def capture_module_region(
    module: Any,
    example_inputs: Sequence[Any],
    *,
    name: str,
    parent_module: str | None = None,
    tracer: str = "auto",
    example_kwargs: Mapping[str, Any] | None = None,
    example_output: Any = _NO_OUTPUT,
    calls: int = 1,
    shape_frequency: Mapping[str, int] | None = None,
) -> CaptureResult:
    """Trace a module and build a metadata-only GraphRegion.

    ``example_inputs`` are used only for tracing shapes/dtypes and one forward
    for output signatures — values are never written into the region.
    """
    import torch

    breaks: list[GraphBreakRecord] = []
    unsupported: list[UnsupportedOpRecord] = []
    operations: list[str] = []
    dependencies: list[str] = []
    safe_constants: dict[str, Any] = {}

    kwargs = dict(example_kwargs or {})
    mode_failures: list[str] = []
    capture_mode: str | None = None
    traced_graph: Any = None
    structural: list[str] = []
    for mode in _capture_mode_order(tracer):
        try:
            traced = _trace_module(
                module,
                mode=mode,
                example_inputs=example_inputs,
                example_kwargs=kwargs,
            )
            graph = getattr(traced.graph, "graph", None)
            if graph is None:
                raise RuntimeError("trace result has no FX graph")
            candidate = _extract_graph_structure(graph)
            if not candidate[0]:
                raise RuntimeError("trace result has no tensor operations")
            operations, dependencies, safe_constants, structural = candidate
            structural.extend(traced.structural_rejections)
            traced_graph = graph
            capture_mode = mode
            break
        except Exception as exc:  # noqa: BLE001 - failures become safe codes
            reason = _capture_failure_reason(mode, exc)
            mode_failures.append(reason)
            breaks.append(
                GraphBreakRecord(scope=name, reason=reason, count=1)
            )

    if traced_graph is None or capture_mode is None:
        return CaptureResult(
            None,
            tuple(breaks),
            tuple(unsupported),
            (),
            capture_mode=None,
            mode_failures=tuple(mode_failures),
        )

    for reason in structural:
        breaks.append(GraphBreakRecord(scope=name, reason=reason, count=1))

    safe_constants = _sanitize_safe_constants(safe_constants)
    rejection = list(reject_region(operations))
    # Nested module / structural reasons already in breaks; fold into rejection.
    for item in breaks:
        if item.reason not in rejection and not item.reason.startswith(
            ("fx_trace_failed", "capture_failed:")
        ):
            rejection.append(item.reason)

    for reason in rejection:
        if (
            "unsupported custom" in reason
            or "not in pure-tensor" in reason
            or "nested module" in reason
        ):
            op_name = reason.rsplit(": ", 1)[0]
            unsupported.append(
                UnsupportedOpRecord(
                    op_name=op_name, reason=reason, count=1, scope=name
                )
            )

    tensor_inputs = _input_tensor_examples(example_inputs, kwargs)
    if not tensor_inputs:
        breaks.append(
            GraphBreakRecord(scope=name, reason="no_tensor_inputs", count=1)
        )
        return CaptureResult(
            None,
            tuple(breaks),
            tuple(unsupported),
            tuple(operations),
            capture_mode=capture_mode,
            mode_failures=tuple(mode_failures),
        )
    inputs = tuple(
        _tensor_meta_from_example(input_name, tensor)
        for input_name, tensor in tensor_inputs
    )
    outputs: tuple[TensorMeta, ...] = ()
    try:
        if example_output is _NO_OUTPUT:
            with torch.no_grad():
                out = module(*example_inputs, **kwargs)
        else:
            out = example_output
        outputs = tuple(
            _tensor_meta_from_example(output_name, tensor)
            for output_name, tensor in _output_tensor_examples(out)
        )
    except Exception as exc:  # noqa: BLE001
        breaks.append(
            GraphBreakRecord(
                scope=name,
                reason=f"output_meta_failed:{type(exc).__name__}",
                count=1,
            )
        )

    freq = dict(shape_frequency) if shape_frequency else {}
    if not freq:
        freq[_shape_frequency_key(inputs)] = max(1, calls)

    family = classify_pattern_family(operations)
    region = GraphRegion.build(
        name=name,
        operations=operations,
        inputs=inputs,
        outputs=outputs,
        dependencies=tuple(dependencies),
        parent_module=parent_module,
        safe_constants=safe_constants or None,
        pattern_family=family,
        rejection_reasons=tuple(dict.fromkeys(rejection)),
        calls=max(1, calls),
        shape_frequency=freq,
        cuda_time_us=0.0,
        self_cuda_time_us=0.0,
        attributes={
            "capture_mode": capture_mode,
            "capture_attempts": list(_capture_mode_order(tracer))[
                : list(_capture_mode_order(tracer)).index(capture_mode) + 1
            ],
            "capture_failures": list(mode_failures),
        },
    )
    _assert_region_metadata_only(region)
    return CaptureResult(
        region=region,
        graph_breaks=tuple(breaks),
        unsupported=tuple(unsupported),
        operations=tuple(operations),
        capture_mode=capture_mode,
        mode_failures=tuple(mode_failures),
    )


def capture_callable_region(
    fn: Callable[..., Any],
    example_inputs: Sequence[Any],
    *,
    name: str,
    tracer: str = "auto",
) -> CaptureResult:
    """Wrap a pure function as an nn.Module and capture it."""
    from torch import nn

    arity = len(example_inputs)
    if arity == 1:

        class _Wrapper1(nn.Module):
            def forward(self, x):  # type: ignore[no-untyped-def]
                return fn(x)

        wrapper: nn.Module = _Wrapper1()
    elif arity == 2:

        class _Wrapper2(nn.Module):
            def forward(self, x, y):  # type: ignore[no-untyped-def]
                return fn(x, y)

        wrapper = _Wrapper2()
    elif arity == 3:

        class _Wrapper3(nn.Module):
            def forward(self, x, y, z):  # type: ignore[no-untyped-def]
                return fn(x, y, z)

        wrapper = _Wrapper3()
    else:
        return CaptureResult(
            None,
            (
                GraphBreakRecord(
                    scope=name,
                    reason=f"callable arity {arity} not supported for FX wrap",
                    count=1,
                ),
            ),
            (),
            (),
            capture_mode=None,
        )

    return capture_module_region(
        wrapper,
        example_inputs,
        name=name,
        parent_module=None,
        tracer=tracer,
    )


class RegionCaptureSession:
    """Hook selected modules, capture FX regions on each forward, aggregate.

    Designed for repeated DiT block / leaf-module calls without capturing the
    entire generation pipeline as one graph.
    """

    def __init__(
        self,
        *,
        tracer: str = "auto",
        name_prefix: str = "region",
    ) -> None:
        self.tracer = tracer
        self.name_prefix = name_prefix
        self._accumulators: dict[str, _RegionAccumulator] = {}
        self._graph_breaks: list[GraphBreakRecord] = []
        self._unsupported: list[UnsupportedOpRecord] = []
        self._hooks: list[Any] = []
        self._call_counters: dict[str, int] = defaultdict(int)
        # Prevent re-entry when capture_module_region runs symbolic_trace /
        # a meta forward on the same hooked module.
        self._capturing: bool = False

    def register_module(
        self,
        module: Any,
        *,
        scope: str,
    ) -> None:
        """Install a forward hook that captures each invocation."""

        def _hook(
            _mod: Any,
            inputs: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            if self._capturing:
                return
            self._call_counters[scope] += 1
            if not _input_tensor_examples(inputs, kwargs):
                self._graph_breaks.append(
                    GraphBreakRecord(
                        scope=scope,
                        reason="no_tensor_inputs",
                        count=1,
                    )
                )
                return
            capture_name = re.sub(
                r"[^A-Za-z0-9._-]", "_", f"{self.name_prefix}.{scope}"
            )
            if not re.match(r"^[A-Za-z0-9]", capture_name):
                capture_name = f"r.{capture_name}"
            # GraphRegion names must match _NAME_PATTERN (128 chars max),
            # so truncate after any prefix is applied.
            capture_name = capture_name[:128]
            self._capturing = True
            try:
                result = capture_module_region(
                    _mod,
                    inputs,
                    name=capture_name,
                    parent_module=scope,
                    tracer=self.tracer,
                    example_kwargs=kwargs,
                    example_output=output,
                )
            finally:
                self._capturing = False
            self._graph_breaks.extend(result.graph_breaks)
            self._unsupported.extend(result.unsupported)
            if result.region is None:
                return
            key = result.region.fingerprint
            shape_key = _shape_frequency_key(result.region.inputs)
            if key not in self._accumulators:
                self._accumulators[key] = _RegionAccumulator(region=result.region)
            acc = self._accumulators[key]
            acc.calls += 1
            acc.shape_keys[shape_key] = acc.shape_keys.get(shape_key, 0) + 1
            acc.breaks.extend(result.graph_breaks)
            acc.unsupported.extend(result.unsupported)

        handle = module.register_forward_hook(_hook, with_kwargs=True)
        self._hooks.append(handle)

    def register_named_children(
        self,
        root: Any,
        *,
        predicate: Callable[[str, Any], bool] | None = None,
    ) -> int:
        """Register hooks on ``named_modules`` matching predicate (default: leaves)."""
        count = 0
        for name, child in root.named_modules():
            if name == "":
                continue
            if predicate is not None:
                if not predicate(name, child):
                    continue
            else:
                # Default: leaf modules with no children
                if any(True for _ in child.children()):
                    continue
            self.register_module(child, scope=name or "root")
            count += 1
        return count

    def close(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def __enter__(self) -> RegionCaptureSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def regions(self) -> tuple[GraphRegion, ...]:
        """Materialize aggregated GraphRegion records."""
        regions: list[GraphRegion] = []
        for acc in self._accumulators.values():
            base = acc.region
            regions.append(
                GraphRegion.build(
                    name=base.name,
                    operations=base.operations,
                    inputs=base.inputs,
                    outputs=base.outputs,
                    dependencies=base.dependencies,
                    parent_module=base.parent_module,
                    safe_constants=base.safe_constants,
                    pattern_family=base.pattern_family,
                    rejection_reasons=base.rejection_reasons,
                    calls=max(1, acc.calls),
                    shape_frequency=dict(acc.shape_keys) or base.shape_frequency,
                    cuda_time_us=base.cuda_time_us,
                    self_cuda_time_us=base.self_cuda_time_us,
                    attributes=base.attributes,
                )
            )
        return tuple(regions)

    def graph_breaks(self) -> tuple[GraphBreakRecord, ...]:
        # Coalesce identical break reasons.
        counts: dict[tuple[str, str, str | None], int] = defaultdict(int)
        for item in self._graph_breaks:
            counts[(item.scope, item.reason, item.op_name)] += item.count
        return tuple(
            GraphBreakRecord(
                scope=scope, reason=reason, op_name=op_name, count=count
            )
            for (scope, reason, op_name), count in sorted(counts.items())
        )

    def unsupported(self) -> tuple[UnsupportedOpRecord, ...]:
        counts: dict[tuple[str, str, str | None], int] = defaultdict(int)
        for item in self._unsupported:
            counts[(item.op_name, item.reason, item.scope)] += item.count
        return tuple(
            UnsupportedOpRecord(
                op_name=op_name, reason=reason, count=count, scope=scope
            )
            for (op_name, reason, scope), count in sorted(counts.items())
        )

    def to_discovery_report(
        self,
        *,
        workload: Mapping[str, Any],
        environment: Mapping[str, Any] | None = None,
        producer: Mapping[str, Any] | None = None,
        total_cuda_time_us: float = 0.0,
    ) -> DiscoveryReport:
        """Build a DiscoveryReport from aggregated FX captures (CPU-safe)."""
        return DiscoveryReport.from_dict(
            {
                "schema_version": 1,
                "producer": dict(
                    producer
                    or {"name": "motionkernel.fx_capture", "version": "1"}
                ),
                "workload": dict(workload),
                "environment": dict(
                    environment
                    or {
                        "hardware_profile_id": "cpu",
                        "software_profile_id": "fx-capture",
                    }
                ),
                "total_cuda_time_us": float(total_cuda_time_us),
                "operators": [],
                "regions": [region.as_dict() for region in self.regions()],
                "graph_breaks": [item.as_dict() for item in self.graph_breaks()],
                "unsupported": [item.as_dict() for item in self.unsupported()],
            }
        )


@contextmanager
def capture_model_regions(
    root: Any,
    *,
    tracer: str = "auto",
    predicate: Callable[[str, Any], bool] | None = None,
    name_prefix: str = "region",
) -> Iterator[RegionCaptureSession]:
    """Context manager: hook modules, run caller forward, yield session."""
    session = RegionCaptureSession(tracer=tracer, name_prefix=name_prefix)
    try:
        session.register_named_children(root, predicate=predicate)
        yield session
    finally:
        session.close()
