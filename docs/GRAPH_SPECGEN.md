# Graph-derived kernel specifications

MotionKernel can turn an operand-aware FastVideo discovery region into a
benchmark problem without importing model source or serializing model data.

```bash
python discovery.py specgen workspace/wan/discovery.json \
  --region 0123456789abcdef0123456789abcdef \
  --output workspace/generated_specs/wan
```

The output directory contains:

- `manifest.json`: parent fingerprint/timing provenance and validated node IR;
- `spec.py`: a `KernelSpec` entry point for `bench.py --spec`;
- `kernel.py`: the correct eager starter candidate;
- `corpus.json`: the observed shape and production call weight.

Run it through the existing harness from the generated directory:

```bash
cd workspace/generated_specs/wan
python /path/to/motionkernel/bench.py \
  --spec ./spec.py:SPEC \
  --shape-corpus ./corpus.json \
  --shape-corpus-only \
  --quick
```

## Trust and failure behavior

The ordinary discovery operation list is not executable: it does not describe
operand order, constants, or outputs. Spec generation therefore requires
`attributes.executable_ir`, emitted by FastVideo's `torch.export` fallback.
The IR contains graph wiring, tensor shapes/dtypes, and finite scalar metadata
only. Runtime and lifted tensors have generic names; values, parameter paths,
prompts, activations, source, and credentials are not exported.

MotionKernel validates the IR before any execution, checks that every IR node
matches the region's canonical operation list, and executes only exact ATen
overloads in a small allowlist. Unknown expressions and operations are never
evaluated. A connected allowlisted component may use an unsupported parent's
tensor output as an explicit generated input when output metadata is
available. Parent-region CUDA timing is retained as provenance and is labeled
as parent timing; it is not misreported as selected-subregion latency.

The generated manifest also contains the export rewrite recipe:
`selected_node_ids`, ordered `boundary_refs`, and `output_node_ids`. The
artifact packager derives runtime operation and tensor-signature sections with
`build_dispatch_contract()`. Boundary shape, stride, dtype, device type, and
gradient metadata must all be present; a partial recipe fails closed.
`write_runtime_adapter()` bridges the benchmark candidate's keyword arguments
to FastVideo's positional boundary-tensor calling convention.

The allowlist covers arithmetic, LayerNorm, cast, broadcast, view, transpose,
slice, and tuple-selection operations needed to derive Wan normalization and
residual candidates. It also includes the exact `mean.dim`, `rsqrt.default`,
`silu.default`, and `gelu.default` overloads observed in the LTX transformer
profile. Extend it only with captured profile evidence and exact parity tests.

Each captured region currently represents one bounded shape variant, so its
generated corpus contains that exact observation weighted by call count.
Small, medium, and large all select the same observation. Multi-shape generated
specs require a future cross-region aggregation contract; MotionKernel does
not invent unobserved intermediate shapes.
