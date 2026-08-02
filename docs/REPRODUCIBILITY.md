# Reproducibility

Every performance or correctness claim MotionKernel makes should be
reproducible from what is written down. This page states what is recorded, what
is genuinely reproducible, and what is not.

## What is recorded with every result

Benchmark results (`bench.py --result-json`) carry a versioned envelope:

- GPU name, compute capability, SM count, memory, and peak specs;
- PyTorch, Triton, CUDA and driver versions, plus platform and Python version;
- the resolved spec, the shape corpus and its identity, sizes and dtypes;
- the parity policy in force, including whether it was exact and whether
  approximate math was permitted;
- per-stage correctness outcomes and per-leaf error statistics;
- warmup and measurement counts, median latency, and the baseline mode
  (`eager` or `compile`) the candidate was compared against.

Artifact bundles additionally pin the graph fingerprint, input and output
signatures, compatibility ranges, and the SHA-256 of every file.

Campaign runs write a `preflight.json` recording MotionKernel and FastVideo
commit identities, the workload identity and its SHA-256, and the execution
policy — so a run can be tied to exact source revisions after the fact.

## Reproducing the V1 LTX proof

The evidence is [docs/LTX_V1_R4_ROOT_CAUSE.md](LTX_V1_R4_ROOT_CAUSE.md). To
reproduce it you need:

| | |
|---|---|
| GPU | NVIDIA GB200 (sm100) |
| Model | `FastVideo/LTX2-Distilled-Diffusers` |
| Workload | `workloads/ltx_480p.yaml` |
| Stack | PyTorch 2.8.0a0, Triton 3.3.0, CUDA 12.9 |
| Framework | a FastVideo checkout with the artifact dispatch bridge |

```bash
motionkernel optimize \
  --workload workloads/ltx_480p.yaml \
  --fastvideo-checkout /path/to/FastVideo \
  --output /path/to/run-dir
```

The run is resumable and each stage writes its own `result.json`. The
end-to-end A/B stage writes native and candidate generation results plus
dispatch diagnostics, which is the evidence a claim should cite.

## What reproduces exactly, and what does not

**Bitwise reproducible.** Output parity under `parity.policy: byte_equal`. The
V1 proof asserts byte-identical generated frames, and that is a deterministic
property: it either holds or it does not. It does not depend on machine load.

**Not bitwise reproducible.** Wall-clock latency. On a shared cluster the
baseline moves substantially. Measured during V1 validation, on the same node
and workload:

- native medians across five paired A/B runs spanned 3.1457s to 3.7494s (19%);
- candidate medians spanned 2.9963s to 3.1199s (4%).

The candidate arm is more stable because it replays a fixed graph. The
implication for anyone reading a speedup number: **the ratio is only meaningful
when both arms were measured in the same job, on the same node, in the same
session.** Comparing a candidate median from one run against a native median
from another is not a valid comparison, and MotionKernel's own reports pair
them.

This is also why the V1 evidence reports five paired measurements rather than
one, and reports the conservative minimum-to-minimum ratio alongside the
median.

## Rules for claiming a speedup

A performance claim in this repository must state:

- GPU model and compute capability;
- PyTorch, Triton, CUDA and driver versions;
- input shapes, dtypes and layouts;
- warmup and measurement methodology, including the number of timed runs;
- median latency and its spread;
- the exact baseline compared against (`eager` or `compile`, native or
  optimized);
- whether both arms were measured in the same session.

A single-run measurement is not a claim. If the effect is smaller than the
observed baseline spread, say so.

## Determinism knobs

Workload manifests fix the sampling seed, resolution, frame count and step
count, so the generation itself is deterministic given the same model weights.
The remaining sources of run-to-run variation are scheduler placement, clock
and power state, and other tenants on the node — none of which MotionKernel
controls. Report medians over several runs rather than best-of.
