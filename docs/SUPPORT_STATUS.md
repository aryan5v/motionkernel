# Support status

This page exists so that "supported" means one thing. A model, framework, or
GPU is listed at exactly one of four levels, and moving up a level requires
evidence that is linked from this page.

## Levels

| Level | What it means |
|---|---|
| **Proven** | An artifact passed strict independent correctness, was packaged and hash-verified, was dispatched by the runtime, preserved the workload's declared output parity, met the campaign's end-to-end speedup gate, and was promoted. Evidence is linked. |
| **Validated (isolated)** | Kernels pass correctness and performance gates against a production shape corpus, on named hardware, with a linked result. No end-to-end model claim. |
| **In progress** | Active work. No correctness or performance claim of any kind. |
| **Target** | On the roadmap. Nothing has been run. |

A target being *listed* is not a support claim. Nothing is described as
"supported" without a link in the Evidence column.

## Models

| Model | Framework | Level | Evidence |
|---|---|---|---|
| LTX2 (`FastVideo/LTX2-Distilled-Diffusers`) | FastVideo | **Proven** | [V1 evidence report](LTX_V1_R4_ROOT_CAUSE.md) |
| Wan | FastVideo | **Validated (isolated)** | [Wan kernel results](WAN_KERNEL_RESULTS.md) |
| Cosmos | FastVideo | **In progress** | — |
| Kandinsky | FastVideo | **Target** | — |
| Diffusers video pipelines | Diffusers | **Target** | — |

### LTX2 — scope of the proof

The proven result is deliberately narrow, and the boundaries matter more than
the headline:

- one artifact (`mk-2c92e356aa34bc0d-7df21b47-sm100`), targeting a subregion of
  `transformer.model.transformer_blocks`;
- one workload: `workloads/ltx_480p.yaml`, 480x768, 97 frames, 8 inference
  steps, `byte_equal` parity;
- one GPU: NVIDIA GB200 (sm100), single-GPU inference, no FSDP;
- one software stack: PyTorch 2.8.0a0, Triton 3.3.0, CUDA 12.9.

Measured: 6,143 dispatched calls with zero runtime fallbacks, byte-identical
generated frames, and a median end-to-end improvement of 1.0857x across 15
timed runs per arm, replicated at 1.2514x in an independent 15-run pair.

Not claimed: any other resolution, frame count, step count, GPU architecture,
multi-GPU or sequence-parallel configuration, or model. Those are untested, not
known-good.

One honest caveat, also stated in the evidence report: the artifact's kernel
saves roughly 124 microseconds per call, which alone bounds the end-to-end gain
near 1.015x. The larger measured figure comes from the runtime additionally
replaying the whole block from a CUDA graph, which removes host-side dispatch
cost. That acceleration exists only on the artifact path, so the A/B is sound,
but more of the gain comes from the framework than from the kernel.

### Wan — scope of the validation

Three fused boundaries (modulated pre-attention LayerNorm, post-attention gated
residual plus LayerNorm, post-MLP gated residual) pass all five forward
correctness gates across their production shape corpora on an NVIDIA GB200.

These are **isolated operator results**. There is no published end-to-end Wan
benchmark, so Wan is not proven and is not described as supported.

## GPU architectures

| Architecture | Level | Notes |
|---|---|---|
| NVIDIA GB200 / sm100 | **Proven** | All V1 evidence was produced here |
| Other NVIDIA (sm80, sm89, sm90) | **In progress** | The platform is architecture-agnostic; no artifact has been validated on them |
| AMD ROCm | **In progress** | Detection and device specs are inherited from upstream and are unverified here |

Artifacts declare the architectures they were validated on. The runtime refuses
to load an artifact on an architecture its manifest does not list, so an
unvalidated architecture fails closed rather than silently running an untested
kernel.

## Frameworks

| Framework | Level | Notes |
|---|---|---|
| FastVideo | **Proven** | Generic graph dispatch; no model-specific code in the bridge |
| Diffusers | **Target** | Artifacts are framework-agnostic by design; no adapter exists yet |

## Changing a level

Raising a level requires a linked artifact or result file and a named hardware
and software stack. Lowering one requires the same evidence to be withdrawn.
If evidence and this page disagree, this page is wrong — fix it.
