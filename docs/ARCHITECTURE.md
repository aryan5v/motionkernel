# Architecture

MotionKernel is two things that must not be confused:

1. an **optimization platform** that discovers, generates, validates and
   packages GPU kernels — heavyweight, runs offline, needs a GPU and a model
   checkout;
2. an **artifact format plus a runtime bridge** that lets a framework consume a
   promoted kernel — lightweight, needs neither the platform nor the search
   agent at inference time.

A framework integrating MotionKernel depends only on (2). That separation is
why a promoted artifact can be consumed by FastVideo, Diffusers, or any other
PyTorch runtime without vendoring the research harness.

## The pipeline

```
workload manifest (YAML)
        |
        v
  [baseline]      native generation, timed, frames saved
        |
        v
  [profile]       torch.profiler export of a real generation
        |
        v
  [discover]      repeated module stacks -> graph regions, fingerprints,
                  impact ranking against measured CUDA time
        |
        v
  [specgen]       graph region -> KernelSpec + manifest + shape corpus
        |
        v
  [search]        autonomous agent edits only kernel.py, benchmarks after
                  every meaningful edit
        |
        v
  [isolated_validate]
                  independent strict re-measurement in a separate process,
                  under the workload's parity policy
        |
        v
  [package]       artifact bundle: manifest + payload + SHA-256 of every file
        |
        v
  [end_to_end_validate]
                  full native-versus-artifact generation A/B, output parity,
                  wall time, peak memory
        |
        v
  [finalize]      promoted / rejected / quarantined -- fail-closed
```

Each stage writes a versioned `result.json` and is resumable. Campaign state is
a single JSON document, so an interrupted overnight run resumes without
repeating GPU work.

## Packages

| Package | Responsibility |
|---|---|
| `autokernel.specs` | `KernelSpec` types, registry, external spec loader, dtype and tolerance vocabulary |
| `autokernel.discovery` | Profiler export parsing, FX/export graph capture, region fingerprinting, impact ranking |
| `autokernel.specgen` | Graph region to executable IR, spec and corpus generation, runtime adapter emission |
| `autokernel.verification` | Output-tree comparison, parity policy, backward and compile gates, corpus validation |
| `autokernel.artifact` | Bundle packaging, hash verification, promotion finalizer |
| `autokernel.workload` | Workload manifest schema, FastVideo launcher bridge, generation-result comparison |
| `autokernel.optimize` | Campaign control plane: preflight, run contract, stage adapters, search, isolation |
| `autokernel.campaign` | Campaign contract, ranking, overnight runner |
| `autokernel.cli` | `motionkernel` console entry point |

`motionkernel` is a canonical alias for all of the above — see
[NAMESPACE_MIGRATION.md](NAMESPACE_MIGRATION.md).

Nothing in these packages may initialize a GPU at import time. Registry
discovery, spec inspection, manifest validation and every policy decision are
CPU-only and unit-testable without hardware, which is what makes the CPU suite
meaningful.

## The trust boundary

The autonomous search agent is the only untrusted component, and it is treated
that way:

- it may edit exactly one file, the candidate's `kernel.py`;
- the harness, spec, manifest, shape corpus, tolerances and verifier are
  outside its write boundary;
- its own benchmark result is never the acceptance signal — after the search,
  the candidate is re-measured by a fixed harness in a separate process;
- the artifact request is derived only from that independent measurement and
  the validated manifest, never from anything the agent reported.

Every gate fails closed. A stage that cannot determine an answer produces
`quarantined`, not a pass.

## The artifact bundle

A bundle is a directory containing `artifact.json` plus the files it declares.
The manifest pins:

- **identity**: graph fingerprint, input and output signatures, target kind
  (`module` or `subgraph`), and for a subgraph the exact node ids to replace;
- **compatibility**: model id and revision, GPU architectures, torch/CUDA/Triton
  ranges, execution mode, distributed mode;
- **evidence**: the isolated benchmark and the full-generation A/B that
  justified the decision;
- **promotion**: `quarantined`, `rejected`, or `promoted`, with a reason;
- **files**: SHA-256 and byte length of every file, including the entry point.

The runtime re-derives the graph fingerprint from the live model and refuses
the bundle if it disagrees, so an artifact cannot be applied to a model it was
not measured against. See [ARTIFACT_BUNDLE.md](ARTIFACT_BUNDLE.md).

## Runtime dispatch

The framework-side bridge attaches to repeated module stacks, and per stack and
per observed input signature decides once whether a trusted artifact may
replace the native forward. The first call always runs natively — that is what
reveals the output signature. Any failure at any point demotes the signature to
native execution permanently and records a structured reason; it never
propagates to the caller.

With no artifact directory configured, nothing is attached, no graph is traced,
and the model runs byte-for-byte as it would without MotionKernel.

## What is inherited

The single-kernel research harness at the repository root (`bench.py`,
`extract.py`, `profile.py`, `reference.py`, `kernels/`, `models/`,
`kernelbench/`) comes from upstream and predates the platform above.
`bench.py` is still the fixed benchmark the platform shells out to; the rest is
kept for the standalone single-kernel workflow. See
[PROVENANCE.md](../PROVENANCE.md) for the per-file split.
