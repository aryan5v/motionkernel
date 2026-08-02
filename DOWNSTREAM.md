# MotionKernel provenance

MotionKernel is an independently maintained, MIT-licensed downstream fork of
[RightNow-AI/AutoKernel](https://github.com/RightNow-AI/autokernel). Its focus
is GPU kernel discovery, optimization, verification, and packaging for video
generation models.

## Provenance

- Upstream repository: `https://github.com/RightNow-AI/autokernel`
- Initial downstream base: `7843582` (`test hf kernels export`)
- License: MIT
- Original copyright: Copyright (c) 2026 RightNow AI

The upstream `LICENSE` file is preserved. Source files substantially derived
from upstream remain covered by that notice. No upstream copyright header has
been removed from any file.

[PROVENANCE.md](PROVENANCE.md) records the per-file split -- which files are
byte-identical to upstream, which are modified descendants, and which are
MotionKernel-original -- generated from Git history so it can be re-checked:

```bash
python scripts/provenance_inventory.py --check
```

As of this release, 45 files are unchanged from upstream, 9 are modified
descendants, and 128 are MotionKernel-original; about 30% of the Python line
count is inherited.

## MotionKernel direction

MotionKernel is intended to become a video-first, framework-agnostic platform
for discovering, testing, tuning, and exporting production GPU kernels. Its
initial work focuses on:

- external custom-operation specifications;
- multi-output, backward, determinism, and compile verification;
- production shape corpora captured from real models;
- modulated normalization, gated-residual, attention, and layout fusion;
- architecture-aware tuning and reproducible experiment records; and
- clean export into runtime kernel packages for FastVideo, Diffusers, and other
  PyTorch video runtimes.

The optimization platform and shipped runtime kernels are separate products:
the platform searches and validates candidates, while downstream applications
consume only promoted kernel implementations.

The initial model families are Wan, LTX-Video, Cosmos, and Kandinsky. Listing a
model as a target does not imply complete support: support is earned through a
published integration, representative workload corpus, correctness results,
and an end-to-end benchmark. The current level for each model, and the evidence
behind it, is in [docs/SUPPORT_STATUS.md](docs/SUPPORT_STATUS.md).

## Naming and the compatibility namespace

| | Name |
|---|---|
| Product | MotionKernel |
| Distribution | `motionkernel` |
| Canonical import namespace | `motionkernel` |
| Compatibility import namespace | `autokernel` |

`autokernel` is a **compatibility namespace** inherited from upstream. It is
not the product name, not a second product, and not a sign that you are running
AutoKernel. It aliases the same modules, so `motionkernel.specs is
autokernel.specs`.

It is *not* pinned by artifacts. Packaged bundles carry `candidate.py`,
`entry.py` and `manifest.json`, none of which import this package, so a rename
would invalidate nothing. It remains for internal reasons -- 139 import sites
and resumable run directories -- set out with their real cost in
[docs/NAMESPACE_MIGRATION.md](docs/NAMESPACE_MIGRATION.md).

Both namespaces are supported and both resolve for type checkers.
`autokernel` is what generated specs and schema-1 artifact bundles import and
is not deprecated; `motionkernel` re-exports the same objects under the
canonical name. See
[docs/NAMESPACE_MIGRATION.md](docs/NAMESPACE_MIGRATION.md) for the four-phase
plan and the gate on each phase.

## Upstream relationship

Useful upstream changes can be incorporated without making upstream a release
dependency:

```bash
git fetch upstream
git switch main
git merge upstream/main
```

MotionKernel features do not require upstream approval. Improvements may still
be offered upstream when doing so benefits both projects.
