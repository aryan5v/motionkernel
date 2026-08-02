# Contributing to MotionKernel

## Development workflow

1. Start from an up-to-date `main`.
2. Create a focused branch.
3. Keep framework changes separate from generated kernel experiments.
4. Run CPU validation before pushing.
5. Run the relevant GPU correctness and performance suites before promoting a
   kernel.

Do not run autonomous experiments in a checkout containing unrelated or
uncommitted work. Use a disposable clone or Git worktree so an experiment can
be abandoned without affecting development state.

## Validation levels

### CPU baseline

The CPU suite requires no GPU and is expected to pass on every change:

```bash
uv pip install -e ".[dev]"
pytest -q
ruff check .
```

It covers spec and manifest schemas, the parity policy and output comparison,
impact and ranking arithmetic, artifact packaging and finalization, the
optimize control plane, and the public-metadata guards in
`tests/test_public_metadata.py`. Nothing in the package may initialize a GPU at
import time, which is what keeps this suite meaningful.

If you change anything public-facing -- product naming, package metadata,
documentation links, or the compatibility namespace -- run that last file
specifically; it is what stops those from drifting.

### GPU correctness

GPU changes must run the relevant benchmark correctness stages across their
declared shapes, dtypes, layouts, and edge cases. Multi-output or training
operations must also validate every returned tensor and requested gradient.

### Performance

Performance claims must include:

- GPU model and compute capability;
- PyTorch, Triton, CUDA, and driver versions;
- input shapes, dtypes, and layouts;
- warmup and measurement methodology;
- median latency and variance; and
- the exact baseline being compared.

Generated candidates are experimental until their correctness and performance
results are reproducible.

## Provenance rules

This repository is a fork and its provenance is part of its licensing
position. When contributing:

- do not remove or edit the upstream copyright in `LICENSE`, or an upstream
  copyright header in any file;
- do not squash or rewrite historical commits;
- if you add, remove, or substantially modify a file inherited from upstream,
  regenerate the inventory:

  ```bash
  python scripts/provenance_inventory.py --write
  ```

- describe `autokernel` as a compatibility namespace, never as the product.

## Claiming support or performance

A model, framework, or GPU may only be described as supported if it has a row
in [docs/SUPPORT_STATUS.md](docs/SUPPORT_STATUS.md) with a linked evidence
file. Adding a claim means adding the evidence in the same change.

Performance numbers must meet the bar in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md), including that both arms of
a comparison were measured in the same session — on a shared machine the
baseline moves more than most speedups being reported.

## Git safety

- Never force-push shared branches.
- Never push changes to the `upstream` remote.
- Never use a destructive reset outside an isolated experiment branch or
  disposable worktree.
- Preserve the MIT license and upstream attribution.
