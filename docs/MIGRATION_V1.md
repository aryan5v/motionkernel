# Migrating to MotionKernel V1

Short version: **nothing breaks.** V1 adds a canonical import namespace and
tightens several gates. No existing import, artifact, or workload stops
working.

## Imports

Both namespaces are supported and both resolve for type checkers.

```python
from autokernel.specs import KernelSpec      # unchanged, not deprecated
from motionkernel.specs import KernelSpec    # canonical name, same object
```

`motionkernel.specs.KernelSpec is autokernel.specs.KernelSpec`, so `isinstance`
works across both. `autokernel` is a compatibility namespace **by name, not by
support level**: it is what generated `spec.py` files import and what schema-1
artifact bundles expect, and it is not going away in V1.

One difference to be aware of: the canonical namespace re-exports each
subpackage's public API flat, so `from motionkernel.verification import
ParityPolicy` works but `import motionkernel.verification.policy` does not.
Deep module paths remain available under `autokernel`.

## Artifacts

Schema-1 bundles are unchanged and load as before. `autokernel.specgen` keeps
its name for exactly this reason.

## Behaviour changes worth knowing

These tighten correctness rather than change an API, but they can turn a run
that previously "passed" into one that reports a failure — which is the point.

| Change | What you may notice |
|---|---|
| The workload's `parity.policy` now governs kernel-level gates, not just the final frame comparison | A `byte_equal` workload rejects approximate-math kernels that previously passed on a 1e-2 tolerance |
| Packaging is gated on measured end-to-end impact, not an optimistic upper bound | Candidates whose measured savings cannot reach `min_end_to_end_speedup` are no longer packaged |
| An undeclared dtype tolerance raises instead of defaulting to 1e-2/1e-2 | A spec with `tolerances: null` now fails loudly instead of being silently permissive |
| Non-finite handling is stricter | A candidate NaN where the reference is finite fails regardless of tolerance |

If a campaign that used to promote an artifact now quarantines it, read the
reason string: it names the gate and the measurement.

## Nothing to do

There is no migration step. If you want the canonical name in new code, use
it; otherwise keep `autokernel`.
