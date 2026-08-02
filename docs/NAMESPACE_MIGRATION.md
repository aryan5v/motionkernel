# Python namespace migration

## The situation

| | Name |
|---|---|
| Product | MotionKernel |
| Distribution (PyPI) | `motionkernel` |
| Console script | `motionkernel` |
| Canonical import namespace | `motionkernel` (phase 1, this release) |
| Compatibility import namespace | `autokernel` |

**`autokernel` is a compatibility namespace, not the product name.** It is
inherited from [RightNow-AI/AutoKernel](https://github.com/RightNow-AI/autokernel),
the upstream project this one forked from. Seeing it in an import does not mean
you are using AutoKernel.

## What a rename would actually cost

An earlier version of this document claimed the namespace was pinned by
hash-verified artifacts. **That was wrong**, and the correction matters because
it was the main argument for keeping `autokernel`.

A packaged bundle contains `candidate.py`, `entry.py` and `manifest.json`, and
none of them import this package:

```
candidate.py   import torch, triton, triton.language
entry.py       import importlib.util, sys, pathlib
```

The only occurrence of the string is `"autokernel_runtime_candidate"` in
`entry.py`, which is a synthetic `sys.modules` key passed to
`spec_from_file_location`, not an import. The generated `spec.py` that does
`from autokernel.specgen import spec_from_manifest` lives in the candidate
*search workspace* and is never packaged, hashed, or shipped.

**A rename would not invalidate any artifact, promoted or otherwise.**

The genuine cost is internal and bounded:

| What | Scale |
|---|---|
| Import sites in the repository | 139 lines (85 in tests, 35 in the package, ~19 in root harness and examples) |
| Inherited-upstream files affected | 2 (`bench.py`, `extract.py`), both already modified descendants |
| Emitted import in future generated specs | one string in `specgen/generator.py` |
| Resumable run directories | existing `spec.py` files would stop importing |
| Externally released versions to keep working | none — the distribution has never been tagged or published |

That last row is the important one. A rename is cheapest now, before the first
public release, and becomes genuinely expensive afterwards.

## Phase 1 (this release)

`motionkernel` is importable and is the documented namespace for new code:

```python
from motionkernel.specs import KernelSpec, Tolerance
from motionkernel.verification.policy import ParityPolicy
```

It is an **alias, not a copy**. `motionkernel.specs` and `autokernel.specs`
resolve to the same module object:

```python
>>> import motionkernel.specs, autokernel.specs
>>> motionkernel.specs is autokernel.specs
True
```

That identity is the whole point. A duplicated package would give callers two
spec registries and two sets of class objects, so an `isinstance` check against
`motionkernel.specs.KernelSpec` would fail for an object built by
`autokernel.specs` — a subtle breakage far worse than the cosmetic problem it
would be solving.

What phase 1 deliberately does **not** do:

- it does not rewrite any existing import;
- it does not change what generated artifacts emit;
- it does not deprecate `autokernel` or emit any warning;
- it does not change any on-disk artifact format or hash.

Implementation: `motionkernel/` contains one ordinary module per public
subpackage, each re-exporting `autokernel.<x>`. No import hook, no
`sys.meta_path` mutation, no import-time side effects beyond importing the
package it re-exports.

### Why not an import hook

An earlier revision resolved `motionkernel.<x>` through a `sys.meta_path`
finder. It worked at run time and preserved module identity, but a runtime
finder is invisible to static analysis by construction — so the namespace this
project was recommending was the one that lost type checking and IDE
completion. Measured against a clean install of the wheel:

| import | with the finder | with plain modules |
|---|---|---|
| `from autokernel.specs import Tolerance` | resolves | resolves |
| `from motionkernel.specs import Tolerance` | `Cannot find implementation or library stub`, type `Any` | resolves, `autokernel.specs.types.Tolerance` |

The finder also mutated `sys.meta_path` on import, reported `__name__` as
`autokernel.specs`, and returned nothing from `pkgutil.iter_modules`.

### What the re-export gives up

**Module identity.** `motionkernel.specs is autokernel.specs` is now `False`.
Nothing depends on that. What callers depend on is *class* identity —
`motionkernel.specs.KernelSpec is autokernel.specs.KernelSpec` — which holds,
because these modules re-export the same objects rather than redefining them.
`isinstance` works across both namespaces, and a test enforces it.

**Deep module paths.** `motionkernel.verification.policy` is not importable;
`motionkernel.verification` is a module, not a package. Each subpackage's
public API is re-exported flat, so `from motionkernel.verification import
ParityPolicy` works. Deep paths remain available under `autokernel`, which is
fully supported.

## Later phases

Each phase is gated on the previous one shipping, not on a date.

**Phase 2 — generation switches.** Artifact generation emits
`from motionkernel.specgen import ...` behind an artifact-manifest schema
version bump. Bundles at the old schema keep verifying and keep importing
`autokernel`; new bundles use the canonical name. Both are loadable by the same
runtime. Gate: the artifact loader must accept both schema versions, with tests
covering a bundle of each.

**Phase 3 — soft deprecation.** `import autokernel` emits a
`DeprecationWarning` naming the replacement. This release must overlap phase 2
by at least one minor version so that anyone upgrading sees a working
`motionkernel` before they see a warning about `autokernel`. Gate: no
first-party code imports `autokernel`.

**Phase 4 — removal.** `autokernel` becomes a thin shim or is removed. No
earlier than a major version bump, and not before the oldest artifact schema
still in the wild has been re-issued. Gate: an announced deprecation window has
elapsed.

## For downstream users

Nothing is required of you in phase 1, and nothing is deprecated.

`autokernel` is the namespace to use today. It is fully supported, it is what
type checkers and IDEs resolve, and it is what every generated artifact
imports. It is a compatibility namespace by *name*, not by support level.

`motionkernel` is available and works at runtime. Use it if you prefer the
canonical name and do not depend on static analysis of these imports. It
becomes the recommended namespace at phase 2, when the modules exist as real
files and type checkers can see them.

Artifact bundles are unaffected in phase 1 and remain loadable across phase 2.

## Enforcement

`tests/test_public_metadata.py` asserts that the alias resolves to the same
module objects, that the distribution and package versions agree, and that the
compatibility namespace still imports — so the two namespaces cannot silently
drift apart.
