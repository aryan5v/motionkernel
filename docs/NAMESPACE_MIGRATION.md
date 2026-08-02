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

## Why it cannot simply be renamed

Every generated `spec.py` MotionKernel has ever emitted contains:

```python
from autokernel.specgen import spec_from_manifest
SPEC = spec_from_manifest(Path(__file__).with_name("manifest.json"))
```

Packaged artifact bundles are hash-verified: `artifact.json` pins the SHA-256 of
every file it declares, and the runtime refuses to load a bundle whose bytes
have changed. Rewriting that import would therefore invalidate the manifest of
every artifact already produced — including promoted ones already dispatched in
production — and there is no way for a consumer to repair them, because
repairing them is exactly the change the hash is there to detect.

A mass `sed` across the repository would also break any downstream user who
imports `autokernel` today, for no functional gain.

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

Implementation: `motionkernel/__init__.py` installs a `MetaPathFinder` that
resolves `motionkernel.<x>` to the already-imported `autokernel.<x>`. It
declines cleanly for names it does not own, so unrelated imports are unaffected.

### Known limitations of the alias

These are measured, not theoretical, and they are why `autokernel` remains the
namespace this project points production users at for now.

**Static analysis cannot see it.** Type checkers resolve modules from the
filesystem; a runtime `MetaPathFinder` is invisible to them. Against a clean
install of the wheel:

| import | mypy result | `reveal_type` |
|---|---|---|
| `from autokernel.specs import Tolerance` | resolves | `autokernel.specs.types.Tolerance` |
| `from motionkernel.specs import Tolerance` | `Cannot find implementation or library stub` | `Any` |

Both packages ship `py.typed`, which is what makes the first row work. The
second row cannot be fixed without real files on disk. Anyone who type-checks
their code, or relies on IDE completion, is better served by `autokernel` until
phase 2.

**Submodule discovery returns nothing.** `pkgutil.iter_modules` over
`motionkernel.__path__` lists no submodules, because the real ones live under
`autokernel/`. Documentation generators and plugin scanners that enumerate a
package will find an empty one.

**`__name__` reports the compatibility name.** `motionkernel.specs.__name__` is
`"autokernel.specs"`, and classes defined there have
`__module__ == "autokernel.specs.types"`. This is deliberate — it keeps
pickles stable across both import paths — but it surprises anyone reading a
traceback or a `repr`.

None of these affect runtime behaviour: imports, class identity, `isinstance`,
and pickling all work correctly through either name.

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
