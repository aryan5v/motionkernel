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

Nothing is required of you in phase 1. If you are writing new code, prefer
`motionkernel`; if you have existing code importing `autokernel`, it will keep
working and will warn you well before it stops.

Artifact bundles are unaffected in phase 1 and remain loadable across phase 2.

## Enforcement

`tests/test_public_metadata.py` asserts that the alias resolves to the same
module objects, that the distribution and package versions agree, and that the
compatibility namespace still imports — so the two namespaces cannot silently
drift apart.
