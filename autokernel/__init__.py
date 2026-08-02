"""MotionKernel platform core, under its compatibility import namespace.

The product is **MotionKernel**; the distribution on PyPI is ``motionkernel``.
``autokernel`` is a *compatibility namespace* inherited from the upstream
project, not a second product and not the current name of this one.

It is still the namespace that holds the implementation because every
generated ``spec.py`` MotionKernel has emitted contains
``from autokernel.specgen import spec_from_manifest``, and packaged artifact
bundles are hash-verified. Renaming the import would invalidate the manifest of
every artifact already produced, including promoted ones.

New code should import from :mod:`motionkernel`, which aliases this package so
that ``motionkernel.specs is autokernel.specs``. See
``docs/NAMESPACE_MIGRATION.md`` for the staged plan; ``autokernel`` is not
deprecated in this release and emits no warning.

This package holds the reusable, importable core. The command-line entry points
(``bench.py``, ``extract.py``, ``profile.py``, ``verify.py``) live at the
repository root and import from here.

Nothing in this package may initialize a GPU at import time: registry discovery
and specification inspection must work on a CPU-only machine.

Portions derived from the upstream AutoKernel project (MIT License). See
LICENSE and PROVENANCE.md.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
