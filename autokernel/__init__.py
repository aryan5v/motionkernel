"""MotionKernel platform core, under its compatibility import namespace.

The product is **MotionKernel**; the distribution on PyPI is ``motionkernel``.
``autokernel`` is a *compatibility namespace* inherited from the upstream
project, not a second product and not the current name of this one.

It still holds the implementation for internal reasons rather than
compatibility ones. Packaged artifact bundles do not import this package --
they carry only ``candidate.py``, ``entry.py`` and ``manifest.json`` -- so a
rename would not invalidate any artifact. What a rename does touch is 139
import sites, the import emitted into future generated specs, and resumable run
directories.

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
