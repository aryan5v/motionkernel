"""MotionKernel: the canonical import namespace.

The product is MotionKernel and the distribution is ``motionkernel``. The
implementation lives under ``autokernel``, a compatibility namespace inherited
from the upstream project this one forked from -- a name, not a support level.

This package re-exports it under the canonical name using ordinary Python
modules, one per public subpackage:

    >>> from motionkernel.specs import KernelSpec
    >>> from motionkernel.verification.policy import ParityPolicy

Why plain modules and not an import hook
----------------------------------------
An earlier revision resolved ``motionkernel.<x>`` through a
``sys.meta_path`` finder. It worked at run time and preserved module identity,
but a runtime finder is invisible to static analysis by construction, so the
namespace this project was telling people to prefer was the one that lost type
checking and IDE completion. Measured against a clean install of the wheel:

    from autokernel.specs   import Tolerance  -> mypy resolves it
    from motionkernel.specs import Tolerance  -> "Cannot find implementation
                                                  or library stub", type Any

It also mutated ``sys.meta_path`` on import, reported ``__name__`` as
``autokernel.specs``, and returned nothing from ``pkgutil.iter_modules``.

Real modules have none of those problems. What they give up is *module*
identity: ``motionkernel.specs is autokernel.specs`` is now False. Nothing
depends on that. What callers actually depend on is *class* identity --
``motionkernel.specs.KernelSpec is autokernel.specs.KernelSpec`` -- which holds,
because these modules re-export the same objects rather than redefining them.
``isinstance`` therefore works across both namespaces.

Both namespaces are supported. ``autokernel`` is not deprecated and emits no
warning; it is what every generated ``spec.py`` imports and what schema-1
artifact bundles expect. See ``docs/NAMESPACE_MIGRATION.md``.

Nothing here may initialize a GPU at import time.
"""

from __future__ import annotations

import autokernel as _autokernel

#: The compatibility namespace this package re-exports.
COMPATIBILITY_NAMESPACE = "autokernel"

#: The canonical namespace.
CANONICAL_NAMESPACE = "motionkernel"

#: Single source of truth, read from the package this one re-exports.
__version__ = _autokernel.__version__

__all__ = [
    "CANONICAL_NAMESPACE",
    "COMPATIBILITY_NAMESPACE",
    "__version__",
]
