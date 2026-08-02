"""MotionKernel: the canonical import namespace (phase 1).

The distribution has been ``motionkernel`` since the project forked, but the
import namespace is still ``autokernel``. That is a compatibility namespace
inherited from upstream, not the product name.

This package is phase 1 of moving to the canonical name. It is an *alias*, not
a copy: ``motionkernel.specs`` and ``autokernel.specs`` resolve to the same
module object, so there is exactly one registry, one set of dataclasses, and
one version of every check. Nothing is duplicated and no import is rewritten.

    >>> import motionkernel, autokernel
    >>> from motionkernel.specs import KernelSpec
    >>> motionkernel.specs is autokernel.specs
    True

Why the rename has not happened yet
-----------------------------------
Not because artifacts pin it. Packaged bundles do not import this package at
all: a bundle is ``candidate.py`` (torch and triton only), ``entry.py``
(importlib, sys, pathlib) and ``manifest.json``. The generated ``spec.py`` that
does ``from autokernel.specgen import ...`` lives in the candidate search
workspace and is never packaged or hashed, so a rename would not invalidate any
artifact.

The real reasons are smaller and internal: 139 import sites across the
repository, the import emitted into future generated specs, and resumable run
directories whose existing ``spec.py`` files would stop importing. A rename is
cheapest before the first release, not after.

Staged migration
----------------
* **Phase 1 (this release).** ``motionkernel`` becomes importable and is the
  documented name for new code. ``autokernel`` keeps working, unchanged and
  un-deprecated. Generated artifacts keep emitting ``autokernel``.
* **Phase 2.** Artifact generation emits ``motionkernel`` behind a manifest
  schema bump, so old bundles keep verifying against the old import while new
  ones use the canonical name.
* **Phase 3.** ``autokernel`` starts emitting :class:`DeprecationWarning` on
  import, with a release that overlaps phase 2 by at least one minor version.
* **Phase 4.** ``autokernel`` is reduced to a thin shim or removed, no earlier
  than a major version bump.

See ``docs/NAMESPACE_MIGRATION.md`` for the full plan and the reasoning behind
each gate.

This module must stay importable on a CPU-only machine and must not initialize
a GPU, exactly like the package it aliases.
"""

from __future__ import annotations

import importlib
import importlib.util  # noqa: F401 - importlib.util is not bound by `import importlib`
import sys
from collections.abc import Sequence
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Any

#: The compatibility namespace this package forwards to.
COMPATIBILITY_NAMESPACE = "autokernel"

#: The canonical namespace. New code should import from here.
CANONICAL_NAMESPACE = "motionkernel"

__all__ = [
    "CANONICAL_NAMESPACE",
    "COMPATIBILITY_NAMESPACE",
    "__version__",
    "resolve_compatibility_name",
]


def resolve_compatibility_name(name: str) -> str | None:
    """Map a ``motionkernel.*`` module name onto its ``autokernel.*`` twin.

    Returns ``None`` for anything that is not a submodule of this package, so
    the finder below declines rather than claiming unrelated imports.
    """
    prefix = f"{CANONICAL_NAMESPACE}."
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix) :]
    if not suffix:
        return None
    return f"{COMPATIBILITY_NAMESPACE}.{suffix}"


class _AliasLoader(Loader):
    """Bind an already-imported ``autokernel`` submodule under the new name."""

    def __init__(self, target: str) -> None:
        self._target = target

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        # Import the real module and hand back that exact object, so the two
        # names share state. Returning a copy would give callers a second
        # registry and a second set of class identities, which would silently
        # break isinstance checks across the two namespaces.
        return importlib.import_module(self._target)

    def exec_module(self, module: ModuleType) -> None:
        # Already executed under its canonical name; re-executing it would run
        # module-level side effects twice.
        return None


class _AliasFinder(MetaPathFinder):
    """Resolve ``motionkernel.<x>`` to the imported ``autokernel.<x>``."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        compatibility_name = resolve_compatibility_name(fullname)
        if compatibility_name is None:
            return None
        try:
            importlib.util.find_spec(compatibility_name)
        except (ImportError, AttributeError, ValueError):
            return None
        return ModuleSpec(fullname, _AliasLoader(compatibility_name))


#: Marks an installed finder. Identity of the *class* is not usable for this:
#: ``importlib.reload(motionkernel)`` rebinds ``_AliasFinder`` to a new class
#: object, so an ``isinstance`` check against the new class misses the finder
#: installed by the previous execution and a duplicate is appended every time.
_FINDER_MARKER = "_motionkernel_alias_finder"


def _install_finder() -> None:
    """Install the alias finder once, idempotently across module reloads."""
    if any(getattr(entry, _FINDER_MARKER, False) for entry in sys.meta_path):
        return
    finder = _AliasFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.append(finder)


_install_finder()


def __getattr__(name: str) -> Any:
    """Forward attribute access, so ``motionkernel.specs`` works eagerly.

    Submodules imported through the finder above land in ``sys.modules`` but
    are not automatically bound as attributes of this package, because the real
    parent package is ``autokernel``. Forwarding here keeps
    ``import motionkernel; motionkernel.specs`` working the way it would for an
    ordinary package.
    """
    if name.startswith("_"):
        raise AttributeError(name)
    try:
        module = importlib.import_module(f"{COMPATIBILITY_NAMESPACE}.{name}")
    except ModuleNotFoundError as exc:
        # Only "this submodule does not exist" may fall through to the
        # attribute lookup below. A submodule that *does* exist and failed --
        # a missing optional dependency, a syntax error, a circular import --
        # must surface its own error. Reporting that as "module 'motionkernel'
        # has no attribute 'workload'" hides the cause during exactly the
        # debugging session where it matters.
        missing = getattr(exc, "name", None)
        if missing not in (f"{COMPATIBILITY_NAMESPACE}.{name}", None):
            raise
    else:
        globals()[name] = module
        return module
    compatibility = importlib.import_module(COMPATIBILITY_NAMESPACE)
    try:
        return getattr(compatibility, name)
    except AttributeError:
        raise AttributeError(
            f"module {CANONICAL_NAMESPACE!r} has no attribute {name!r}"
        ) from None


def __dir__() -> list[str]:
    compatibility = importlib.import_module(COMPATIBILITY_NAMESPACE)
    return sorted(set(__all__) | set(dir(compatibility)))


#: Single source of truth, read from the package this one aliases.
__version__ = importlib.import_module(COMPATIBILITY_NAMESPACE).__version__
