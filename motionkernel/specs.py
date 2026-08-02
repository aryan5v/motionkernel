"""Canonical alias for :mod:`autokernel.specs`.

A plain re-export, so type checkers resolve it and the objects are the *same*
objects: ``motionkernel.specs.X is autokernel.specs.X``.
"""

from __future__ import annotations

from autokernel.specs import *  # noqa: F401,F403
from autokernel.specs import __all__  # noqa: F401
