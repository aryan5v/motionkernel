"""Canonical alias for :mod:`autokernel.specgen`.

A plain re-export, so type checkers resolve it and the objects are the *same*
objects: ``motionkernel.specgen.X is autokernel.specgen.X``.
"""

from __future__ import annotations

from autokernel.specgen import *  # noqa: F401,F403
from autokernel.specgen import __all__  # noqa: F401
