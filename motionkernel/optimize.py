"""Canonical alias for :mod:`autokernel.optimize`.

A plain re-export, so type checkers resolve it and the objects are the *same*
objects: ``motionkernel.optimize.X is autokernel.optimize.X``.
"""

from __future__ import annotations

from autokernel.optimize import *  # noqa: F401,F403
from autokernel.optimize import __all__  # noqa: F401
