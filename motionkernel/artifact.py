"""Canonical alias for :mod:`autokernel.artifact`.

A plain re-export, so type checkers resolve it and the objects are the *same*
objects: ``motionkernel.artifact.X is autokernel.artifact.X``.
"""

from __future__ import annotations

from autokernel.artifact import *  # noqa: F401,F403
from autokernel.artifact import __all__  # noqa: F401
