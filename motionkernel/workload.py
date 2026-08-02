"""Canonical alias for :mod:`autokernel.workload`.

A plain re-export, so type checkers resolve it and the objects are the *same*
objects: ``motionkernel.workload.X is autokernel.workload.X``.
"""

from __future__ import annotations

from autokernel.workload import *  # noqa: F401,F403
from autokernel.workload import __all__  # noqa: F401
