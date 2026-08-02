"""Top-level MotionKernel optimize control plane."""

from .preflight import (
    PREFLIGHT_SCHEMA_VERSION,
    RUN_CONTRACT_SCHEMA_VERSION,
    PreflightError,
    PreflightFinding,
    PreflightReport,
    build_run_contract,
    compare_run_contract,
    execute_preflight,
)
from .runner import run_optimize
from .state import OptimizeError
from .types import PIPELINE_STAGES, TERMINAL_STATUSES, OptimizeConfig

__all__ = [
    "PIPELINE_STAGES",
    "PREFLIGHT_SCHEMA_VERSION",
    "RUN_CONTRACT_SCHEMA_VERSION",
    "TERMINAL_STATUSES",
    "OptimizeConfig",
    "OptimizeError",
    "PreflightError",
    "PreflightFinding",
    "PreflightReport",
    "build_run_contract",
    "compare_run_contract",
    "execute_preflight",
    "run_optimize",
]
