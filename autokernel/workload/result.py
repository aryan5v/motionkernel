"""Structured generation-run result schema written by FastVideo launchers.

MotionKernel validates these artifacts before ranking native-versus-optimized
end-to-end outcomes. Values are metadata and numeric measurements only; frame
tensors live in separate ``.npy`` files referenced by path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from autokernel._io import write_json_atomic

from ._validate import (
    finite_number,
)
from ._validate import (
    mapping as _mapping_base,
)
from ._validate import (
    non_negative_int as _non_negative_int_base,
)
from ._validate import (
    optional_text as _optional_text_base,
)
from ._validate import (
    positive_int as _positive_int_base,
)
from ._validate import (
    text as _text_base,
)
from .types import FORBIDDEN_METADATA_KEYS, WorkloadError

RESULT_SCHEMA_VERSION = 1

#: Comparison payloads (classify_end_to_end output) carry this version.
#: v2 adds the variance block: per-arm coefficients of variation, the
#: declared ceiling, and whether the measurement is valid for gating.
COMPARISON_SCHEMA_VERSION = 2

#: Declared ceiling for the native-arm coefficient of variation. A timed A/B
#: whose native arm varies more than this is node contention, not evidence
#: (R4 section 3: a 38.4% native spread reversed per-artifact verdicts).
DEFAULT_CV_CEILING = 0.03

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "mode",
    "status",
    "workload_id",
    "model_id",
    "request",
    "warmups",
    "runs",
    "wall_seconds",
    "median_wall_seconds",
    "generation_seconds",
    "peak_memory_mb",
    "environment",
    "frames_path",
    "profiler_path",
    "log_path",
    "failure_reason",
    "stage",
}

_STATUSES = {"ok", "failed", "skipped"}
_MODES = {"native", "optimized", "fused", "candidate"}


def _kind() -> str:
    return "generation result"


def _text(value: Any, source: object, location: str) -> str:
    try:
        return _text_base(value, source, location, kind=_kind())
    except Exception as exc:  # SchemaError subclass of ValueError
        raise WorkloadError(str(exc)) from exc


def _optional_text(
    value: Any, source: object, location: str
) -> str | None:
    try:
        return _optional_text_base(value, source, location, kind=_kind())
    except Exception as exc:
        raise WorkloadError(str(exc)) from exc


def _non_negative_int(value: Any, source: object, location: str) -> int:
    try:
        return _non_negative_int_base(value, source, location, kind=_kind())
    except Exception as exc:
        raise WorkloadError(str(exc)) from exc


def _positive_int(value: Any, source: object, location: str) -> int:
    try:
        return _positive_int_base(value, source, location, kind=_kind())
    except Exception as exc:
        raise WorkloadError(str(exc)) from exc


def _mapping(value: Any, source: object, location: str, *, non_empty: bool = False):
    try:
        return _mapping_base(
            value,
            source,
            location,
            kind=_kind(),
            non_empty=non_empty,
            forbidden_keys=FORBIDDEN_METADATA_KEYS,
        )
    except Exception as exc:
        raise WorkloadError(str(exc)) from exc


def _finite_number(
    value: Any,
    source: object,
    location: str,
    *,
    minimum: float | None = None,
) -> float:
    try:
        return finite_number(
            value, source, location, kind=_kind(), minimum=minimum
        )
    except Exception as exc:
        raise WorkloadError(str(exc)) from exc


def _number_list(
    value: Any, source: object, location: str
) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise WorkloadError(
            f"generation result {source!r}: {location}: must be a list"
        )
    numbers: list[float] = []
    for index, item in enumerate(value):
        if item is None:
            raise WorkloadError(
                f"generation result {source!r}: {location}[{index}]: "
                "must be a finite non-negative number (None not allowed)"
            )
        numbers.append(
            _finite_number(
                item,
                source,
                f"{location}[{index}]",
                minimum=0.0,
            )
        )
    return numbers


def _optional_number_list(
    value: Any, source: object, location: str
) -> list[float | None]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise WorkloadError(
            f"generation result {source!r}: {location}: must be a list"
        )
    result: list[float | None] = []
    for index, item in enumerate(value):
        if item is None:
            result.append(None)
            continue
        result.append(
            _finite_number(
                item,
                source,
                f"{location}[{index}]",
                minimum=0.0,
            )
        )
    return result


@dataclass(frozen=True)
class GenerationRunResult:
    """One mode's measured generation run (native or optimized)."""

    mode: str
    status: str
    workload_id: str
    model_id: str
    request: Mapping[str, Any]
    warmups: int
    runs: int
    wall_seconds: tuple[float, ...]
    median_wall_seconds: float | None
    generation_seconds: tuple[float | None, ...]
    peak_memory_mb: tuple[float | None, ...]
    environment: Mapping[str, Any]
    frames_path: str | None = None
    profiler_path: str | None = None
    log_path: str | None = None
    failure_reason: str | None = None
    stage: str = "generate"
    schema_version: int = RESULT_SCHEMA_VERSION
    source: str = "<memory>"

    @classmethod
    def from_dict(
        cls, raw_value: Any, *, source: object = "<memory>"
    ) -> "GenerationRunResult":
        raw = _mapping(raw_value, source, "top level", non_empty=True)
        unknown = sorted(set(raw) - _TOP_LEVEL_FIELDS)
        if unknown:
            raise WorkloadError(
                f"generation result {source!r}: top level: "
                f"unknown field(s) {unknown}"
            )

        version = raw.get("schema_version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise WorkloadError(
                f"generation result {source!r}: schema_version: "
                "must be an integer"
            )
        if version != RESULT_SCHEMA_VERSION:
            raise WorkloadError(
                f"generation result {source!r}: schema_version: "
                f"unsupported version {version}; expected {RESULT_SCHEMA_VERSION}"
            )

        mode = _text(raw.get("mode"), source, "mode")
        if mode not in _MODES:
            raise WorkloadError(
                f"generation result {source!r}: mode: "
                f"must be one of {sorted(_MODES)}"
            )
        status = _text(raw.get("status", "ok"), source, "status")
        if status not in _STATUSES:
            raise WorkloadError(
                f"generation result {source!r}: status: "
                f"must be one of {sorted(_STATUSES)}"
            )

        wall = tuple(_number_list(raw.get("wall_seconds", []), source, "wall_seconds"))
        median = raw.get("median_wall_seconds")
        median_value = (
            None
            if median is None
            else _finite_number(
                median, source, "median_wall_seconds", minimum=0.0
            )
        )
        if status == "ok" and not wall:
            raise WorkloadError(
                f"generation result {source!r}: wall_seconds: "
                "required for successful runs"
            )

        return cls(
            mode=mode,
            status=status,
            workload_id=_text(raw.get("workload_id"), source, "workload_id"),
            model_id=_text(raw.get("model_id"), source, "model_id"),
            request=dict(
                _mapping(raw.get("request", {}), source, "request")
            ),
            warmups=_non_negative_int(raw.get("warmups", 0), source, "warmups"),
            runs=_positive_int(raw.get("runs", 1), source, "runs")
            if status == "ok"
            else _non_negative_int(raw.get("runs", 0), source, "runs"),
            wall_seconds=wall,
            median_wall_seconds=median_value,
            generation_seconds=tuple(
                _optional_number_list(
                    raw.get("generation_seconds", []),
                    source,
                    "generation_seconds",
                )
            ),
            peak_memory_mb=tuple(
                _optional_number_list(
                    raw.get("peak_memory_mb", []),
                    source,
                    "peak_memory_mb",
                )
            ),
            environment=dict(
                _mapping(raw.get("environment", {}), source, "environment")
            ),
            frames_path=_optional_text(
                raw.get("frames_path"), source, "frames_path"
            ),
            profiler_path=_optional_text(
                raw.get("profiler_path"), source, "profiler_path"
            ),
            log_path=_optional_text(raw.get("log_path"), source, "log_path"),
            failure_reason=_optional_text(
                raw.get("failure_reason"), source, "failure_reason"
            ),
            stage=_text(raw.get("stage", "generate"), source, "stage"),
            schema_version=version,
            source=str(source),
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "status": self.status,
            "workload_id": self.workload_id,
            "model_id": self.model_id,
            "request": dict(self.request),
            "warmups": self.warmups,
            "runs": self.runs,
            "wall_seconds": list(self.wall_seconds),
            "median_wall_seconds": self.median_wall_seconds,
            "generation_seconds": list(self.generation_seconds),
            "peak_memory_mb": list(self.peak_memory_mb),
            "environment": dict(self.environment),
            "stage": self.stage,
        }
        if self.frames_path is not None:
            payload["frames_path"] = self.frames_path
        if self.profiler_path is not None:
            payload["profiler_path"] = self.profiler_path
        if self.log_path is not None:
            payload["log_path"] = self.log_path
        if self.failure_reason is not None:
            payload["failure_reason"] = self.failure_reason
        return payload


def load_generation_result(path: str | Path) -> GenerationRunResult:
    """Load and validate a launcher result JSON file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise WorkloadError(f"generation result {file_path!s}: file: not found")
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkloadError(
            f"generation result {file_path!s}: JSON: invalid JSON: {exc}"
        ) from exc
    return GenerationRunResult.from_dict(raw, source=str(file_path))


def write_generation_result(
    result: GenerationRunResult, path: str | Path
) -> None:
    """Atomically write a generation result JSON file."""
    write_json_atomic(path, result.as_dict())


def compare_frame_outputs(
    native_path: str | Path | None,
    optimized_path: str | Path | None,
    *,
    policy: str = "byte_equal",
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, Any]:
    """Compare saved full-generation frame arrays under the workload policy."""
    if native_path is None or optimized_path is None:
        return {
            "passed": False,
            "policy": policy,
            "reason": "one or both frame artifacts are missing",
        }
    native_file = Path(native_path)
    optimized_file = Path(optimized_path)
    if not native_file.is_file() or not optimized_file.is_file():
        return {
            "passed": False,
            "policy": policy,
            "reason": "one or both frame artifact paths do not exist",
        }

    import numpy as np

    native = np.load(native_file, allow_pickle=False, mmap_mode="r")
    optimized = np.load(optimized_file, allow_pickle=False, mmap_mode="r")
    same_shape = native.shape == optimized.shape
    result: dict[str, Any] = {
        "policy": policy,
        "native_shape": list(native.shape),
        "optimized_shape": list(optimized.shape),
        "native_dtype": str(native.dtype),
        "optimized_dtype": str(optimized.dtype),
        "same_shape": same_shape,
    }
    if policy == "frames_only":
        result["passed"] = same_shape
        result["reason"] = (
            "frame shapes match" if same_shape else "frame shapes differ"
        )
        return result
    if not same_shape:
        result["passed"] = False
        result["reason"] = "frame shapes differ"
        return result

    if policy == "byte_equal":
        passed = bool(
            native.dtype == optimized.dtype and np.array_equal(native, optimized)
        )
        result["passed"] = passed
        result["reason"] = (
            "frame arrays are byte equal"
            if passed
            else "frame arrays are not byte equal"
        )
        return result
    if policy == "tolerance":
        difference = np.abs(native - optimized)
        result["max_abs_diff"] = (
            float(difference.max()) if difference.size else 0.0
        )
        result["mean_abs_diff"] = (
            float(difference.mean()) if difference.size else 0.0
        )
        passed = bool(
            np.allclose(
                native,
                optimized,
                atol=atol,
                rtol=rtol,
                equal_nan=False,
            )
        )
        result.update(
            {
                "passed": passed,
                "atol": atol,
                "rtol": rtol,
                "reason": (
                    "frame arrays are within tolerance"
                    if passed
                    else "frame arrays exceed tolerance"
                ),
            }
        )
        return result
    return {
        **result,
        "passed": False,
        "reason": f"unsupported parity policy {policy!r}",
    }



def coefficient_of_variation(values: Sequence[float]) -> float | None:
    """stdev/mean of one arm's wall times; None when it cannot be measured."""
    import statistics

    samples = [float(value) for value in values]
    if len(samples) < 2:
        return None
    mean = statistics.fmean(samples)
    if mean <= 0:
        return None
    return statistics.stdev(samples) / mean


def _variance_block(
    native: "GenerationRunResult",
    optimized: "GenerationRunResult",
    *,
    cv_ceiling: float,
) -> dict[str, Any]:
    native_cv = coefficient_of_variation(native.wall_seconds)
    optimized_cv = coefficient_of_variation(optimized.wall_seconds)
    if native_cv is None:
        valid, reason = True, "native-arm variance not measured (fewer than two runs)"
    elif native_cv > cv_ceiling:
        valid = False
        reason = (
            f"native-arm coefficient of variation {native_cv:.4f} exceeds the "
            f"{cv_ceiling:.2f} ceiling; measurement is node contention, not "
            "promotion evidence"
        )
    else:
        valid, reason = True, (
            f"native-arm coefficient of variation {native_cv:.4f} within the "
            f"{cv_ceiling:.2f} ceiling"
        )
    return {
        "native_cv": native_cv,
        "optimized_cv": optimized_cv,
        "cv_ceiling": cv_ceiling,
        "variance_valid": valid,
        "variance_reason": reason,
    }


def classify_end_to_end(
    native: GenerationRunResult,
    optimized: GenerationRunResult,
    *,
    min_speedup: float = 1.01,
    max_peak_memory_regression: float = 0.05,
    cv_ceiling: float = DEFAULT_CV_CEILING,
) -> dict[str, Any]:
    """Classify a native-versus-optimized pair without claiming microbench wins."""
    if native.status != "ok" or optimized.status != "ok":
        return {
            "classification": "failed",
            "reason": "one or both modes failed",
            "native_status": native.status,
            "optimized_status": optimized.status,
        }
    if (
        native.median_wall_seconds is None
        or optimized.median_wall_seconds is None
        or native.median_wall_seconds <= 0
        or optimized.median_wall_seconds <= 0
    ):
        return {
            "classification": "failed",
            "reason": "missing or non-positive median wall times",
        }

    speedup = native.median_wall_seconds / optimized.median_wall_seconds
    native_mem = [m for m in native.peak_memory_mb if m is not None]
    opt_mem = [m for m in optimized.peak_memory_mb if m is not None]
    memory_regression = None
    if native_mem and opt_mem:
        native_peak = max(native_mem)
        opt_peak = max(opt_mem)
        if native_peak > 0:
            memory_regression = (opt_peak - native_peak) / native_peak

    if memory_regression is not None and memory_regression > max_peak_memory_regression:
        classification = "regressed"
        reason = "peak memory regression exceeds threshold"
    elif speedup >= min_speedup:
        classification = "improved"
        reason = "repeatable end-to-end wall-time improvement"
    elif speedup <= (1.0 / min_speedup):
        classification = "regressed"
        reason = "end-to-end wall time regressed"
    else:
        classification = "neutral"
        reason = "change within timing noise / below promotion threshold"

    return {
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "classification": classification,
        "reason": reason,
        "native_median_wall_seconds": native.median_wall_seconds,
        "optimized_median_wall_seconds": optimized.median_wall_seconds,
        "end_to_end_speedup": speedup,
        "min_end_to_end_speedup": min_speedup,
        "peak_memory_regression": memory_regression,
        "max_peak_memory_regression": max_peak_memory_regression,
        **_variance_block(native, optimized, cv_ceiling=cv_ceiling),
    }
