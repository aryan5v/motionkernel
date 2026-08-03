"""Dispatch-overhead measurement, attribution, and break-even analysis.

The public surface is intentionally small: :func:`run_dispatch_measurement`
produces the evidence, :func:`attribute_overhead` and
:func:`overhead_from_e2e` turn it into the published number, and
:func:`breakeven_curve` derives what other tracks need from it.
"""

from .controls import DEFAULT_CV_CEILING, capture_controls, variance_block
from .paired import (
    MEASUREMENT_SCHEMA_VERSION_PAIRED,
    PROTOCOL_PAIRED,
    PROTOCOL_SEQUENTIAL,
    PairedResult,
    ProtocolError,
    bootstrap_median_ci,
    interleaved_schedule,
    paired_speedups,
    summarize_paired,
    wilcoxon_signed_rank_p,
)
from .warmup import WarmupResult, clock_plateaued, sustained_warmup
from .measure import (
    MEASUREMENT_SCHEMA,
    MEASUREMENT_SCHEMA_VERSION,
    MIN_TIMED_RUNS,
    MeasurementError,
    run_dispatch_measurement,
)
from .overhead import (
    DEFAULT_CALL_VOLUMES,
    DEFAULT_GATE,
    BreakEvenPoint,
    DispatchAnalysisError,
    E2EOverhead,
    OverheadAttribution,
    TimingReport,
    attribute_overhead,
    breakeven_curve,
    host_profile_summary,
    load_timing_report,
    overhead_from_e2e,
    required_saving_ms_per_call,
)

__all__ = [
    "wilcoxon_signed_rank_p",
    "sustained_warmup",
    "summarize_paired",
    "paired_speedups",
    "interleaved_schedule",
    "clock_plateaued",
    "bootstrap_median_ci",
    "WarmupResult",
    "ProtocolError",
    "PairedResult",
    "PROTOCOL_SEQUENTIAL",
    "PROTOCOL_PAIRED",
    "MEASUREMENT_SCHEMA_VERSION_PAIRED",
    "BreakEvenPoint",
    "DEFAULT_CALL_VOLUMES",
    "DEFAULT_CV_CEILING",
    "DEFAULT_GATE",
    "DispatchAnalysisError",
    "E2EOverhead",
    "MEASUREMENT_SCHEMA",
    "MEASUREMENT_SCHEMA_VERSION",
    "MIN_TIMED_RUNS",
    "MeasurementError",
    "OverheadAttribution",
    "TimingReport",
    "attribute_overhead",
    "breakeven_curve",
    "capture_controls",
    "variance_block",
    "host_profile_summary",
    "load_timing_report",
    "overhead_from_e2e",
    "required_saving_ms_per_call",
    "run_dispatch_measurement",
]
