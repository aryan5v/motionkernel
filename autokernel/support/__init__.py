"""Support matrix: evidence in, generated page out, nothing by hand."""

from .evidence import (
    OUTCOMES,
    RECORD_SCHEMA,
    RECORD_SCHEMA_VERSION,
    RunRecord,
    SupportEvidenceError,
    load_run_record,
    record_filename,
    record_from_receipt,
    write_run_record,
)
from .matrix import (
    DEFAULT_ARCHES,
    DEFAULT_STALE_DAYS,
    MATRIX_SCHEMA,
    MATRIX_SCHEMA_VERSION,
    MatrixError,
    build_cells,
    generate_matrix,
    load_records,
    load_rows,
)
from .nightly import NightlyConfig, NightlyError, NightlyTarget, run_nightly

__all__ = [
    "DEFAULT_ARCHES",
    "DEFAULT_STALE_DAYS",
    "MATRIX_SCHEMA",
    "MATRIX_SCHEMA_VERSION",
    "MatrixError",
    "NightlyConfig",
    "NightlyError",
    "NightlyTarget",
    "OUTCOMES",
    "RECORD_SCHEMA",
    "RECORD_SCHEMA_VERSION",
    "RunRecord",
    "SupportEvidenceError",
    "build_cells",
    "generate_matrix",
    "load_records",
    "load_run_record",
    "load_rows",
    "record_filename",
    "record_from_receipt",
    "run_nightly",
    "write_run_record",
]
