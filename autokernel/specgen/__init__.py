"""Graph-derived, fail-closed kernel specification generation."""

from .generator import (
    DerivedSubregion,
    build_dispatch_contract,
    build_manifest,
    derive_safe_subregion,
    select_region,
    spec_from_manifest,
    write_generated_artifacts,
)
from .ir import (
    ALLOWED_TARGETS,
    ExecutableIR,
    IRInput,
    IRNode,
    SpecGenerationError,
    ValueMeta,
)
from .runtime import execute_ir, load_generated_reference, load_manifest

__all__ = [
    "ALLOWED_TARGETS",
    "DerivedSubregion",
    "ExecutableIR",
    "IRInput",
    "IRNode",
    "SpecGenerationError",
    "ValueMeta",
    "build_manifest",
    "build_dispatch_contract",
    "derive_safe_subregion",
    "execute_ir",
    "load_generated_reference",
    "load_manifest",
    "select_region",
    "spec_from_manifest",
    "write_generated_artifacts",
]
