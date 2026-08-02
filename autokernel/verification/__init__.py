"""Correctness verification for structured kernel outputs.

This package generalizes the benchmark harness beyond single-tensor outputs:

* :mod:`autokernel.verification.outputs` flattens and compares arbitrary
  output trees (tensors, tuples, lists, dictionaries, named tuples and
  nested combinations) leaf by leaf, with stable diagnostic paths.
* :mod:`autokernel.verification.backward` compares declared input gradients.
* :mod:`autokernel.verification.compile` enforces optional full-graph checks.
* :mod:`autokernel.verification.corpus` validates production shape corpora.
* :mod:`autokernel.verification.results` writes versioned result artifacts.

Modules here never initialize a GPU at import time; ``torch`` is imported
lazily inside the functions that need it.
"""

from __future__ import annotations

from .backward import (
    BackwardReport,
    GradientRecord,
    check_backward,
)
from .compile import CompileCaseRecord, CompileReport, check_compile
from .corpus import (
    CORPUS_SCHEMA_VERSION,
    CorpusCase,
    CorpusError,
    ShapeCorpus,
    load_shape_corpus,
    validate_corpus_against_spec,
    weighted_aggregate,
)
from .outputs import (
    DEFAULT_TOLERANCE,
    LeafRecord,
    OutputTreeError,
    TreeComparison,
    compare_deterministic,
    compare_output_trees,
    compare_tensor_leaf,
    flatten_output_tree,
    tree_has_nan_or_inf,
)
from .policy import (
    APPROXIMATE_MATH_MARKERS,
    EXACT_POLICIES,
    KNOWN_POLICIES,
    ParityPolicy,
    ToleranceResolutionError,
    detect_approximate_math,
    resolve_leaf_tolerance,
)
from .results import (
    RESULT_SCHEMA_VERSION,
    collect_environment_metadata,
    result_envelope,
    write_result_atomic,
)

__all__ = [
    "APPROXIMATE_MATH_MARKERS",
    "CORPUS_SCHEMA_VERSION",
    "DEFAULT_TOLERANCE",
    "EXACT_POLICIES",
    "KNOWN_POLICIES",
    "RESULT_SCHEMA_VERSION",
    "BackwardReport",
    "CompileCaseRecord",
    "CompileReport",
    "CorpusCase",
    "CorpusError",
    "GradientRecord",
    "LeafRecord",
    "OutputTreeError",
    "ParityPolicy",
    "ShapeCorpus",
    "ToleranceResolutionError",
    "TreeComparison",
    "check_backward",
    "check_compile",
    "collect_environment_metadata",
    "compare_deterministic",
    "compare_output_trees",
    "compare_tensor_leaf",
    "detect_approximate_math",
    "flatten_output_tree",
    "load_shape_corpus",
    "resolve_leaf_tolerance",
    "result_envelope",
    "tree_has_nan_or_inf",
    "validate_corpus_against_spec",
    "weighted_aggregate",
    "write_result_atomic",
]
