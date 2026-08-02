"""Structured output-tree traversal and comparison.

A kernel may return a single tensor or an arbitrary tree of tensors, tuples,
lists, dictionaries and named tuples. Every leaf receives a stable diagnostic
path::

    output
    output[0]
    output.updated_residual
    output["aux"][1]

Comparison rules:

* the candidate and reference trees must have identical structure (same
  containers, same keys, same leaf kinds);
* tensor leaves must share shape and dtype; floating tensors are compared
  with the tolerance declared for their dtype, non-floating tensors must be
  bitwise equal;
* NaN and infinity are detected per path;
* non-tensor (metadata) leaves must match exactly unless the operation's
  :class:`~autokernel.specs.OutputSpec` disables that comparison;
* nothing is silently dropped: a configured ``included_paths`` entry that
  does not exist is an error, and mismatched structures fail loudly.

This module never initializes a GPU; ``torch`` is imported lazily so the
package stays importable on CPU-only machines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..specs.dtypes import canonical_dtype_name
from ..specs.types import OutputSpec, Tolerance
from .policy import ParityPolicy, resolve_leaf_tolerance

__all__ = [
    "DEFAULT_TOLERANCE",
    "LeafRecord",
    "OutputTreeError",
    "TreeComparison",
    "compare_deterministic",
    "compare_output_trees",
    "compare_tensor_leaf",
    "flatten_output_tree",
    "tree_has_nan_or_inf",
]

#: Historical fallback when a spec declares no tolerance for a leaf dtype.
DEFAULT_TOLERANCE = Tolerance(atol=1e-2, rtol=1e-2)


class OutputTreeError(ValueError):
    """Raised when an output tree cannot be traversed as configured."""


def _torch() -> Any:
    import torch  # local import: keep module import torch-free

    return torch


def _is_tensor(value: Any) -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return False
    return isinstance(value, torch.Tensor)


def _is_namedtuple(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and hasattr(value, "_fields")
        and all(isinstance(field, str) for field in value._fields)
    )


def _dict_key_path(path: str, key: Any) -> str:
    return f'{path}["{key}"]' if isinstance(key, str) else f"{path}[{key!r}]"


def flatten_output_tree(output: Any) -> tuple[tuple[str, Any], ...]:
    """Flatten ``output`` into ``(path, leaf)`` pairs with stable ordering.

    Dictionary keys are visited in sorted order so flattening never depends on
    insertion order. Tensors, empty containers and any other value become
    leaves; non-empty tuples, lists, dicts and named tuples are traversed.
    """
    leaves: list[tuple[str, Any]] = []

    def walk(node: Any, path: str) -> None:
        if _is_tensor(node):
            leaves.append((path, node))
            return
        if _is_namedtuple(node):
            for field in node._fields:
                walk(getattr(node, field), f"{path}.{field}")
            return
        if isinstance(node, Mapping):
            if not node:
                leaves.append((path, node))
                return
            for key in sorted(node, key=lambda k: str(k)):
                walk(node[key], _dict_key_path(path, key))
            return
        if isinstance(node, (tuple, list)):
            if not node:
                leaves.append((path, node))
                return
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]")
            return
        leaves.append((path, node))

    walk(output, "output")
    return tuple(leaves)


def _leaf_kind(leaf: Any) -> str:
    return "tensor" if _is_tensor(leaf) else "metadata"


def _filter_leaves(
    leaves: Iterable[tuple[str, Any]],
    included_paths: tuple[str, ...] | None,
    *,
    side: str,
) -> list[tuple[str, Any]]:
    pairs = list(leaves)
    if included_paths is None:
        return pairs
    available = {path for path, _ in pairs}
    missing = [path for path in included_paths if path not in available]
    if missing:
        raise OutputTreeError(
            f"output_spec.included_paths not present in the {side} output: "
            f"{missing}; available paths: {sorted(available)}"
        )
    included = set(included_paths)
    return [(path, leaf) for path, leaf in pairs if path in included]



@dataclass(frozen=True)
class LeafRecord:
    """Comparison outcome for one output leaf."""

    path: str
    kind: str  # "tensor" | "metadata"
    match: bool
    reason: str = ""
    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    pct_within_tol: float | None = None
    has_nan: bool = False
    has_inf: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "match": self.match,
            "reason": self.reason,
            "max_abs_error": self.max_abs_error,
            "mean_abs_error": self.mean_abs_error,
            "pct_within_tol": self.pct_within_tol,
            "has_nan": self.has_nan,
            "has_inf": self.has_inf,
        }


@dataclass(frozen=True)
class TreeComparison:
    """Aggregated comparison of two output trees."""

    match: bool
    reason: str
    leaves: tuple[LeafRecord, ...]
    structure_match: bool = True

    @property
    def worst_abs_error(self) -> float:
        """Largest per-leaf maximum absolute error (0.0 when none)."""
        worst = 0.0
        for leaf in self.leaves:
            if leaf.max_abs_error is not None:
                worst = max(worst, leaf.max_abs_error)
        return worst

    @property
    def worst_pct_within_tol(self) -> float:
        """Smallest per-leaf within-tolerance percentage (100.0 when none)."""
        worst = 100.0
        for leaf in self.leaves:
            if leaf.pct_within_tol is not None:
                worst = min(worst, leaf.pct_within_tol)
        return worst

    def has_nan_or_inf(self) -> bool:
        return any(leaf.has_nan or leaf.has_inf for leaf in self.leaves)

    def first_failure(self) -> LeafRecord | None:
        for leaf in self.leaves:
            if not leaf.match:
                return leaf
        return None

    def leaf_records(self) -> list[dict[str, Any]]:
        return [leaf.as_dict() for leaf in self.leaves]


def _structure_mismatch_reason(
    candidate: Sequence[tuple[str, Any]], expected: Sequence[tuple[str, Any]]
) -> str | None:
    cand_map = {path: leaf for path, leaf in candidate}
    exp_map = {path: leaf for path, leaf in expected}
    extra = [path for path, _ in candidate if path not in exp_map]
    missing = [path for path, _ in expected if path not in cand_map]
    parts = []
    if missing:
        parts.append(f"missing output path(s) {missing}")
    if extra:
        parts.append(f"unexpected output path(s) {extra}")
    if parts:
        return "; ".join(parts)
    for path, exp_leaf in expected:
        cand_leaf = cand_map[path]
        if _leaf_kind(cand_leaf) != _leaf_kind(exp_leaf):
            return (
                f"leaf kind mismatch at {path}: "
                f"{_leaf_kind(cand_leaf)} vs {_leaf_kind(exp_leaf)}"
            )
    return None


def _tolerance_for(
    leaf: Any,
    tolerances: Mapping[str, Tolerance],
    default: Tolerance,
) -> Tolerance:
    """Pick the tolerance declared for a tensor leaf's dtype."""
    if not leaf.is_floating_point():
        return Tolerance(atol=0.0, rtol=0.0)
    try:
        name = canonical_dtype_name(leaf.dtype)
    except ValueError:
        # Non-canonical floating dtype (e.g. float64): the spec cannot declare
        # a tolerance for it, so apply the harness fallback.
        return default
    return tolerances.get(name, default)


def _canonical_name_or_none(leaf: Any) -> str | None:
    """Canonical dtype name for a tensor leaf, or ``None`` when it has none."""
    try:
        return canonical_dtype_name(leaf.dtype)
    except ValueError:
        return None


def _metadata_equal(candidate: Any, expected: Any) -> bool:
    try:
        equal = candidate == expected
    except Exception:
        return False
    if isinstance(equal, bool):
        return equal
    try:
        return bool(equal)
    except Exception:
        return False


def compare_tensor_leaf(
    path: str,
    candidate: Any,
    expected: Any,
    tolerance: Tolerance,
    *,
    exact: bool = False,
    max_absolute_error: float | None = None,
    check_layout: bool = False,
) -> LeafRecord:
    """Compare two tensor leaves with statistics, NaN/Inf flags and a reason.

    Public so other verifiers (e.g. gradient comparison) can reuse the exact
    forward-comparison semantics.

    Args:
        exact: require bitwise equality. Set by a ``byte_equal`` parity policy;
            ``tolerance`` is then ignored entirely.
        max_absolute_error: hard ceiling applied on top of ``tolerance``. A
            relative tolerance scales with the reference magnitude, so on a
            tensor whose values reach 1e6 an rtol of 1e-2 silently permits an
            absolute error of 1e4. This cap is what makes that visible.
        check_layout: also require matching strides. A kernel that returns the
            right values in the wrong layout satisfies ``allclose`` but changes
            what every downstream consumer reads.
    """
    torch = _torch()
    is_float = candidate.is_floating_point()
    has_nan = bool(torch.isnan(candidate).any().item()) if is_float else False
    has_inf = bool(torch.isinf(candidate).any().item()) if is_float else False

    if candidate.shape != expected.shape:
        return LeafRecord(
            path=path,
            kind="tensor",
            match=False,
            reason=f"shape mismatch: {tuple(candidate.shape)} vs {tuple(expected.shape)}",
            max_abs_error=float("inf"),
            mean_abs_error=float("inf"),
            pct_within_tol=0.0,
            has_nan=has_nan,
            has_inf=has_inf,
        )
    if candidate.dtype != expected.dtype:
        return LeafRecord(
            path=path,
            kind="tensor",
            match=False,
            reason=f"dtype mismatch: {candidate.dtype} vs {expected.dtype}",
            max_abs_error=float("inf"),
            mean_abs_error=float("inf"),
            pct_within_tol=0.0,
            has_nan=has_nan,
            has_inf=has_inf,
        )

    if check_layout and candidate.stride() != expected.stride():
        return LeafRecord(
            path=path,
            kind="tensor",
            match=False,
            reason=(
                f"layout mismatch: stride {tuple(candidate.stride())} vs "
                f"{tuple(expected.stride())}"
            ),
            max_abs_error=float("inf"),
            mean_abs_error=float("inf"),
            pct_within_tol=0.0,
            has_nan=has_nan,
            has_inf=has_inf,
        )

    if not is_float:
        # Integer and boolean leaves carry exact values; a tolerance would be
        # meaningless.
        match = bool(torch.equal(candidate, expected))
        return LeafRecord(
            path=path,
            kind="tensor",
            match=match,
            reason="" if match else "non-floating tensors are not bitwise equal",
            max_abs_error=0.0 if match else float("inf"),
            mean_abs_error=0.0 if match else float("inf"),
            pct_within_tol=100.0 if match else 0.0,
        )

    out_f = candidate.float()
    exp_f = expected.float()
    if out_f.numel() == 0:
        return LeafRecord(
            path=path,
            kind="tensor",
            match=True,
            max_abs_error=0.0,
            mean_abs_error=0.0,
            pct_within_tol=100.0,
        )

    if exact:
        # A byte_equal workload admits no numerical difference at all. Compare
        # the raw tensors, not their float() upcasts: two distinct bfloat16
        # values can share a float32 image only if they were already equal, but
        # comparing the originals keeps the check honest for every dtype.
        equal = bool(torch.equal(candidate, expected))
        abs_diff = (out_f - exp_f).abs()
        finite_diff = abs_diff[torch.isfinite(abs_diff)]
        max_abs = float(finite_diff.max().item()) if finite_diff.numel() else 0.0
        mean_abs = float(finite_diff.mean().item()) if finite_diff.numel() else 0.0
        differing = int((out_f != exp_f).sum().item())
        reason = ""
        if not equal:
            reason = (
                f"outputs are not bitwise equal under an exact parity policy: "
                f"{differing} differing element(s), max_abs_error={max_abs:.6e}"
            )
            if has_nan:
                reason += "; output contains NaN"
            elif has_inf:
                reason += "; output contains infinity"
        return LeafRecord(
            path=path,
            kind="tensor",
            match=equal,
            reason=reason,
            max_abs_error=max_abs,
            mean_abs_error=mean_abs,
            pct_within_tol=100.0 if equal else 0.0,
            has_nan=has_nan,
            has_inf=has_inf,
        )

    # Non-finite handling. ``inf - inf`` is NaN, so a candidate that reproduces
    # the reference's infinities used to report ``max_abs_error=nan`` while
    # ``allclose`` returned True. Compare the non-finite *pattern* explicitly
    # and derive statistics only from the finite elements.
    expected_nan = torch.isnan(exp_f)
    expected_inf = torch.isinf(exp_f)
    candidate_nan = torch.isnan(out_f)
    candidate_inf = torch.isinf(out_f)
    nonfinite_mismatch = ""
    if not bool(torch.equal(candidate_nan, expected_nan)):
        nonfinite_mismatch = "NaN positions differ from the reference"
    elif not bool(torch.equal(candidate_inf, expected_inf)):
        nonfinite_mismatch = "infinity positions differ from the reference"
    elif bool((candidate_inf & (torch.sign(out_f) != torch.sign(exp_f))).any().item()):
        nonfinite_mismatch = "infinity signs differ from the reference"

    abs_diff = (out_f - exp_f).abs()
    finite = torch.isfinite(abs_diff)
    finite_diff = abs_diff[finite]
    max_abs = float(finite_diff.max().item()) if finite_diff.numel() else 0.0
    mean_abs = float(finite_diff.mean().item()) if finite_diff.numel() else 0.0

    allowed = tolerance.atol + tolerance.rtol * exp_f.abs()
    within_mask = (abs_diff <= allowed) | (candidate_nan & expected_nan) | (
        candidate_inf & expected_inf & (torch.sign(out_f) == torch.sign(exp_f))
    )
    within = float(within_mask.float().mean().item()) * 100.0
    match = bool(within_mask.all().item()) and not nonfinite_mismatch

    cap_exceeded = (
        max_absolute_error is not None and max_abs > max_absolute_error
    )
    if match and cap_exceeded:
        match = False

    reason = ""
    if nonfinite_mismatch:
        reason = nonfinite_mismatch
    elif cap_exceeded and within >= 100.0:
        # Everything sat inside the relative tolerance, yet the absolute error
        # is enormous. That is exactly the R4 VAE signature: rtol=1e-2 against
        # reference values near 3e5 licensed an error of 32768.0.
        reason = (
            f"max_abs_error={max_abs:.6e} exceeds the policy's absolute ceiling "
            f"{max_absolute_error:.6e} despite satisfying "
            f"tol(atol={tolerance.atol}, rtol={tolerance.rtol}); the relative "
            f"tolerance is being scaled by large reference values"
        )
    elif not match:
        reason = (
            f"max_abs_error={max_abs:.6e} exceeds "
            f"tol(atol={tolerance.atol}, rtol={tolerance.rtol})"
        )
        if cap_exceeded:
            reason += (
                f" and the absolute ceiling {max_absolute_error:.6e}"
            )
        if has_nan:
            reason += "; output contains NaN"
        elif has_inf:
            reason += "; output contains infinity"
    return LeafRecord(
        path=path,
        kind="tensor",
        match=match,
        reason=reason,
        max_abs_error=max_abs,
        mean_abs_error=mean_abs,
        pct_within_tol=within,
        has_nan=has_nan,
        has_inf=has_inf,
    )


def compare_output_trees(
    candidate: Any,
    expected: Any,
    tolerances: Mapping[str, Tolerance],
    *,
    output_spec: OutputSpec | None = None,
    default_tolerance: Tolerance = DEFAULT_TOLERANCE,
    relax: float = 1.0,
    policy: ParityPolicy | None = None,
) -> TreeComparison:
    """Compare two output trees leaf by leaf.

    Args:
        candidate: the kernel's output tree.
        expected: the reference output tree.
        tolerances: canonical dtype name -> :class:`Tolerance`, typically
            ``spec.tolerances``. Each floating tensor leaf is compared with
            the tolerance declared for its own dtype.
        output_spec: optional policy controlling which paths participate and
            whether non-tensor leaves must match exactly.
        default_tolerance: fallback for leaf dtypes the mapping does not
            cover (matches the historical harness default of 1e-2/1e-2).
        relax: multiplies both tolerances (used by the numerical-stability
            stage for adversarial inputs). An exact policy clamps this to 1.0:
            widening a zero tolerance is meaningless, and widening it for
            adversarial inputs is how R4 accepted a 32768.0 error.
        policy: the workload's parity contract. ``None`` preserves the
            historical tolerance-only behaviour for callers that have no
            workload in hand; passing a policy is what makes ``byte_equal``
            actually mean byte-equal at this layer.
    """
    included = output_spec.included_paths if output_spec is not None else None
    compare_meta = output_spec.compare_non_tensors if output_spec is not None else True

    cand_leaves = _filter_leaves(
        flatten_output_tree(candidate), included, side="candidate"
    )
    exp_leaves = _filter_leaves(
        flatten_output_tree(expected), included, side="reference"
    )

    mismatch = _structure_mismatch_reason(cand_leaves, exp_leaves)
    if mismatch is not None:
        return TreeComparison(
            match=False,
            reason=f"output structure mismatch: {mismatch}",
            leaves=(),
            structure_match=False,
        )

    effective_relax = policy.effective_relax(relax) if policy is not None else relax
    exact = policy.exact if policy is not None else False
    abs_cap = policy.max_absolute_error if policy is not None else None

    records: list[LeafRecord] = []
    exp_map = dict(exp_leaves)
    for path, cand_leaf in cand_leaves:
        exp_leaf = exp_map[path]
        if _is_tensor(cand_leaf):
            if policy is None:
                tol = _tolerance_for(exp_leaf, tolerances, default_tolerance)
            elif not exp_leaf.is_floating_point():
                tol = Tolerance(atol=0.0, rtol=0.0)
            else:
                tol = resolve_leaf_tolerance(
                    _canonical_name_or_none(exp_leaf),
                    tolerances,
                    policy,
                    default=default_tolerance,
                    path=path,
                )
            if effective_relax != 1.0:
                tol = Tolerance(
                    atol=tol.atol * effective_relax,
                    rtol=tol.rtol * effective_relax,
                )
            records.append(
                compare_tensor_leaf(
                    path,
                    cand_leaf,
                    exp_leaf,
                    tol,
                    exact=exact,
                    max_absolute_error=abs_cap,
                    check_layout=exact,
                )
            )
            continue
        if not compare_meta:
            records.append(
                LeafRecord(path=path, kind="metadata", match=True, reason="not compared")
            )
            continue
        equal = _metadata_equal(cand_leaf, exp_leaf)
        records.append(
            LeafRecord(
                path=path,
                kind="metadata",
                match=equal,
                reason="" if equal else f"metadata mismatch: {cand_leaf!r} != {exp_leaf!r}",
            )
        )

    failed = [record for record in records if not record.match]
    if failed:
        first = failed[0]
        reason = first.reason
        if first.path != "output":
            reason = f"{first.path}: {reason}"
        return TreeComparison(match=False, reason=reason, leaves=tuple(records))
    return TreeComparison(match=True, reason="", leaves=tuple(records))



def compare_deterministic(
    first: Any,
    other: Any,
    *,
    output_spec: OutputSpec | None = None,
) -> TreeComparison:
    """Bitwise comparison of two runs of the same kernel.

    Every tensor leaf must be bitwise identical (``torch.equal``) and every
    compared metadata leaf must be exactly equal; the tree structures must
    match. Statistics for differing tensor leaves are reported so failures
    stay diagnosable.
    """
    torch = _torch()
    included = output_spec.included_paths if output_spec is not None else None
    compare_meta = output_spec.compare_non_tensors if output_spec is not None else True
    first_leaves = _filter_leaves(flatten_output_tree(first), included, side="first run")
    other_leaves = _filter_leaves(flatten_output_tree(other), included, side="later run")

    mismatch = _structure_mismatch_reason(first_leaves, other_leaves)
    if mismatch is not None:
        return TreeComparison(
            match=False,
            reason=f"output structure mismatch between runs: {mismatch}",
            leaves=(),
            structure_match=False,
        )

    records: list[LeafRecord] = []
    for (path, a), (_, b) in zip(first_leaves, other_leaves):
        if _is_tensor(a):
            max_diff: float | None = None
            mean_diff: float | None = None
            if a.shape == b.shape and a.dtype == b.dtype and a.is_floating_point():
                diff = (a.float() - b.float()).abs()
                max_diff = diff.max().item() if diff.numel() else 0.0
                mean_diff = diff.mean().item() if diff.numel() else 0.0
            equal = bool(torch.equal(a, b))
            if equal:
                reason = ""
            elif max_diff is not None:
                reason = f"runs differ (max_diff={max_diff:.6e})"
            else:
                reason = "runs differ (shape or dtype changed)"
            records.append(
                LeafRecord(
                    path=path,
                    kind="tensor",
                    match=equal,
                    reason=reason,
                    max_abs_error=max_diff,
                    mean_abs_error=mean_diff,
                )
            )
            continue
        if not compare_meta:
            records.append(
                LeafRecord(path=path, kind="metadata", match=True, reason="not compared")
            )
            continue
        equal = _metadata_equal(a, b)
        records.append(
            LeafRecord(
                path=path,
                kind="metadata",
                match=equal,
                reason="" if equal else f"metadata changed between runs: {a!r} != {b!r}",
            )
        )

    failed = [record for record in records if not record.match]
    if failed:
        first_fail = failed[0]
        reason = first_fail.reason
        if first_fail.path != "output":
            reason = f"{first_fail.path}: {reason}"
        return TreeComparison(match=False, reason=reason, leaves=tuple(records))
    return TreeComparison(match=True, reason="", leaves=tuple(records))


def tree_has_nan_or_inf(tree: Any) -> bool:
    """True when any floating tensor leaf contains NaN or infinity."""
    torch = _torch()
    for _, leaf in flatten_output_tree(tree):
        if _is_tensor(leaf) and leaf.is_floating_point():
            if bool(torch.isnan(leaf).any().item()) or bool(torch.isinf(leaf).any().item()):
                return True
    return False
