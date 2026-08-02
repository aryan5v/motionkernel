"""Per-call dispatch-overhead attribution for the artifact dispatch path.

Two measurements produced the 3.104 ms figure this module supersedes
(docs/LTX_V1_R4_ROOT_CAUSE.md section 7):

1. End-to-end arithmetic: ``(candidate - native) / calls_per_generation``,
   plus the artifact's isolated per-call kernel saving.
2. In-situ shadow profiling: with
   ``FASTVIDEO_OPTIMIZATION_ARTIFACT_TIMING=shadow`` the FastVideo dispatch
   path records ``dispatch.candidate_total`` against
   ``shadow.native_forward`` on identical inputs, plus the ``subgraph.*``
   plumbing phases.

This module validates the FastVideo timing report, computes the attribution,
and derives the break-even curve: the per-call kernel saving a region must
deliver to clear the promotion gate at a given call volume. It is pure
analysis -- no torch, no GPU -- so the published number's arithmetic is
CPU-testable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

TIMING_REPORT_SCHEMA_VERSION = 1

#: Phase names recorded by fastvideo/optimization/{dispatch,subgraph,timing}.py.
PHASE_CANDIDATE_TOTAL = "dispatch.candidate_total"
PHASE_NATIVE_SHADOW = "shadow.native_forward"
PHASE_NATIVE_REFERENCE = "dispatch.native_reference"
PHASE_NATIVE_FALLBACK = "dispatch.native_fallback"
PHASE_SHAPE_KEY = "dispatch.shape_key"
PHASE_GRAPH_REPLAY = "subgraph.execute_cuda_graph"
PHASE_EAGER_EXECUTE = "subgraph.execute"
PHASE_FLATTEN = "subgraph.flatten"
PHASE_VALIDATE = "subgraph.validate"
PHASE_UNFLATTEN = "subgraph.unflatten"

#: Plumbing phases amortized over candidate calls in the R4 attribution.
PLUMBING_PHASES = (PHASE_FLATTEN, PHASE_VALIDATE, PHASE_UNFLATTEN)

NOTE_WARMUP = "cuda_graph_warmup"
NOTE_DECLINED_PREFIX = "cuda_graph_declined"

#: Call volumes (dispatched calls per generation) spanning the regions seen so
#: far: video VAE blocks at ~5 through transformer block stacks at 384 and
#: beyond. 384 is the LTX transformer stack (48 blocks x 8 steps).
DEFAULT_CALL_VOLUMES: tuple[int, ...] = (
    1, 5, 10, 25, 50, 100, 200, 384, 500, 1000, 2000, 5000,
)

DEFAULT_GATE = 1.01


class DispatchAnalysisError(ValueError):
    """A timing report or overhead input is malformed or inconsistent."""


def _fail(source: object, location: str, message: str) -> DispatchAnalysisError:
    return DispatchAnalysisError(f"dispatch timing {source!r}: {location}: {message}")


@dataclass(frozen=True)
class TimingPhase:
    """One named phase aggregate from a FastVideo timing report."""

    total_seconds: float
    calls: int

    @property
    def mean_ms(self) -> float:
        if self.calls == 0:
            return 0.0
        return self.total_seconds / self.calls * 1000.0

    @classmethod
    def from_dict(cls, raw_value: Any, *, source: object, location: str) -> "TimingPhase":
        if not isinstance(raw_value, Mapping):
            raise _fail(source, location, "must be an object")
        total = raw_value.get("total_seconds")
        calls = raw_value.get("calls")
        if isinstance(total, bool) or not isinstance(total, (int, float)) or total < 0:
            raise _fail(source, f"{location}.total_seconds", "must be a non-negative number")
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
            raise _fail(source, f"{location}.calls", "must be a non-negative integer")
        if calls == 0 and total != 0:
            raise _fail(source, location, "zero calls with non-zero total")
        return cls(total_seconds=float(total), calls=calls)


@dataclass(frozen=True)
class TimingReport:
    """Validated FastVideo ``timing.json`` payload."""

    synchronized: bool
    phases: Mapping[str, TimingPhase]
    notes: Mapping[str, int]
    source: str = "<memory>"

    @classmethod
    def from_dict(cls, raw_value: Any, *, source: object = "<memory>") -> "TimingReport":
        if not isinstance(raw_value, Mapping):
            raise _fail(source, "top level", "must be an object")
        version = raw_value.get("timing_schema_version")
        if version != TIMING_REPORT_SCHEMA_VERSION:
            raise _fail(
                source,
                "timing_schema_version",
                f"unsupported version {version!r}; expected {TIMING_REPORT_SCHEMA_VERSION}",
            )
        synchronized = raw_value.get("synchronized")
        if not isinstance(synchronized, bool):
            raise _fail(source, "synchronized", "must be a bool")
        raw_phases = raw_value.get("phases")
        if not isinstance(raw_phases, Mapping):
            raise _fail(source, "phases", "must be an object")
        phases: dict[str, TimingPhase] = {}
        for name, payload in raw_phases.items():
            if not isinstance(name, str) or not name:
                raise _fail(source, "phases", "phase names must be non-empty strings")
            phases[name] = TimingPhase.from_dict(payload, source=source, location=f"phases.{name}")
        raw_notes = raw_value.get("notes", {})
        if not isinstance(raw_notes, Mapping):
            raise _fail(source, "notes", "must be an object")
        notes: dict[str, int] = {}
        for note, count in raw_notes.items():
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise _fail(source, f"notes.{note}", "must be a non-negative integer")
            notes[str(note)] = count
        return cls(
            synchronized=synchronized,
            phases=phases,
            notes=notes,
            source=str(source),
        )

    def phase(self, name: str) -> TimingPhase | None:
        return self.phases.get(name)

    def note_count(self, name: str) -> int:
        return self.notes.get(name, 0)

    def declined_captures(self) -> Mapping[str, int]:
        """Reason -> count for permanently declined CUDA-graph captures."""
        return {
            note[len(NOTE_DECLINED_PREFIX):].strip(": "): count
            for note, count in self.notes.items()
            if note.startswith(NOTE_DECLINED_PREFIX)
        }


def load_timing_report(path: str | Path) -> TimingReport:
    """Load and validate a FastVideo timing report from disk."""
    file_path = Path(path)
    if not file_path.is_file():
        raise DispatchAnalysisError(f"dispatch timing {file_path!s}: file: not found")
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DispatchAnalysisError(f"dispatch timing {file_path!s}: invalid JSON: {exc}") from exc
    return TimingReport.from_dict(raw, source=str(file_path))


@dataclass(frozen=True)
class OverheadAttribution:
    """Per-call overhead of the dispatched candidate path, measured in situ.

    ``net_overhead_ms_per_call`` is the honest headline: the full candidate
    path (graph replay or eager replay, plumbing, hooks, parameter
    materialization) minus the native forward it replaced, on identical
    inputs. Negative means the dispatched path is *faster* than the native
    module forward -- possible because a CUDA graph replays the whole
    rewritten block without its host-side dispatch cost.
    """

    candidate_calls: int
    native_forward_mean_ms: float
    candidate_total_mean_ms: float
    net_overhead_ms_per_call: float
    replay_path: str
    replay_mean_ms: float
    graph_replay_calls: int
    eager_execute_calls: int
    warmup_calls: int
    declined_captures: Mapping[str, int]
    plumbing_ms_per_candidate_call: float
    shape_key_ms_per_candidate_call: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_calls": self.candidate_calls,
            "native_forward_mean_ms": round(self.native_forward_mean_ms, 4),
            "candidate_total_mean_ms": round(self.candidate_total_mean_ms, 4),
            "net_overhead_ms_per_call": round(self.net_overhead_ms_per_call, 4),
            "replay_path": self.replay_path,
            "replay_mean_ms": round(self.replay_mean_ms, 4),
            "graph_replay_calls": self.graph_replay_calls,
            "eager_execute_calls": self.eager_execute_calls,
            "warmup_calls": self.warmup_calls,
            "declined_captures": dict(self.declined_captures),
            "plumbing_ms_per_candidate_call": round(self.plumbing_ms_per_candidate_call, 4),
            "shape_key_ms_per_candidate_call": round(self.shape_key_ms_per_candidate_call, 4),
        }


def attribute_overhead(report: TimingReport) -> OverheadAttribution:
    """Attribute per-call dispatch overhead from a shadow-timing report.

    The report must come from a ``FASTVIDEO_OPTIMIZATION_ARTIFACT_TIMING=shadow``
    run: every candidate call then runs one shadow native forward on identical
    inputs, which is what makes the comparison honest. A report whose shadow
    and candidate call counts disagree mixed modes and is rejected.
    """
    if not report.synchronized:
        raise DispatchAnalysisError(
            f"dispatch timing {report.source!r}: report is not synchronized; "
            "re-run with FASTVIDEO_OPTIMIZATION_ARTIFACT_TIMING=shadow"
        )
    candidate = report.phase(PHASE_CANDIDATE_TOTAL)
    native = report.phase(PHASE_NATIVE_SHADOW)
    if candidate is None or candidate.calls == 0:
        raise DispatchAnalysisError(
            f"dispatch timing {report.source!r}: {PHASE_CANDIDATE_TOTAL} has no "
            "calls; the artifact path never engaged"
        )
    if native is None or native.calls == 0:
        raise DispatchAnalysisError(
            f"dispatch timing {report.source!r}: {PHASE_NATIVE_SHADOW} missing; "
            "re-run with FASTVIDEO_OPTIMIZATION_ARTIFACT_TIMING=shadow"
        )
    if native.calls != candidate.calls:
        raise DispatchAnalysisError(
            f"dispatch timing {report.source!r}: {PHASE_NATIVE_SHADOW} has "
            f"{native.calls} calls against {candidate.calls} candidate calls; "
            "the run mixed shadow and non-shadow modes"
        )
    calls = candidate.calls

    graph = report.phase(PHASE_GRAPH_REPLAY)
    eager = report.phase(PHASE_EAGER_EXECUTE)
    graph_calls = graph.calls if graph is not None else 0
    eager_calls = eager.calls if eager is not None else 0
    if graph_calls and eager_calls:
        replay_path = "mixed"
        replay_mean_ms = (
            (graph.total_seconds + eager.total_seconds) / (graph_calls + eager_calls) * 1000.0
        )
    elif graph_calls:
        replay_path = "cuda_graph"
        replay_mean_ms = graph.mean_ms  # type: ignore[union-attr]
    elif eager_calls:
        replay_path = "eager"
        replay_mean_ms = eager.mean_ms  # type: ignore[union-attr]
    else:
        replay_path = "none"
        replay_mean_ms = 0.0

    plumbing_total_s = sum(
        report.phase(name).total_seconds
        for name in PLUMBING_PHASES
        if report.phase(name) is not None
    )
    shape_key = report.phase(PHASE_SHAPE_KEY)

    return OverheadAttribution(
        candidate_calls=calls,
        native_forward_mean_ms=native.mean_ms,
        candidate_total_mean_ms=candidate.mean_ms,
        net_overhead_ms_per_call=candidate.mean_ms - native.mean_ms,
        replay_path=replay_path,
        replay_mean_ms=replay_mean_ms,
        graph_replay_calls=graph_calls,
        eager_execute_calls=eager_calls,
        warmup_calls=report.note_count(NOTE_WARMUP),
        declined_captures=report.declined_captures(),
        plumbing_ms_per_candidate_call=plumbing_total_s * 1000.0 / calls,
        shape_key_ms_per_candidate_call=(
            shape_key.total_seconds * 1000.0 / calls if shape_key is not None else 0.0
        ),
    )


@dataclass(frozen=True)
class E2EOverhead:
    """Dispatch overhead derived from an end-to-end A/B (the 3.104 ms method).

    ``net_cost_ms_per_call`` is what the dispatched path actually cost (or,
    when negative, saved) per call end-to-end. ``overhead_ms_per_call`` adds
    back the artifact's isolated kernel saving, isolating the framework's
    charge for delivering that saving.
    """

    native_median_seconds: float
    candidate_median_seconds: float
    calls_per_generation: int
    kernel_saving_ms_per_call: float
    net_cost_ms_per_call: float
    overhead_ms_per_call: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "native_median_seconds": round(self.native_median_seconds, 4),
            "candidate_median_seconds": round(self.candidate_median_seconds, 4),
            "calls_per_generation": self.calls_per_generation,
            "kernel_saving_ms_per_call": round(self.kernel_saving_ms_per_call, 4),
            "net_cost_ms_per_call": round(self.net_cost_ms_per_call, 4),
            "overhead_ms_per_call": round(self.overhead_ms_per_call, 4),
        }


def overhead_from_e2e(
    *,
    native_median_seconds: float,
    candidate_median_seconds: float,
    calls_per_generation: int,
    kernel_saving_ms_per_call: float,
) -> E2EOverhead:
    """Reproduce the R4 section 7 arithmetic for the current dispatch path.

    R4: native 3.6789s, candidate 4.8226s, 384 calls/generation, kernel saving
    0.124 ms/call -> net cost 2.980 ms/call, overhead 3.104 ms/call.
    """
    if isinstance(calls_per_generation, int) is False or calls_per_generation <= 0:
        raise DispatchAnalysisError("calls_per_generation must be a positive integer")
    for name, value in (
        ("native_median_seconds", native_median_seconds),
        ("candidate_median_seconds", candidate_median_seconds),
        ("kernel_saving_ms_per_call", kernel_saving_ms_per_call),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise DispatchAnalysisError(f"{name} must be a non-negative number")
    net_cost = (candidate_median_seconds - native_median_seconds) * 1000.0 / calls_per_generation
    return E2EOverhead(
        native_median_seconds=float(native_median_seconds),
        candidate_median_seconds=float(candidate_median_seconds),
        calls_per_generation=calls_per_generation,
        kernel_saving_ms_per_call=float(kernel_saving_ms_per_call),
        net_cost_ms_per_call=net_cost,
        overhead_ms_per_call=net_cost + kernel_saving_ms_per_call,
    )


@dataclass(frozen=True)
class BreakEvenPoint:
    """Minimum per-call kernel saving that clears the gate at one call volume."""

    calls_per_generation: int
    required_saving_ms_per_call: float
    overhead_ms_per_call: float
    gate: float

    @property
    def required_saving_us_per_call(self) -> float:
        return self.required_saving_ms_per_call * 1000.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls_per_generation": self.calls_per_generation,
            "required_saving_ms_per_call": round(self.required_saving_ms_per_call, 4),
            "required_saving_us_per_call": round(self.required_saving_us_per_call, 2),
            "overhead_ms_per_call": round(self.overhead_ms_per_call, 4),
            "gate": self.gate,
        }


def required_saving_ms_per_call(
    *,
    native_e2e_seconds: float,
    calls_per_generation: int,
    overhead_ms_per_call: float,
    gate: float = DEFAULT_GATE,
) -> float:
    """Minimum kernel saving per call for a region to clear the promotion gate.

    A region dispatched ``calls_per_generation`` times per generation, on a
    workload whose native end-to-end is ``native_e2e_seconds``, clears the
    gate when::

        native - calls * (saving - overhead) / 1000 <= native / gate

    i.e. ``saving >= overhead + native * (1 - 1/gate) * 1000 / calls``.
    """
    if isinstance(calls_per_generation, bool) or not isinstance(calls_per_generation, int) or calls_per_generation <= 0:
        raise DispatchAnalysisError("calls_per_generation must be a positive integer")
    if isinstance(native_e2e_seconds, bool) or not isinstance(native_e2e_seconds, (int, float)) or native_e2e_seconds <= 0:
        raise DispatchAnalysisError("native_e2e_seconds must be a positive number")
    if isinstance(gate, bool) or not isinstance(gate, (int, float)) or gate < 1.0:
        raise DispatchAnalysisError("gate must be >= 1.0")
    if isinstance(overhead_ms_per_call, bool) or not isinstance(overhead_ms_per_call, (int, float)):
        raise DispatchAnalysisError("overhead_ms_per_call must be a number")
    return overhead_ms_per_call + float(native_e2e_seconds) * (1.0 - 1.0 / float(gate)) * 1000.0 / calls_per_generation


def breakeven_curve(
    *,
    native_e2e_seconds: float,
    overhead_ms_per_call: float,
    gate: float = DEFAULT_GATE,
    call_volumes: Sequence[int] = DEFAULT_CALL_VOLUMES,
) -> tuple[BreakEvenPoint, ...]:
    """The break-even curve other tracks use to decide if a region is worth searching."""
    points = []
    for volume in call_volumes:
        points.append(
            BreakEvenPoint(
                calls_per_generation=volume,
                required_saving_ms_per_call=required_saving_ms_per_call(
                    native_e2e_seconds=native_e2e_seconds,
                    calls_per_generation=volume,
                    overhead_ms_per_call=overhead_ms_per_call,
                    gate=gate,
                ),
                overhead_ms_per_call=float(overhead_ms_per_call),
                gate=float(gate),
            )
        )
    return tuple(points)
