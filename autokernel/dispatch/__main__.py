"""Command-line entry point for the dispatch-overhead harness.

Usage:
    python -m autokernel.dispatch measure --fastvideo-checkout ... \
        --workload ... --artifact-root ... --output ...
    python -m autokernel.dispatch analyze timing.json
    python -m autokernel.dispatch breakeven --native-e2e-s 3.36 \
        --calls-per-generation 384 --overhead-ms 0.05
    python -m autokernel.dispatch summarize measurement.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .measure import MIN_TIMED_RUNS, MeasurementError, run_dispatch_measurement
from .overhead import (
    DEFAULT_CALL_VOLUMES,
    DEFAULT_GATE,
    DispatchAnalysisError,
    attribute_overhead,
    breakeven_curve,
    load_timing_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m autokernel.dispatch",
        description="Measure, attribute, and publish artifact dispatch overhead",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    measure = sub.add_parser(
        "measure",
        help="Run the A/B + shadow-profile measurement against a FastVideo checkout",
    )
    measure.add_argument("--fastvideo-checkout", type=Path, required=True)
    measure.add_argument("--workload", type=Path, required=True)
    measure.add_argument("--artifact-root", type=Path, required=True)
    measure.add_argument("--output", type=Path, required=True)
    measure.add_argument("--model", default=None, help="Model id override")
    measure.add_argument(
        "--runs",
        type=int,
        default=MIN_TIMED_RUNS,
        help=f"Timed runs per arm (minimum {MIN_TIMED_RUNS})",
    )
    measure.add_argument("--warmups", type=int, default=1)
    measure.add_argument(
        "--kernel-saving-ms",
        type=float,
        default=None,
        help="The artifact's isolated per-call kernel saving; required for the "
        "overhead decomposition and break-even curve",
    )
    measure.add_argument("--gate", type=float, default=DEFAULT_GATE)
    measure.add_argument("--python", default=None, help="Python for launcher subprocesses")
    measure.add_argument("--timeout", type=float, default=None)

    analyze = sub.add_parser(
        "analyze", help="Attribute overhead from an existing shadow timing report"
    )
    analyze.add_argument("timing", type=Path)

    breakeven = sub.add_parser(
        "breakeven", help="Print the break-even curve for one overhead figure"
    )
    breakeven.add_argument("--native-e2e-s", type=float, required=True)
    breakeven.add_argument("--overhead-ms", type=float, required=True)
    breakeven.add_argument("--gate", type=float, default=DEFAULT_GATE)
    breakeven.add_argument(
        "--calls-per-generation",
        type=int,
        nargs="+",
        default=list(DEFAULT_CALL_VOLUMES),
    )

    summarize = sub.add_parser(
        "summarize", help="Print the publishable summary of a measurement record"
    )
    summarize.add_argument("measurement", type=Path)
    return parser


def _cmd_measure(args: argparse.Namespace) -> int:
    record = run_dispatch_measurement(
        fastvideo_checkout=args.fastvideo_checkout,
        workload=args.workload,
        artifact_root=args.artifact_root,
        output_dir=args.output,
        model_override=args.model,
        runs=args.runs,
        warmups=args.warmups,
        kernel_saving_ms_per_call=args.kernel_saving_ms,
        gate=args.gate,
        python=args.python,
        timeout=args.timeout,
    )
    print(json.dumps(record, indent=2))
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    report = load_timing_report(args.timing)
    attribution = attribute_overhead(report)
    payload = attribution.as_dict()
    print(f"{'phase':40s} {'value':>12s}")
    print(f"{'native forward mean (ms)':40s} {payload['native_forward_mean_ms']:12.4f}")
    print(f"{'candidate total mean (ms)':40s} {payload['candidate_total_mean_ms']:12.4f}")
    print(f"{'net overhead (ms/call)':40s} {payload['net_overhead_ms_per_call']:12.4f}")
    print(f"{'replay path':40s} {payload['replay_path']:>12s}")
    print(f"{'replay mean (ms)':40s} {payload['replay_mean_ms']:12.4f}")
    print(f"{'plumbing (ms/candidate call)':40s} {payload['plumbing_ms_per_candidate_call']:12.4f}")
    print(f"{'shape key (ms/candidate call)':40s} {payload['shape_key_ms_per_candidate_call']:12.4f}")
    print(f"{'candidate calls':40s} {payload['candidate_calls']:12d}")
    print(f"{'graph replay calls':40s} {payload['graph_replay_calls']:12d}")
    print(f"{'eager execute calls':40s} {payload['eager_execute_calls']:12d}")
    print(f"{'warmup calls':40s} {payload['warmup_calls']:12d}")
    for reason, count in sorted(payload["declined_captures"].items()):
        print(f"  declined: {reason} x{count}")
    return 0


def _cmd_breakeven(args: argparse.Namespace) -> int:
    curve = breakeven_curve(
        native_e2e_seconds=args.native_e2e_s,
        overhead_ms_per_call=args.overhead_ms,
        gate=args.gate,
        call_volumes=args.calls_per_generation,
    )
    print(
        f"native e2e {args.native_e2e_s:.4f}s, overhead {args.overhead_ms:+.4f} ms/call, "
        f"gate {args.gate:.2f}x"
    )
    print(f"{'calls/generation':>18s} {'required saving/call':>22s}")
    for point in curve:
        print(f"{point.calls_per_generation:18d} {point.required_saving_us_per_call:19.1f} us")
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    record = json.loads(args.measurement.read_text(encoding="utf-8"))
    print(f"workload: {record['workload_id']}  model: {record['model_id']}")
    print(f"arch: {record['arch']}  status: {record['status']}  date: {record['created_utc']}")
    variance = record.get("variance")
    if variance:
        print(
            f"variance: native_cv={variance['native_cv']} candidate_cv={variance['candidate_cv']} "
            f"ceiling={variance['cv_ceiling']} valid_for_gating={variance['valid_for_gating']}"
        )
        print(f"  {variance['reason']}")
    controls = record.get("controls")
    if controls:
        clocks = controls.get("gpu_clocks") or {}
        print(
            f"controls: node_exclusive={controls.get('node_exclusive')} "
            f"clocks={clocks.get('state')} (sm {clocks.get('sm_clock_mhz')}/{clocks.get('sm_clock_max_mhz')} MHz) "
            f"slurm_job={controls.get('slurm_job_id')}"
        )
    native, candidate = record["native"], record["candidate"]
    print(
        f"native    median {native['median']:.4f}s stdev {native['stdev']:.4f} "
        f"min {native['min']:.4f} ({native['count']} runs)"
    )
    print(
        f"candidate median {candidate['median']:.4f}s stdev {candidate['stdev']:.4f} "
        f"min {candidate['min']:.4f} ({candidate['count']} runs)"
    )
    print(
        f"e2e speedup: {record['e2e']['speedup_median']:.4f}x median, "
        f"{record['e2e']['speedup_min_to_min']:.4f}x min-to-min"
    )
    print(f"parity: {record['parity']['passed']} ({record['parity']['policy']})")
    print(
        f"candidate calls: {record['candidate_calls']} "
        f"({record['calls_per_generation']}/generation), "
        f"runtime fallbacks: {record['runtime_fallbacks']}"
    )
    attribution = record["shadow_attribution"]
    print(
        f"shadow: native {attribution['native_forward_mean_ms']:.4f} ms vs candidate "
        f"{attribution['candidate_total_mean_ms']:.4f} ms -> net "
        f"{attribution['net_overhead_ms_per_call']:+.4f} ms/call "
        f"({attribution['replay_path']}, sync-serialized)"
    )
    host = record.get("host_profile")
    if host:
        print(
            f"host: candidate path {host['candidate_total_host_ms_per_call']:.4f} ms/call "
            f"host-side (plumbing {host['plumbing_host_ms_per_call']:.4f}, "
            f"shape key {host['shape_key_host_ms_per_call']:.4f}, "
            f"graph replay launch {host['graph_replay_host_ms_per_call']:.4f})"
        )
    if "e2e_overhead" in record:
        overhead = record["e2e_overhead"]
        print(
            f"e2e overhead: net {overhead['net_cost_ms_per_call']:+.4f} ms/call, "
            f"dispatch overhead {overhead['overhead_ms_per_call']:+.4f} ms/call"
        )
    if "breakeven" in record:
        print("break-even (required kernel saving per call):")
        for point in record["breakeven"]:
            print(
                f"  {point['calls_per_generation']:>6d} calls/gen: "
                f"{point['required_saving_us_per_call']:>9.1f} us"
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "measure":
            return _cmd_measure(args)
        if args.command == "analyze":
            return _cmd_analyze(args)
        if args.command == "breakeven":
            return _cmd_breakeven(args)
        if args.command == "summarize":
            return _cmd_summarize(args)
    except (MeasurementError, DispatchAnalysisError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
