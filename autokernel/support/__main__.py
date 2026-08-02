"""Support-matrix CLI.

    python -m autokernel.support matrix [--check]
    python -m autokernel.support nightly --fastvideo-checkout ... --arch sm100 ...
    python -m autokernel.support record --receipt ... --workload-id ... ...
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .evidence import (
    RECORD_SCHEMA,
    RECORD_SCHEMA_VERSION,
    RunRecord,
    record_from_receipt,
    write_run_record,
)
from .matrix import DEFAULT_ARCHES, DEFAULT_STALE_DAYS, generate_matrix
from .nightly import NightlyConfig, NightlyError, NightlyTarget, run_nightly


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m autokernel.support",
        description="Generate and check the evidence-derived support matrix",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    matrix = sub.add_parser("matrix", help="Generate the matrix page and sidecar")
    matrix.add_argument("--workloads", type=Path, default=Path("workloads"))
    matrix.add_argument("--evidence", type=Path, default=Path("docs/support-evidence"))
    matrix.add_argument("--markdown", type=Path, default=Path("docs/SUPPORT_MATRIX.md"))
    matrix.add_argument("--sidecar", type=Path, default=Path("docs/support_matrix.json"))
    matrix.add_argument("--arches", nargs="+", default=list(DEFAULT_ARCHES))
    matrix.add_argument("--stale-days", type=int, default=DEFAULT_STALE_DAYS)
    matrix.add_argument(
        "--check",
        action="store_true",
        help="Regenerate deterministically and fail if the committed page differs",
    )

    nightly = sub.add_parser("nightly", help="Run discovery-only campaigns for tonight")
    nightly.add_argument("--fastvideo-checkout", type=Path, required=True)
    nightly.add_argument("--nightly-root", type=Path, required=True)
    nightly.add_argument("--records-dir", type=Path, required=True)
    nightly.add_argument("--arch", required=True)
    nightly.add_argument("--workload", type=Path, action="append", required=True)
    nightly.add_argument("--model", default=None, help="Model id override (single-target runs)")
    nightly.add_argument("--budget-hours", type=float, default=2.0)
    nightly.add_argument("--date", default=None, help="Night directory name (default: today UTC)")
    nightly.add_argument("--repo-root", type=Path, default=None)

    record = sub.add_parser("record", help="Distill an existing campaign receipt into a run record")
    record.add_argument("--receipt", type=Path, required=True)
    record.add_argument("--workload-id", required=True)
    record.add_argument("--model-id", required=True)
    record.add_argument("--family", required=True)
    record.add_argument("--arch", required=True)
    record.add_argument("--evidence", required=True)
    record.add_argument("--recorded-utc", default=None)
    record.add_argument("--output", type=Path, required=True)
    return parser


def _cmd_matrix(args: argparse.Namespace) -> int:
    today = None
    if args.check:
        # Deterministic check: regenerate for the committed sidecar's date so
        # CI verifies content, not the passage of time.
        if args.sidecar.is_file():
            try:
                committed = json.loads(args.sidecar.read_text(encoding="utf-8"))
                today = datetime.fromisoformat(committed["generated_on"]).date()
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                print(
                    f"error: cannot read committed sidecar date from {args.sidecar}",
                    file=sys.stderr,
                )
                return 2
    markdown, sidecar = generate_matrix(
        workloads_dir=args.workloads,
        evidence_dir=args.evidence,
        arches=args.arches,
        stale_days=args.stale_days,
        today=today,
    )
    sidecar_text = json.dumps(sidecar, indent=2) + "\n"
    if args.check:
        drift = False
        for path, generated in ((args.markdown, markdown), (args.sidecar, sidecar_text)):
            committed_text = path.read_text(encoding="utf-8") if path.is_file() else ""
            if committed_text != generated:
                drift = True
                diff = difflib.unified_diff(
                    committed_text.splitlines(),
                    generated.splitlines(),
                    fromfile=str(path),
                    tofile=f"{path} (regenerated)",
                    lineterm="",
                )
                print("\n".join(diff), file=sys.stderr)
        if drift:
            print(
                "error: committed support matrix differs from the generated one; "
                "run `python -m autokernel.support matrix` and commit the result",
                file=sys.stderr,
            )
            return 1
        print("support matrix is up to date")
        return 0
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown, encoding="utf-8")
    args.sidecar.write_text(sidecar_text, encoding="utf-8")
    print(f"wrote {args.markdown} and {args.sidecar}")
    return 0


def _cmd_nightly(args: argparse.Namespace) -> int:
    if args.model is not None and len(args.workload) != 1:
        print("error: --model override requires exactly one --workload", file=sys.stderr)
        return 2
    config = NightlyConfig(
        fastvideo_checkout=args.fastvideo_checkout,
        targets=tuple(
            NightlyTarget(workload=path, model_override=args.model)
            for path in args.workload
        ),
        nightly_root=args.nightly_root,
        records_dir=args.records_dir,
        arch=args.arch,
        budget_hours=args.budget_hours,
        date=args.date,
        repo_root=args.repo_root,
    )
    try:
        report = run_nightly(config)
    except NightlyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    try:
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: cannot read receipt {args.receipt}: {exc}", file=sys.stderr)
        return 2
    record = record_from_receipt(
        receipt,
        workload_id=args.workload_id,
        model_id=args.model_id,
        family=args.family,
        arch=args.arch,
        evidence=args.evidence,
        recorded_utc=args.recorded_utc or datetime.now(timezone.utc).isoformat(),
    )
    if record is None:
        print("receipt terminal is budget_exhausted; no verdict, no record", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_run_record(record, args.output)
    print(f"wrote {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "matrix":
        return _cmd_matrix(args)
    if args.command == "nightly":
        return _cmd_nightly(args)
    if args.command == "record":
        return _cmd_record(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
