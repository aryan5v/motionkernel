"""CLI: backfill the store, and check that new records still ingest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .ingest import ingest_path
from .schema import connect, ingest_row

DEFAULT_STORE = Path("docs/experiments.sqlite")

#: Documents that summarize other records rather than reporting a measurement.
_AGGREGATE_SCHEMAS = frozenset({"motionkernel.support-matrix"})


def _backfill(args) -> int:
    connection = connect(args.store)
    new = seen = unknown = 0
    unknown_paths: list[Path] = []
    for root in args.roots:
        for path, row in ingest_path(Path(root)):
            seen += 1
            if row is None:
                unknown += 1
                unknown_paths.append(path)
                continue
            if ingest_row(connection, row):
                new += 1
    connection.commit()
    total = connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    print(f"scanned {seen} documents, ingested {new} new, store now holds {total}")
    print(f"unrecognized: {unknown}")
    if args.show_unknown:
        for path in unknown_paths[: args.show_unknown]:
            print(f"  ? {path}")
    by_kind = connection.execute(
        "SELECT kind, COUNT(*) FROM experiments GROUP BY kind ORDER BY 2 DESC"
    ).fetchall()
    for kind, count in by_kind:
        print(f"  {count:5d}  {kind}")
    return 0


def _claims_to_be_a_result(path: Path) -> bool:
    """Whether a document asserts one of our result schemas.

    Keyed on the declared `schema` field rather than the filename, so a record
    that moves or is renamed is still gated, and an intermediate that happens
    to be called measurement.json is not.
    """
    import json

    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(doc, dict):
        return False
    schema = str(doc.get("schema", ""))
    if not schema.startswith("motionkernel."):
        return False
    # Generated aggregates are derived FROM results, not results. Ingesting
    # the support matrix would store a summary of rows the store already
    # holds individually, and any disagreement between the two would be a
    # second source of truth rather than a check on the first.
    return schema not in _AGGREGATE_SCHEMAS


def _check(args) -> int:
    """CI gate: every record under --roots must ingest into a fresh store.

    Fails when a document that a reader *should* recognize does not, which is
    how a source schema change is caught before it silently stops being
    indexed.
    """
    connection = connect(":memory:")
    failures: list[Path] = []
    for root in args.roots:
        for path, row in ingest_path(Path(root)):
            if row is None:
                # Only a document that *claims* to be a result is a failure.
                # A run directory is full of intermediates -- native_result,
                # dispatch diagnostics, timing traces -- which are inputs to a
                # measurement, not measurements. Failing on those would make
                # the gate noise, and a noisy gate gets disabled.
                if _claims_to_be_a_result(path):
                    failures.append(path)
                continue
            ingest_row(connection, row)
    if failures:
        print(f"{len(failures)} record(s) did not ingest:", file=sys.stderr)
        for path in failures[:20]:
            print(f"  {path}", file=sys.stderr)
        return 1
    total = connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    print(f"all {total} record(s) ingest cleanly")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m autokernel.store")
    sub = parser.add_subparsers(dest="command", required=True)

    backfill = sub.add_parser("backfill", help="ingest evidence into the store")
    backfill.add_argument("roots", nargs="+")
    backfill.add_argument("--store", default=str(DEFAULT_STORE))
    backfill.add_argument("--show-unknown", type=int, default=0)
    backfill.set_defaults(func=_backfill)

    check = sub.add_parser("check", help="fail if any record does not ingest")
    check.add_argument("roots", nargs="+")
    check.set_defaults(func=_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
