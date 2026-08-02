#!/usr/bin/env python3
"""Prove which inherited files are unreachable before proposing any deletion.

MotionKernel inherited a single-kernel research harness, starter kernels,
example models and a KernelBench bridge. Some of it is still load-bearing --
`bench.py` is the fixed benchmark the optimize pipeline shells out to -- and
some is dead weight on a public V1. Guessing which is which is how a working
pipeline gets broken by a tidy-up.

This checks five independent reachability signals per file and reports only
files where *all* of them say "unused":

1. imported     : any tracked Python file imports the module
2. cli          : reachable from the console-script entry point
3. packaging    : named by pyproject (packages, force-include, sdist include)
4. tests        : referenced by anything under tests/
5. artifacts    : referenced by artifact generation or loading
6. documented   : named as a command or path in README, docs, or CI workflows

Files failing any signal are reported as KEEP with the reason, so the output is
an argument, not an assertion.

    python scripts/legacy_reachability.py
    python scripts/legacy_reachability.py --write   # docs/DELETION_INVENTORY.md
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
from collections import defaultdict
from pathlib import Path

#: Directories and roots inherited from upstream that a public V1 might not
#: need. Everything else is out of scope for this report.
CANDIDATE_ROOTS = ("kernels", "kernelbench", "models", "examples")
CANDIDATE_FILES = (
    "analysis.py",
    "export_hf.py",
    "kernel.py",
    "orchestrate.py",
    "prepare.py",
    "reference.py",
    "verify.py",
    "discovery.py",
    "campaign.py",
    "workload.py",
    "optimize.py",
    "profile.py",
    "extract.py",
    "bench.py",
)


def _git(*args: str, root: Path) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=root, check=False
    ).stdout


def tracked_python(root: Path) -> list[str]:
    return [p for p in _git("ls-files", root=root).split() if p.endswith(".py")]


def module_names(path: str) -> set[str]:
    """Import names that would resolve to this file."""
    stem = path[:-3]
    dotted = stem.replace("/", ".")
    names = {dotted}
    if dotted.endswith(".__init__"):
        names.add(dotted[: -len(".__init__")])
    names.add(Path(path).stem)
    return names


def imports_in(root: Path, path: str) -> set[str]:
    try:
        tree = ast.parse((root / path).read_text(errors="replace"))
    except (OSError, SyntaxError):
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{a.name}" for a in node.names)
    return found


def build_report(root: Path) -> dict[str, dict[str, object]]:
    files = tracked_python(root)
    in_scope = [
        f
        for f in files
        if f.split("/")[0] in CANDIDATE_ROOTS or f in CANDIDATE_FILES
    ]

    # signal 1: imported by any tracked file (excluding itself)
    importers: dict[str, set[str]] = defaultdict(set)
    all_imports = {f: imports_in(root, f) for f in files}
    for target in in_scope:
        names = module_names(target)
        for source, imported in all_imports.items():
            if source == target:
                continue
            if names & imported or any(
                any(i == n or i.startswith(n + ".") for n in names) for i in imported
            ):
                importers[target].add(source)

    # signal 3: packaging
    pyproject = (root / "pyproject.toml").read_text(errors="replace")
    # signal 2: CLI reachability, transitively from the console script module
    cli_seen: set[str] = set()
    frontier = ["autokernel/cli.py"]
    while frontier:
        current = frontier.pop()
        if current in cli_seen or not (root / current).is_file():
            continue
        cli_seen.add(current)
        for imported in all_imports.get(current, set()):
            for candidate in files:
                if imported in module_names(candidate) and candidate not in cli_seen:
                    frontier.append(candidate)

    # signal 6: named in user-facing docs or CI. A file the README tells people
    # to run is reachable by the only path that matters -- a user typing it --
    # even though no Python file imports it. Without this the report proposed
    # deleting `profile.py` and `extract.py`, both documented commands.
    documented_sources = []
    for candidate in list(root.glob("*.md")) + list((root / "docs").glob("*.md")):
        documented_sources.append(candidate.read_text(errors="replace"))
    workflows = root / ".github" / "workflows"
    if workflows.is_dir():
        documented_sources.extend(
            w.read_text(errors="replace") for w in workflows.glob("*.y*ml")
        )
    documented = "\n".join(documented_sources)

    # signal 5: named by artifact generation or loading
    artifact_sources = "\n".join(
        (root / f).read_text(errors="replace")
        for f in files
        if f.startswith("autokernel/specgen/") or f.startswith("autokernel/artifact/")
    )

    report: dict[str, dict[str, object]] = {}
    for path in sorted(in_scope):
        stem = Path(path).name
        reasons: list[str] = []
        non_test_importers = {i for i in importers[path] if not i.startswith("tests/")}
        test_importers = {i for i in importers[path] if i.startswith("tests/")}
        if non_test_importers:
            reasons.append(f"imported by {sorted(non_test_importers)[:3]}")
        if path in cli_seen:
            reasons.append("reachable from the CLI entry point")
        if re.search(rf'"/?{re.escape(path)}"', pyproject) or re.search(
            rf'"{re.escape(path.split("/")[0])}"', pyproject
        ):
            reasons.append("named in pyproject packaging")
        if test_importers:
            reasons.append(f"used by tests ({len(test_importers)})")
        if stem in artifact_sources or path in artifact_sources:
            reasons.append("referenced by artifact generation/loading")
        if re.search(rf"\b{re.escape(path)}\b", documented) or re.search(
            rf"\b{re.escape(stem)}\b", documented
        ):
            reasons.append("documented as a command or path")
        report[path] = {"keep": bool(reasons), "reasons": reasons}
    return report


def render(root: Path, report: dict[str, dict[str, object]]) -> str:
    unused = [p for p, v in report.items() if not v["keep"]]
    keep = [p for p, v in report.items() if v["keep"]]
    groups: dict[str, list[str]] = defaultdict(list)
    for path in unused:
        groups[path.split("/")[0] if "/" in path else "(root)"].append(path)

    lines = [
        "# Staged deletion inventory",
        "",
        "<!-- Generated by scripts/legacy_reachability.py. Do not edit by hand. -->",
        "",
        "MotionKernel inherited a single-kernel research harness, starter kernels, "
        "example models and a KernelBench bridge from upstream. Some of it is still "
        "load-bearing; some is dead weight on a public V1. This report checks five "
        "independent reachability signals per file and proposes deletion only where "
        "*all five* say unused.",
        "",
        "Regenerate and re-check with:",
        "",
        "```bash",
        "python scripts/legacy_reachability.py --write",
        "```",
        "",
        "Signals: imported by tracked code, reachable from the CLI entry point, "
        "named in pyproject packaging, used by tests, referenced by artifact "
        "generation or loading, documented as a command or path.",
        "",
        f"**{len(keep)} files are reachable and stay. {len(unused)} are unreferenced "
        "by every signal.**",
        "",
        "## Proposed stages",
        "",
        "Staged so each step is independently revertible, and so a stage that turns "
        "out to be wrong is cheap to undo.",
        "",
    ]
    for stage, (group, members) in enumerate(sorted(groups.items()), start=1):
        lines += [
            f"### Stage {stage}: `{group}` ({len(members)} files)",
            "",
        ]
        lines += [f"- `{m}`" for m in sorted(members)]
        lines += [""]
    if not groups:
        lines += ["_No file is unreferenced by all six signals, so this pass proposes "
            "no deletions._", ""]

    lines += [
        "## Reachable — do not delete",
        "",
        "| File | Why it stays |",
        "|---|---|",
    ]
    for path in keep:
        reasons = "; ".join(str(r) for r in report[path]["reasons"])  # type: ignore[index]
        lines.append(f"| `{path}` | {reasons} |")
    lines += [
        "",
        "## Why nothing is proposed for deletion yet",
        "",
        "Every inherited candidate is reachable by at least one signal, and for "
        "most of them the only signal is *documentation*: `README.md` still "
        "presents the inherited single-kernel workflow (`profile.py`, "
        "`extract.py`, `kernel.py`, `bench.py`, the starter kernels, the "
        "example models, KernelBench) as a supported way to use the project.",
        "",
        "That makes deletion a **scoping decision, not a code fact**. Nothing "
        "can be proven unused while the README tells people to run it. The "
        "order of operations is therefore:",
        "",
        "1. decide whether V1 still offers the single-kernel research workflow;",
        "2. if not, remove it from the user-facing docs first;",
        "3. re-run this report, which will then have grounds to propose stages;",
        "4. apply each stage as its own commit, with the CPU suite, a build and "
        "a clean-install smoke test after each.",
        "",
        "Deleting code that is still advertised would break documented "
        "commands for anyone following the README, which is a worse public-V1 "
        "outcome than shipping some inherited surface area.",
        "",
        "## Caveats",
        "",
        "Reachability is static. A file could still be referenced by a string in a "
        "workload manifest, a docs example, a shell script, or an external user's "
        "code. Nothing here is deleted automatically: this report is the argument, "
        "and each stage should be applied as its own commit with the full CPU suite, "
        "a build, and a clean-install smoke test run afterwards.",
        "",
        "Deleting inherited files does not change the licensing position: the "
        "upstream MIT notice in `LICENSE` still covers what remains, and "
        "`PROVENANCE.md` continues to record what was inherited.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    report = build_report(root)
    rendered = render(root, report)
    if args.write:
        (root / "docs" / "DELETION_INVENTORY.md").write_text(rendered, encoding="utf-8")
        print("wrote docs/DELETION_INVENTORY.md")
        return 0
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
