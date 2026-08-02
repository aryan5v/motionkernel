"""Installed MotionKernel command-line interface."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _version() -> str:
    try:
        return importlib.metadata.version("motionkernel")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0+source"


def _emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _doctor(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="motionkernel doctor")
    parser.add_argument("--fastvideo-checkout", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)

    checks: dict[str, dict[str, Any]] = {}
    checks["python"] = {"ok": sys.version_info >= (3, 10), "version": sys.version.split()[0]}
    for program in ("git", "codex"):
        resolved = shutil.which(program)
        checks[program] = {"ok": resolved is not None, "path": resolved}

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        checks["torch"] = {"ok": True, "version": str(torch.__version__)}
        checks["cuda"] = {
            "ok": cuda_available or not args.require_cuda,
            "available": cuda_available,
            "version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if cuda_available else None,
        }
    except Exception as exc:  # noqa: BLE001 - doctor reports broken environments
        checks["torch"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        checks["cuda"] = {"ok": not args.require_cuda, "available": False}

    try:
        import triton

        checks["triton"] = {"ok": True, "version": str(triton.__version__)}
    except Exception as exc:  # noqa: BLE001 - doctor reports broken environments
        checks["triton"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if args.fastvideo_checkout is not None:
        checkout = args.fastvideo_checkout.resolve()
        package = checkout / "fastvideo"
        check: dict[str, Any] = {"ok": package.is_dir(), "path": str(checkout)}
        if check["ok"] and shutil.which("git"):
            completed = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                check["commit"] = completed.stdout.strip()
        checks["fastvideo"] = check

    passed = all(bool(check.get("ok")) for check in checks.values())
    _emit({"schema_version": 1, "status": "pass" if passed else "fail", "checks": checks})
    return 0 if passed else 1


def _artifact(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="motionkernel artifact")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "inspect"):
        command = commands.add_parser(name)
        command.add_argument("bundle", type=Path)
    args = parser.parse_args(argv)

    from autokernel.artifact import ArtifactError, describe_bundle, verify_bundle

    try:
        manifest = verify_bundle(args.bundle)
    except ArtifactError as exc:
        _emit({"verified": False, "bundle": str(args.bundle), "error": str(exc)})
        return 1
    if args.command == "inspect":
        _emit({"verified": True, "bundle": str(args.bundle), "manifest": manifest.as_dict()})
    else:
        _emit({"verified": True, "bundle": str(args.bundle), **describe_bundle(manifest)})
    return 0


def _workload(argv: list[str]) -> int:
    root = Path(__file__).resolve().parent / "workloads"
    paths = {
        "cosmos25_2b_704p": root / "cosmos25_2b_704p.yaml",
        "ltx_480p": root / "ltx_480p.yaml",
        "wan_t2v_1.3b_480p": root / "wan_t2v_1.3b_480p.yaml",
    }
    parser = argparse.ArgumentParser(prog="motionkernel workload")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    path_parser = commands.add_parser("path")
    path_parser.add_argument("name", choices=tuple(paths))
    args = parser.parse_args(argv)

    if args.command == "list":
        _emit({name: str(path) for name, path in paths.items()})
    else:
        print(paths[args.name])
    return 0


def _usage() -> str:
    return (
        "usage: motionkernel <command> [options]\n\n"
        "commands:\n"
        "  optimize   run a resumable FastVideo optimization campaign\n"
        "  doctor     validate the local optimization environment\n"
        "  artifact   verify or inspect an artifact bundle\n"
        "  workload   list or locate packaged reference workloads\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Dispatch the installed CLI without importing GPU libraries eagerly."""
    parts = list(sys.argv[1:] if argv is None else argv)
    if not parts or parts[0] in {"-h", "--help"}:
        print(_usage())
        return 0
    if parts[0] in {"-V", "--version"}:
        print(_version())
        return 0

    command, rest = parts[0], parts[1:]
    if command == "optimize":
        from autokernel.optimize.cli import main as optimize_main

        return optimize_main(rest)
    if command == "doctor":
        return _doctor(rest)
    if command == "artifact":
        return _artifact(rest)
    if command == "workload":
        return _workload(rest)
    print(f"motionkernel: unknown command {command!r}\n\n{_usage()}", file=sys.stderr)
    return 2
