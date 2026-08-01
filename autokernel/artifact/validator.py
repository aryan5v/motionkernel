"""Structural and integrity validation of a packaged artifact bundle.

Validation is fail-closed and runs *before* anything in the bundle is
imported: a bundle whose contents do not match the digests recorded at package
time is rejected outright rather than partially trusted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .types import (
    MANIFEST_FILENAME,
    ArtifactError,
    ArtifactManifest,
    iter_manifest_files,
)

#: Files allowed to exist in a bundle without being hashed in the manifest.
#: Everything else must be declared, so an attacker cannot slip an extra
#: importable module alongside a legitimately signed entry point.
_UNDECLARED_ALLOWLIST = frozenset({MANIFEST_FILENAME})

#: No executable directory is ignored. Bytecode caches are executable content
#: and must never be invisible to undeclared-file validation.
_IGNORED_DIRECTORIES: frozenset[str] = frozenset()

_READ_CHUNK_BYTES = 1 << 20


def file_sha256(path: str | Path) -> str:
    """Content hash of one file, streamed so large bundles stay cheap."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_READ_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(bundle_dir: str | Path) -> ArtifactManifest:
    """Parse and validate ``artifact.json`` without touching the payload."""
    directory = Path(bundle_dir)
    manifest_path = directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise ArtifactError(
            f"artifact bundle {str(directory)!r}: {MANIFEST_FILENAME}: not found"
        )
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(
            f"artifact bundle {str(directory)!r}: {MANIFEST_FILENAME}: "
            f"invalid JSON: {exc}"
        ) from exc
    return ArtifactManifest.from_dict(raw, source=str(directory))


def _bundle_files(directory: Path) -> list[Path]:
    result = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            result.append(path)
            continue
        if path.is_dir():
            continue
        if any(part in _IGNORED_DIRECTORIES for part in path.relative_to(directory).parts):
            continue
        result.append(path)
    return result


def verify_bundle(bundle_dir: str | Path) -> ArtifactManifest:
    """Validate a bundle end to end and return its manifest.

    Raises :class:`ArtifactError` if the manifest is malformed, a declared file
    is missing or altered, or the directory holds undeclared content.
    """
    directory = Path(bundle_dir)
    if not directory.is_dir():
        raise ArtifactError(f"artifact bundle {str(directory)!r}: not a directory")
    manifest = read_manifest(directory)
    source = str(directory)

    declared = set(manifest.file_paths())
    present = {
        path.relative_to(directory).as_posix() for path in _bundle_files(directory)
    }
    undeclared = sorted(present - declared - _UNDECLARED_ALLOWLIST)
    if undeclared:
        raise ArtifactError(
            f"artifact bundle {source!r}: files: undeclared file(s) {undeclared}"
        )

    for entry in iter_manifest_files(manifest):
        path = directory / entry.path
        if not path.is_file():
            raise ArtifactError(
                f"artifact bundle {source!r}: files: {entry.path!r} is missing"
            )
        size = path.stat().st_size
        if size != entry.bytes:
            raise ArtifactError(
                f"artifact bundle {source!r}: files: {entry.path!r} is "
                f"{size} bytes, manifest records {entry.bytes}"
            )
        actual = file_sha256(path)
        if actual != entry.sha256:
            raise ArtifactError(
                f"artifact bundle {source!r}: files: {entry.path!r} hash "
                f"{actual} does not match manifest {entry.sha256}"
            )
    return manifest


def describe_bundle(manifest: ArtifactManifest) -> dict[str, Any]:
    """A compact, log-safe summary of a validated bundle."""
    return {
        "artifact_id": manifest.artifact_id,
        "operation": manifest.operation.name,
        "graph_fingerprint": manifest.graph_fingerprint,
        "entry_point": f"{manifest.entry_point.file}:{manifest.entry_point.symbol}",
        "files": len(manifest.files),
        "promotion": manifest.promotion.decision,
        "speedup": manifest.evidence.benchmark.speedup,
    }
