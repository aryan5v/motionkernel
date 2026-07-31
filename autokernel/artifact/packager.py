"""Package an optimized graph kernel into a portable artifact bundle.

Packaging happens once, on the machine that produced and proved the kernel.
The result is a self-describing directory that any runtime can validate and
load without knowing which model it came from.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from autokernel import __version__ as _AUTOKERNEL_VERSION
from autokernel.verification.results import write_result_atomic

from .types import (
    ARTIFACT_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    ArtifactError,
    ArtifactManifest,
)
from .validator import file_sha256, verify_bundle

DEFAULT_PRODUCER = {"name": "motionkernel", "version": _AUTOKERNEL_VERSION}

_IGNORED_DIRECTORIES = frozenset({"__pycache__", ".git"})

#: Sections the caller must supply. ``files``, ``schema_version`` and
#: ``created_at`` are computed here and may not be passed in.
_REQUIRED_SECTIONS = (
    "artifact_id",
    "operation",
    "signature",
    "entry_point",
    "compatibility",
    "evidence",
    "promotion",
)
_COMPUTED_SECTIONS = ("files", "schema_version")


def _payload_files(source: Path) -> list[Path]:
    result = []
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(source)
        if any(part in _IGNORED_DIRECTORIES for part in relative.parts):
            continue
        if relative.as_posix() == MANIFEST_FILENAME:
            # A manifest in the payload would be overwritten by this one; the
            # caller almost certainly pointed at an already-packaged bundle.
            raise ArtifactError(
                f"artifact bundle {str(source)!r}: files: payload already "
                f"contains {MANIFEST_FILENAME}"
            )
        result.append(path)
    return result


def build_manifest(
    source_dir: str | Path,
    sections: Mapping[str, Any],
    *,
    producer: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a manifest document for the payload in ``source_dir``.

    Every regular file under ``source_dir`` is hashed and declared, so a
    bundle can never reference content that was not measured at package time.
    """
    source = Path(source_dir)
    if not source.is_dir():
        raise ArtifactError(f"artifact bundle {str(source)!r}: not a directory")

    missing = [name for name in _REQUIRED_SECTIONS if name not in sections]
    if missing:
        raise ArtifactError(
            f"artifact bundle {str(source)!r}: top level: missing section(s) "
            f"{sorted(missing)}"
        )
    supplied = [name for name in _COMPUTED_SECTIONS if name in sections]
    if supplied:
        raise ArtifactError(
            f"artifact bundle {str(source)!r}: top level: section(s) "
            f"{sorted(supplied)} are computed by the packager"
        )

    payload = _payload_files(source)
    if not payload:
        raise ArtifactError(f"artifact bundle {str(source)!r}: files: payload is empty")

    document: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_at": created_at
        or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "producer": dict(producer or DEFAULT_PRODUCER),
        "files": [
            {
                "path": path.relative_to(source).as_posix(),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in payload
        ],
    }
    document.update(sections)
    # Parse before writing: a manifest that cannot be validated is never
    # allowed to reach disk.
    return ArtifactManifest.from_dict(document, source=str(source)).as_dict()


def package_artifact(
    source_dir: str | Path,
    output_dir: str | Path,
    sections: Mapping[str, Any],
    *,
    producer: Mapping[str, Any] | None = None,
    created_at: str | None = None,
    overwrite: bool = False,
) -> ArtifactManifest:
    """Copy a payload into ``output_dir`` and write its validated manifest.

    Returns the manifest of the finished bundle, which has been re-verified
    from disk exactly the way a consuming runtime will verify it.
    """
    source = Path(source_dir)
    output = Path(output_dir)
    document = build_manifest(
        source,
        sections,
        producer=producer,
        created_at=created_at,
    )

    if output.exists():
        if not overwrite:
            raise ArtifactError(
                f"artifact bundle {str(output)!r}: already exists; "
                "pass overwrite=True to replace it"
            )
        if not output.is_dir():
            raise ArtifactError(f"artifact bundle {str(output)!r}: not a directory")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for entry in document["files"]:
        destination = output / entry["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / entry["path"], destination)
    write_result_atomic(output / MANIFEST_FILENAME, document)

    return verify_bundle(output)


def discover_bundles(root: str | Path) -> list[Path]:
    """List candidate bundle directories under ``root``.

    A bundle is any directory holding an ``artifact.json``. The search is one
    level deep plus the root itself, which keeps a large artifact store from
    turning discovery into a full filesystem walk.
    """
    directory = Path(root)
    if not directory.is_dir():
        return []
    found: list[Path] = []
    if (directory / MANIFEST_FILENAME).is_file():
        found.append(directory)
    for child in sorted(directory.iterdir()):
        if child.is_dir() and (child / MANIFEST_FILENAME).is_file():
            found.append(child)
    return found


def load_bundles(root: str | Path) -> tuple[list[ArtifactManifest], list[str]]:
    """Validate every bundle under ``root``.

    Returns the manifests that verified and a structured error string per
    bundle that did not, so a single bad artifact cannot hide the good ones.
    """
    manifests: list[ArtifactManifest] = []
    errors: list[str] = []
    for path in discover_bundles(root):
        try:
            manifests.append(verify_bundle(path))
        except ArtifactError as exc:
            errors.append(str(exc))
    return manifests, errors
