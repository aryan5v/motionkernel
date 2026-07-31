"""Trusted loading of a packaged artifact's entry point.

The rules this module enforces, in order:

1. Executable code is imported only from inside an explicitly configured
   trusted root. A bundle that resolves outside that root -- through a symlink
   or a crafted path -- is refused.
2. Every declared file is hashed and compared against the manifest *before*
   any import happens.
3. The bundle is never placed on ``sys.path``, so it cannot shadow a stdlib or
   site-packages module for the rest of the process.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, Callable

from .types import ArtifactError, ArtifactManifest
from .validator import verify_bundle

#: Prefix for the synthetic module names bundles are imported under. Keeping
#: them in a private namespace avoids collisions with real packages.
_MODULE_NAMESPACE = "autokernel._artifacts"


def _resolve_inside(root: Path, candidate: Path) -> Path:
    """Resolve ``candidate`` and require it to stay under ``root``."""
    resolved_root = root.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ArtifactError(
            f"artifact bundle {str(candidate)!r}: resolves outside the trusted "
            f"root {str(resolved_root)!r}"
        )
    return resolved


def load_entry_point(
    bundle_dir: str | Path,
    *,
    trusted_root: str | Path,
    manifest: ArtifactManifest | None = None,
) -> Callable[..., Any]:
    """Verify a bundle and import its candidate callable.

    ``manifest`` may be passed to reuse an earlier verification; the bundle is
    re-verified regardless, because the window between validation and import is
    exactly where a tampered file would land.
    """
    root = Path(trusted_root)
    if not root.is_dir():
        raise ArtifactError(
            f"artifact bundle {str(bundle_dir)!r}: trusted root "
            f"{str(root)!r} is not a directory"
        )
    directory = _resolve_inside(root, Path(bundle_dir))

    verified = verify_bundle(directory)
    if manifest is not None and manifest.artifact_id != verified.artifact_id:
        raise ArtifactError(
            f"artifact bundle {str(directory)!r}: artifact_id changed from "
            f"{manifest.artifact_id!r} to {verified.artifact_id!r} since "
            "validation"
        )

    entry_file = _resolve_inside(directory, directory / verified.entry_point.file)
    readable_id = re.sub(r"[^A-Za-z0-9_]", "_", verified.artifact_id)
    unique_id = hashlib.sha256(verified.artifact_id.encode("utf-8")).hexdigest()[:16]
    module_name = f"{_MODULE_NAMESPACE}.{readable_id}_{unique_id}"
    spec = importlib.util.spec_from_file_location(module_name, entry_file)
    if spec is None or spec.loader is None:
        raise ArtifactError(
            f"artifact bundle {str(directory)!r}: entry_point: cannot load "
            f"{verified.entry_point.file!r}"
        )
    module = importlib.util.module_from_spec(spec)
    # Registering before execution lets the module use dataclasses and other
    # machinery that looks itself up in sys.modules. It is removed again if
    # execution fails so a partial module is never reachable.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise ArtifactError(
            f"artifact bundle {str(directory)!r}: entry_point: importing "
            f"{verified.entry_point.file!r} raised "
            f"{type(exc).__name__}"
        ) from exc

    candidate = getattr(module, verified.entry_point.symbol, None)
    if candidate is None:
        sys.modules.pop(module_name, None)
        raise ArtifactError(
            f"artifact bundle {str(directory)!r}: entry_point: "
            f"{verified.entry_point.symbol!r} is not defined in "
            f"{verified.entry_point.file!r}"
        )
    if not callable(candidate):
        sys.modules.pop(module_name, None)
        raise ArtifactError(
            f"artifact bundle {str(directory)!r}: entry_point: "
            f"{verified.entry_point.symbol!r} is not callable"
        )
    return candidate
