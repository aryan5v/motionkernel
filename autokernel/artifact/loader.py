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
import re
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
    if manifest is not None and manifest != verified:
        raise ArtifactError(
            f"artifact bundle {str(directory)!r}: manifest changed since validation"
        )

    entry_file = _resolve_inside(directory, directory / verified.entry_point.file)
    entry_digest = next(
        (item for item in verified.files if item.path == verified.entry_point.file),
        None,
    )
    if entry_digest is None:  # Defensive: manifest parsing already enforces this.
        raise ArtifactError(
            f"artifact bundle {str(directory)!r}: entry_point: file is not declared"
        )
    try:
        source = entry_file.read_bytes()
    except OSError as exc:
        raise ArtifactError(
            f"artifact bundle {str(directory)!r}: entry_point: cannot read "
            f"{verified.entry_point.file!r}"
        ) from exc
    # Execute this immutable snapshot only after binding it to the digest that
    # verify_bundle accepted. A path replacement after this check is harmless:
    # no loader reopens the mutable path.
    actual_hash = hashlib.sha256(source).hexdigest()
    if len(source) != entry_digest.bytes or actual_hash != entry_digest.sha256:
        raise ArtifactError(
            f"artifact bundle {str(directory)!r}: entry_point: "
            f"{verified.entry_point.file!r} changed after verification"
        )

    readable_id = re.sub(r"[^A-Za-z0-9_]", "_", verified.artifact_id)
    unique_id = hashlib.sha256(verified.artifact_id.encode("utf-8")).hexdigest()[:16]
    module_name = f"{_MODULE_NAMESPACE}.{readable_id}_{unique_id}"
    module = types.ModuleType(module_name)
    module.__file__ = str(entry_file)
    module.__package__ = module_name.rpartition(".")[0]
    # Registering before execution lets the module use dataclasses and other
    # machinery that looks itself up in sys.modules. It is removed again if
    # execution fails so a partial module is never reachable.
    sys.modules[module_name] = module
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        code = compile(source, str(entry_file), "exec", dont_inherit=True)
        exec(code, module.__dict__)  # noqa: S102 - verified artifact is executable
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise ArtifactError(
            f"artifact bundle {str(directory)!r}: entry_point: importing "
            f"{verified.entry_point.file!r} raised "
            f"{type(exc).__name__}"
        ) from exc
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode

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
