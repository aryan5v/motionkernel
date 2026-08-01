"""Small fail-clean helpers for atomic text and JSON artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_text_atomic(path: str | Path, text: str) -> Path:
    """Replace ``path`` atomically and remove any failed temporary write."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


def write_json_atomic(path: str | Path, payload: Any) -> Path:
    """Serialize one indented JSON document through ``write_text_atomic``."""
    return write_text_atomic(path, json.dumps(payload, indent=2) + "\n")
