"""Shared helpers for the CPU test suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autokernel.specs import DT_BYTES, KernelSpec, Tolerance, size

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

#: A minimal manifest that passes the real workload schema. Written as JSON so
#: the CPU suite does not require PyYAML.
WORKLOAD_FIXTURE: dict[str, Any] = {
    "schema_version": 1,
    "workload_id": "cpu-contract",
    "task": "t2v",
    "prompt": "a preflight contract test workload",
    "model": {"model_id": "test/model"},
    "sampling": {
        "height": 64,
        "width": 64,
        "num_frames": 9,
        "num_inference_steps": 2,
        "guidance_scale": 1.0,
        "seed": 0,
    },
}


def make_fastvideo_checkout(root: Path, *, complete: bool = True) -> Path:
    """Create a directory with the structure preflight expects of FastVideo."""
    checkout = root / "FastVideo"
    package = checkout / "fastvideo"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    if complete:
        launcher_dir = checkout / "examples" / "inference" / "optimizations"
        launcher_dir.mkdir(parents=True, exist_ok=True)
        (launcher_dir / "generation_launcher.py").write_text(
            "", encoding="utf-8"
        )
    return checkout


def make_workload(path: Path, **overrides: Any) -> Path:
    """Write a schema-valid workload manifest, overriding any top-level field."""
    payload: dict[str, Any] = json.loads(json.dumps(WORKLOAD_FIXTURE))
    payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _ref(x: Any = None, y: Any = None) -> Any:
    return x


def _gen(size_map: Any, dtype: Any, device: str, seed: int = 42) -> dict:
    return {"x": None}


def spec_kwargs(**overrides: Any) -> dict:
    """Keyword arguments for a minimal valid :class:`KernelSpec`."""
    base: dict[str, Any] = {
        "name": "unit_op",
        "reference_fn": _ref,
        "input_generator": _gen,
        "sizes": {
            "small": {"rows": 4, "cols": 4},
            "medium": {"rows": 8, "cols": 8},
            "large": {"rows": 16, "cols": 16},
        },
        "dtypes": ("float16", "float32"),
        "tolerances": {
            "float16": Tolerance(atol=1e-2, rtol=1e-2),
            "float32": Tolerance(atol=1e-5, rtol=1e-5),
        },
        "flops_fn": size("rows") * size("cols"),
        "bytes_fn": 2 * size("rows") * size("cols") * DT_BYTES,
        "shape_keys": ("rows", "cols"),
    }
    base.update(overrides)
    return base


def make_spec(**overrides: Any) -> KernelSpec:
    """Build a minimal valid specification, overriding any field."""
    return KernelSpec(**spec_kwargs(**overrides))


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def in_repo_root(monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run a test with the repository root as the working directory."""
    monkeypatch.chdir(REPO_ROOT)
    return REPO_ROOT


@pytest.fixture
def torch_mod():
    """The torch module, skipping the test when torch is unavailable."""
    return pytest.importorskip("torch")


def cuda_available() -> bool:
    """True when a CUDA device is usable (never raises when torch is absent)."""
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


requires_gpu = pytest.mark.skipif(
    not cuda_available(), reason="requires a CUDA GPU"
)

__all__ = [
    "FIXTURES_DIR",
    "REPO_ROOT",
    "WORKLOAD_FIXTURE",
    "cuda_available",
    "make_fastvideo_checkout",
    "make_spec",
    "make_workload",
    "requires_gpu",
    "spec_kwargs",
]
