"""Modal entrypoints for the Track D sm90 (H100) dispatch-overhead measurement.

Everything to date is sm100 (GB200, SLURM). This app produces the second
architecture honestly:

1. ``seed_weights`` -- one-time download of the LTX2 checkpoint into the
   shared ``fastvideo-hf-cache`` volume (public repo, no token).
2. ``bench_sm90`` -- the artifact candidate's isolated benchmark on H100,
   byte_equal policy, compile baseline: real sm90 evidence, not a re-declared
   window.
3. ``package_sm90`` -- build the sm90 artifact bundle with the real packager,
   decision ``quarantined`` (no sm90 end-to-end evidence yet).
4. ``measure_sm90`` -- the same 15-run/arm A/B plus shadow and host profiles
   the sm100 number comes from, then the measurement record.

Run:  modal run modal/dispatch_overhead_h100.py::seed_weights   # once
      modal run modal/dispatch_overhead_h100.py::run_all
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "motionkernel-dispatch-overhead-sm90"

app = modal.App(APP_NAME)

MOTIONKERNEL_COMMIT = "c8d10b392ba72675529948b97bba33bb613f2b37"
FASTVIDEO_COMMIT = "7299cc9a969879df06ed74bfed4a43b39d05678f"
MODEL_ID = "FastVideo/LTX2-Distilled-Diffusers"

#: Fallback only. The measured sm90 saving is derived from the sm90 isolated
#: benchmark at measure time -- the sm100 value (0.124 ms) does not transfer
#: across architectures, and using it would corrupt the overhead decomposition.
DEFAULT_KERNEL_SAVING_MS_PER_CALL = 0.0018

RUN_ROOT = Path("/runs/dispatch-overhead-sm90")
ARTIFACT_VOLUME_ROOT = Path("/artifacts")

base_image = modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.06-py3")

ltx_image = (
    base_image.apt_install(
        "ffmpeg",
        "git",
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install(
        "accelerate==1.0.1",
        "av",
        "cloudpickle",
        "diffusers>=0.38.0",
        "einops",
        "ftfy>=6.3.1",
        "h5py>=3.12.1",
        "huggingface-hub",
        "imageio>=2.36.0",
        "imageio-ffmpeg>=0.5.1",
        "loguru",
        "matplotlib>=3.10.0",
        "numpy>=1.26.0,<2",
        "omegaconf",
        "opencv-python-headless>=4.10.0.84",
        "pandas>=2.2.0",
        "peft>=0.15.0",
        "pillow>=10.3.0",
        "protobuf>=5.28.3",
        "pyyaml>=6.0.1",
        "safetensors>=0.5.0",
        "scipy>=1.14.1",
        "sentencepiece>=0.2.0",
        "timm>=1.0.11",
        "tokenizers>=0.20.1,<0.23",
        "transformers>=5.0.0",
        "tqdm",
    )
    .run_commands(
        "git clone https://github.com/aryan5v/motionkernel.git /opt/motionkernel",
        f"git -C /opt/motionkernel checkout --detach {MOTIONKERNEL_COMMIT}",
        "git clone https://github.com/aryan5v/FastVideo.git /opt/FastVideo",
        f"git -C /opt/FastVideo checkout --detach {FASTVIDEO_COMMIT}",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HUGGINGFACE_HUB_CACHE": "/cache/huggingface/hub",
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": "/opt/FastVideo:/opt/motionkernel",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TRITON_CACHE_DIR": "/runs/dispatch-overhead-sm90/triton-cache",
        }
    )
)

hf_cache = modal.Volume.from_name("fastvideo-hf-cache")
artifact_volume = modal.Volume.from_name("motionkernel-dispatch-sm90")
runs = modal.Volume.from_name("motionkernel-dispatch-sm90-runs", create_if_missing=True)

VOLUMES = {"/cache": hf_cache, "/artifacts": artifact_volume, "/runs": runs}


def _run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, check=False, text=True, **kwargs)


@app.function(
    image=base_image.pip_install("huggingface-hub", "hf_transfer"),
    timeout=4 * 60 * 60,
    volumes={"/cache": hf_cache},
    env={
        "HF_HOME": "/cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/cache/huggingface/hub",
        "HF_HUB_OFFLINE": "0",
    },
)
def seed_weights() -> str:
    """One-time download of the LTX2 checkpoint (public repo)."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL_ID)
    hf_cache.commit()
    return json.dumps({"model_id": MODEL_ID, "snapshot": path})


@app.function(
    image=ltx_image,
    gpu="H100!",
    timeout=3 * 60 * 60,
    scaledown_window=60,
    volumes=VOLUMES,
)
def bench_sm90() -> str:
    """Isolated benchmark of the LTX transformer candidate on H100."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Modal allocated the function without working CUDA")
    capability = torch.cuda.get_device_capability(0)
    if capability != (9, 0):
        raise RuntimeError(f"expected sm90, got {capability}")

    candidate = RUN_ROOT / "candidate-sm90"
    candidate.mkdir(parents=True, exist_ok=True)
    for name in ("kernel.py", "manifest.json", "spec.py", "corpus.json"):
        (candidate / name).write_bytes((ARTIFACT_VOLUME_ROOT / "ltx-candidate" / name).read_bytes())

    result_json = RUN_ROOT / "benchmark-sm90.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = "/opt/motionkernel"
    completed = _run(
        [
            sys.executable,
            "/opt/motionkernel/bench.py",
            "--spec",
            str(candidate / "spec.py") + ":SPEC",
            "--shape-corpus",
            str(candidate / "corpus.json"),
            "--baseline",
            "compile",
            "--parity-policy",
            "byte_equal",
            "--result-json",
            str(result_json),
        ],
        env=env,
        capture_output=True,
        cwd=str(candidate),
    )
    payload = {}
    if result_json.is_file():
        payload = json.loads(result_json.read_text(encoding="utf-8"))
    receipt = {
        "returncode": completed.returncode,
        "stdout_tail": (completed.stdout or "")[-4000:],
        "stderr_tail": (completed.stderr or "")[-4000:],
        "benchmark": str(result_json),
        "forward_correctness": (payload.get("forward") or {}).get("correctness"),
        "primary": (payload.get("performance") or {}).get("primary"),
    }
    (RUN_ROOT / "bench_receipt-sm90.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    runs.commit()
    if completed.returncode != 0 or receipt["forward_correctness"] != "PASS":
        raise RuntimeError(f"sm90 isolated benchmark failed: {json.dumps(receipt)[:2000]}")
    return json.dumps(receipt, indent=2)


@app.function(
    image=ltx_image,
    gpu="H100!",
    timeout=60 * 60,
    scaledown_window=60,
    volumes=VOLUMES,
)
def package_sm90() -> str:
    """Build the sm90 artifact bundle with the real packager, quarantined."""
    sys.path.insert(0, "/opt/motionkernel")
    from autokernel.artifact import package_artifact
    from autokernel.specgen import (
        build_dispatch_contract,
        spec_from_manifest,
        write_runtime_adapter,
    )
    from autokernel.workload import load_workload

    import hashlib
    import shutil
    from datetime import datetime, timezone

    candidate = RUN_ROOT / "candidate-sm90"
    bench_path = RUN_ROOT / "benchmark-sm90.json"
    payload = json.loads(bench_path.read_text(encoding="utf-8"))
    primary = payload["performance"]["primary"]
    forward = payload["forward"]
    if forward["correctness"] != "PASS":
        raise RuntimeError("refusing to package a failing kernel")
    policy = payload["request"]["parity_policy"]
    if not (policy["policy"] == "byte_equal" and policy["exact"]):
        raise RuntimeError(f"unexpected policy {policy}")

    manifest_value = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
    contract = build_dispatch_contract(manifest_value)
    spec = spec_from_manifest(candidate / "manifest.json")
    workload = load_workload("/opt/motionkernel/workloads/ltx_480p.yaml")

    capability = payload["gpu"]["compute_capability"]
    if isinstance(capability, str):
        parts = [int(part) for part in capability.split(".")]
    else:
        parts = [int(part) for part in capability]
    architecture = "sm" + "".join(str(part) for part in parts)
    if architecture != "sm90":
        raise RuntimeError(f"benchmark gpu is {architecture}, expected sm90")

    fingerprint = manifest_value.get("parent", {}).get("fingerprint") or "2c92e356aa34bc0d3c49522bd1365c1b"
    kernel_digest = hashlib.sha256((candidate / "kernel.py").read_bytes()).hexdigest()
    artifact_id = f"mk-{fingerprint[:16]}-{kernel_digest[:8]}-{architecture}"

    staging = RUN_ROOT / "staging" / artifact_id
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copyfile(candidate / "kernel.py", staging / "candidate.py")
    shutil.copyfile(candidate / "manifest.json", staging / "manifest.json")
    write_runtime_adapter(manifest_value, staging / "entry.py", candidate_file="candidate.py")

    tolerance = spec.tolerance_for(spec.dtypes[0])
    sections = {
        "artifact_id": artifact_id,
        **contract,
        "entry_point": {"file": "entry.py", "symbol": "fused_subgraph"},
        "compatibility": {
            "model_id": MODEL_ID,
            "model_revision": "*",
            "gpu_architectures": [architecture],
            "torch": {},
            "cuda": {},
            "triton": {},
            "execution_modes": ["inference"],
            "distributed_modes": ["single"],
        },
        "evidence": {
            "benchmark": {
                "harness": "motionkernel-bench",
                "device": "cuda:0",
                "samples": 100,
                "baseline_us": float(primary["pytorch_latency_us"]),
                "candidate_us": float(primary["kernel_latency_us"]),
                "speedup": float(primary["speedup_vs_pytorch"]),
                "max_abs_error": 0.0,
                "max_rel_error": 0.0,
                "atol": tolerance.atol,
                "rtol": tolerance.rtol,
                "passed": True,
                "result_ref": str(bench_path),
            },
            "generation": {
                "workload_id": workload.workload_id,
                "steps": workload.sampling.num_inference_steps,
                "metric": "pending_full_generation_validation",
                "value": 0.0,
                "threshold": 0.0,
                "passed": False,
                "baseline_ref": "",
                "candidate_ref": "",
            },
        },
        "promotion": {
            "decision": "quarantined",
            "reason": "sm90 repackage of the sm100-promoted candidate; no sm90 end-to-end evidence yet",
            "decided_at": datetime.now(timezone.utc).isoformat(),
            "campaign": {
                "campaign_id": "dispatch-overhead-sm90",
                "source": "motionkernel-optimize",
                "target_name": str(manifest_value.get("name") or fingerprint),
            },
        },
    }
    artifact_root = RUN_ROOT / "artifacts-sm90"
    manifest = package_artifact(staging, artifact_root / artifact_id, sections, overwrite=True)
    runs.commit()
    return json.dumps(
        {"artifact_id": artifact_id, "bundle": str(artifact_root / artifact_id),
         "verified": manifest.artifact_id},
        indent=2,
    )


@app.function(
    image=ltx_image,
    gpu="H100!",
    timeout=6 * 60 * 60,
    scaledown_window=60,
    volumes=VOLUMES,
)
def measure_sm90() -> str:
    """The same 15-run/arm A/B + shadow/host profiles as the sm100 number."""
    sys.path.insert(0, "/opt/motionkernel")
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID, local_files_only=True)

    artifact_root = RUN_ROOT / "artifacts-sm90"
    bundles = [path for path in artifact_root.iterdir() if path.is_dir()]
    if len(bundles) != 1:
        raise RuntimeError(f"expected exactly one sm90 bundle in {artifact_root}")

    bench_path = RUN_ROOT / "benchmark-sm90.json"
    kernel_saving_ms = DEFAULT_KERNEL_SAVING_MS_PER_CALL
    if bench_path.is_file():
        primary = json.loads(bench_path.read_text(encoding="utf-8"))["performance"]["primary"]
        kernel_saving_ms = (
            float(primary["pytorch_latency_us"]) - float(primary["kernel_latency_us"])
        ) / 1000.0

    output = RUN_ROOT / "measurement-sm90"
    completed = _run(
        [
            sys.executable,
            "-m",
            "autokernel.dispatch",
            "measure",
            "--fastvideo-checkout",
            "/opt/FastVideo",
            "--workload",
            "/opt/motionkernel/workloads/ltx_480p.yaml",
            "--artifact-root",
            str(artifact_root),
            "--output",
            str(output),
            "--model",
            MODEL_ID,
            "--runs",
            "15",
            "--kernel-saving-ms",
            str(kernel_saving_ms),
        ],
        env=os.environ.copy(),
        capture_output=True,
        cwd="/opt/motionkernel",
    )
    record_path = output / "measurement.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
    receipt = {
        "returncode": completed.returncode,
        "stdout_tail": (completed.stdout or "")[-4000:],
        "stderr_tail": (completed.stderr or "")[-4000:],
        "measurement": str(record_path),
        "status": record.get("status"),
        "arch": record.get("arch"),
        "e2e": record.get("e2e"),
        "e2e_overhead": record.get("e2e_overhead"),
    }
    (RUN_ROOT / "measure_receipt-sm90.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    runs.commit()
    if completed.returncode != 0:
        raise RuntimeError(f"sm90 measurement failed: {json.dumps(receipt)[:2000]}")
    return json.dumps(receipt, indent=2)


@app.function(
    image=ltx_image,
    gpu="H100!",
    timeout=12 * 60 * 60,
    scaledown_window=60,
    volumes=VOLUMES,
)
def run_all() -> str:
    """bench -> package -> measure in one H100 session (weights pre-seeded)."""
    bench = json.loads(bench_sm90.local())
    package = json.loads(package_sm90.local())
    measurement = json.loads(measure_sm90.local())
    return json.dumps(
        {"bench": bench, "package": package, "measurement": measurement}, indent=2
    )


@app.local_entrypoint()
def main() -> None:
    print(run_all.remote())
