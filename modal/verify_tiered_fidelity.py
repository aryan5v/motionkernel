"""Modal verification for the tiered fidelity contracts (Track B).

Why this exists
---------------

The tier machinery is deliberately CPU-only: the gate must be decidable without
a GPU, an image backend, or a pretrained network. That is a design property
worth having, but it means the local test run on a laptop could not execute the
parts of the suite that hard-import torch -- 21 tests in ``test_cli_compat.py``
and ``test_builtin_specs.py`` were failing purely because torch was absent, and
5 further modules could not even be collected.

Those tests are not about fidelity tiers, but they are the ones that would
catch a tier change breaking the CLI or the built-in specs. Running the whole
suite against a real torch install is what turns "no regressions observed" into
"no regressions".

The SLURM cluster is unavailable (all 12 GB200 nodes held by another user's
job with ``StartTime=Unknown``), so this runs on Modal under the hao-ai-lab
workspace instead.

Usage
-----

    modal run modal/verify_tiered_fidelity.py::full_suite
    modal run modal/verify_tiered_fidelity.py::tier_gates_only
"""

from __future__ import annotations

import json
import subprocess

import modal

APP_NAME = "motionkernel-tiered-fidelity-verify"

#: The Track B commit under test. Pinned rather than tracking a branch so a
#: verification receipt names exactly what it verified.
MOTIONKERNEL_COMMIT = "30e9bc9bea4a3fb3d4b39f4579021aece6123cec"

#: The merge-base on ``main``. A failure that reproduces here is not ours, and
#: saying so requires running it rather than assuming it.
BASELINE_COMMIT = "2b1da46"

app = modal.App(APP_NAME)


def _image(commit: str) -> modal.Image:
    return (
        modal.Image.from_registry("nvcr.io/nvidia/pytorch:25.06-py3")
        .apt_install("git")
        .pip_install(
            "pytest",
            "pyyaml>=6.0.1",
            # scikit-image is not a runtime dependency. It is installed only so
            # the hand-written SSIM can be cross-checked against a reference
            # implementation; without it that test skips rather than fails.
            "scikit-image",
        )
        .run_commands(
            "git clone https://github.com/aryan5v/motionkernel.git /opt/motionkernel",
            f"git -C /opt/motionkernel checkout --detach {commit}",
        )
        .env(
            {
                "PYTHONPATH": "/opt/motionkernel",
                "PYTHONDONTWRITEBYTECODE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
    )


image = _image(MOTIONKERNEL_COMMIT)
baseline_image = _image(BASELINE_COMMIT)


def _run_pytest(args: list[str]) -> dict:
    """Run pytest in the pinned checkout and return a structured receipt."""
    import torch

    completed = subprocess.run(
        ["python", "-m", "pytest", *args, "-q", "--no-header", "-rf"],
        cwd="/opt/motionkernel",
        capture_output=True,
        text=True,
    )
    stdout = completed.stdout
    print(stdout[-20000:])
    if completed.stderr.strip():
        print("--- stderr ---")
        print(completed.stderr[-4000:])

    summary = ""
    for line in reversed(stdout.strip().splitlines()):
        if " passed" in line or " failed" in line or " error" in line:
            summary = line.strip()
            break

    return {
        "commit": MOTIONKERNEL_COMMIT,
        "args": args,
        "returncode": completed.returncode,
        "summary": summary,
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


@app.function(image=image, gpu="H100!", timeout=30 * 60, scaledown_window=60)
def full_suite() -> str:
    """Run the entire MotionKernel test suite with torch present.

    This is the run the laptop could not do. A green result here is what
    licenses the claim that the tier changes introduce no regressions.
    """
    receipt = _run_pytest(["tests/"])
    rendered = json.dumps(receipt, indent=2, sort_keys=True)
    print(rendered)
    return rendered


@app.function(
    image=baseline_image, gpu="H100!", timeout=30 * 60, scaledown_window=60
)
def baseline_suite() -> str:
    """Run the same suite at the merge-base, for a like-for-like comparison.

    Any test failing in both runs is pre-existing on ``main``, and the only
    honest way to establish that is to run it rather than to reason about
    whether the change could plausibly have caused it.
    """
    receipt = _run_pytest(["tests/"])
    receipt["commit"] = BASELINE_COMMIT
    rendered = json.dumps(receipt, indent=2, sort_keys=True)
    print(rendered)
    return rendered


@app.function(image=image, gpu="H100!", timeout=15 * 60, scaledown_window=60)
def tier_gates_only() -> str:
    """Run just the tier and perceptual suites, including the SSIM cross-check.

    Faster feedback loop while iterating on the contract itself.
    """
    receipt = _run_pytest(
        ["tests/test_fidelity_tiers.py", "tests/test_perceptual_harness.py", "-v"]
    )
    rendered = json.dumps(receipt, indent=2, sort_keys=True)
    print(rendered)
    return rendered


@app.local_entrypoint()
def main() -> None:
    print(tier_gates_only.remote())
    print(full_suite.remote())
