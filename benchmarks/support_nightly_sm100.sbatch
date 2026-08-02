#!/bin/bash
#SBATCH --job-name=mk-support-nightly
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/nfs/vlm-aryan/support-nightly/logs/nightly-%j.out
#SBATCH --chdir=/tmp
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.06-py3
#SBATCH --container-mounts=/mnt/nfs/vlm-aryan:/shared
#
# Track E nightly: discovery-only campaigns for every claimed family on sm100.
# One immutable directory per night under /shared/support-nightly/<date>/;
# distilled run records land in /shared/support-nightly/records/ and are
# committed to docs/support-evidence/ by the operator.
set -uo pipefail

MK=${MK_ROOT:-/shared/mk-track-e-support-matrix}
FV=${FASTVIDEO_CHECKOUT:-/shared/ltx-v1-r4-targeted-fix-20260801-203751/fastvideo}
PY=/shared/fastvideo-wan-venv/bin/python
export PYTHONPATH=$MK
export TRITON_CACHE_DIR=/tmp/triton-cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache
export HF_HOME=/shared/hf-cache

mkdir -p /shared/support-nightly/logs
ARCH=$($PY -c "import torch; c=torch.cuda.get_device_capability(0); print(f'sm{c[0]}{c[1]}')")
echo "arch: $ARCH  date: $(date -u +%F)"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1

$PY -m autokernel.support nightly \
  --fastvideo-checkout "$FV" \
  --nightly-root /shared/support-nightly \
  --records-dir /shared/support-nightly/records \
  --arch "$ARCH" \
  --budget-hours 1.5 \
  --repo-root "$MK" \
  --workload "$MK/workloads/ltx_480p.yaml" \
  --workload "$MK/workloads/wan_t2v_1.3b_480p.yaml" \
  --workload "$MK/workloads/wan_t2v_14b_480p.yaml" \
  --workload "$MK/workloads/fastwan_t2v_1.3b_480p.yaml" \
  --workload "$MK/workloads/hunyuan_t2v_480p.yaml" \
  --workload "$MK/workloads/cosmos25_t2v_2b_704p.yaml"
echo "nightly exit=$?"
echo "=== done ==="
