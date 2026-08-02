#!/bin/bash
#SBATCH --job-name=mk-dispatch-overhead
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/nfs/vlm-aryan/mk-track-d-dispatch-runs/logs/dispatch-overhead-%j.out
#SBATCH --chdir=/tmp
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.06-py3
#SBATCH --container-mounts=/mnt/nfs/vlm-aryan:/shared
#
# Track D: re-measure the per-call dispatch overhead of the CUDA-graph
# dispatch path (sm100, GB200). Same method as R4 section 7: 15 timed runs
# per arm end-to-end A/B plus an in-situ shadow profile. Evidence lands in a
# fresh immutable run directory; nothing writes into the R4 directories.
set -uo pipefail

MK=${MK_ROOT:-/shared/mk-track-d-dispatch}
R4=/shared/ltx-v1-r4-targeted-fix-20260801-203751
RUNS_ROOT=/shared/mk-track-d-dispatch-runs
OUT=$RUNS_ROOT/sm100-$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_ID}
PY=/shared/fastvideo-wan-venv/bin/python

export PYTHONPATH=$MK
export TRITON_CACHE_DIR=/tmp/triton-cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-cache
export HF_HOME=/shared/hf-cache

mkdir -p "$RUNS_ROOT/logs" "$OUT"
echo "evidence dir: $OUT"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1
echo "motionkernel HEAD: $(git -C $MK rev-parse HEAD 2>/dev/null || echo unknown)"
echo "fastvideo HEAD: $(git -C $R4/fastvideo rev-parse HEAD 2>/dev/null || echo unknown)"

# LTX transformer artifact mk-2c92e356aa34bc0d-7df21b47-sm100; isolated kernel
# saving 0.124 ms/call (259.05us -> 134.90us, R4 isolated bench).
$PY -m autokernel.dispatch measure \
  --fastvideo-checkout "$R4/fastvideo" \
  --workload "$MK/workloads/ltx_480p.yaml" \
  --artifact-root "$R4/transformer-artifacts" \
  --output "$OUT" \
  --model "FastVideo/LTX2-Distilled-Diffusers" \
  --runs 15 \
  --kernel-saving-ms 0.124
status=$?
echo "measure exit=$status"
if [ -f "$OUT/measurement.json" ]; then
  $PY -m autokernel.dispatch summarize "$OUT/measurement.json"
fi
echo "=== done ==="
exit $status
