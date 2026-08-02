# Attention backend campaign results

Wan 2.1 T2V 1.3B, 480x832, 49 frames, 20 steps, seed 1024, single GB200
(sm100), torch 2.8.0a0. 15 timed runs per arm, 1 warmup, both arms in one job
on one node. Baseline is `FLASH_ATTN`, pinned explicitly in both experiments so
the comparison never depends on automatic selection.

## Summary

| candidate | speedup (median) | worst-frame SSIM | worst-frame LPIPS | verdict |
|---|---|---|---|---|
| `SAGE_ATTN` (SageAttention 1.0.6) | **0.8031x** | 0.9353 (floor 0.97) | 0.0378 (ceiling 0.05) | **rejected** |
| `VIDEO_SPARSE_ATTN` | **0.5169x** | 0.9527 (floor 0.97) | 0.0253 (ceiling 0.05) | **rejected** |

Gate: >=1.3x end-to-end, SSIM >= 0.97, LPIPS <= 0.05.

**Neither candidate beats FlashAttention on this workload, and both fail the
perceptual budget.** Track A's >=1.3x exit criterion is *not* met, and the
reason is not tuning: both are slower than the baseline, one of them by half.

## SAGE_ATTN (SLURM 1041, fidelity 1051)

| arm | backend | effective | median | stdev | min |
|---|---|---|---|---|---|
| native | FLASH_ATTN | FLASH_ATTN | 15.1683 s | 0.0054 | 15.1628 |
| optimized | SAGE_ATTN | SAGE_ATTN | 18.8880 s | 0.0747 | 18.8832 |

0.8031x median, 0.8030x min-to-min -- 24% *slower* than FlashAttention.

| metric | value | threshold | margin | passed |
|---|---|---|---|---|
| ssim | 0.935326 | 0.970 | **-0.034674** | no |
| lpips | 0.037841 | 0.050 | +0.012159 | yes |

## VIDEO_SPARSE_ATTN (SLURM 1046, fidelity 1053)

| arm | backend | effective | median | stdev | min |
|---|---|---|---|---|---|
| native | FLASH_ATTN | FLASH_ATTN | 15.2035 s | 0.0704 | 15.1610 |
| optimized | VIDEO_SPARSE_ATTN | VIDEO_SPARSE_ATTN | 29.4108 s | 0.0242 | 29.3783 |

0.5169x median -- nearly **2x slower**.

| metric | value | threshold | margin | passed |
|---|---|---|---|---|
| ssim | 0.952735 | 0.970 | **-0.017265** | no |
| lpips | 0.025329 | 0.050 | +0.024671 | yes |

## What the numbers say

**Both candidates are slower, and the faster-degrading one is the less
degrading one.** VSA is perceptually closer to the baseline than SageAttention
(SSIM 0.9527 vs 0.9353, LPIPS 0.0253 vs 0.0378) while being considerably
slower. There is no speed/fidelity trade-off to tune along here: one candidate
is worse on both counts than the other, and both are worse than doing nothing.

**Why they lose.** Both approaches assume attention dominates and that the
baseline kernel is beatable. Neither holds here. SageAttention v1 quantizes the
attention product; on sm100 the quantize/dequantize overhead exceeds the saving
against a well-tuned FlashAttention. VSA skips attention blocks, but its tile
selection and gather/scatter cost more than the blocks it removes at this
sequence length (81120 tokens). Both are designed for regimes -- older
architectures, longer contexts -- that this workload is not in.

**LPIPS and SSIM disagree, consistently.** Both candidates pass LPIPS and fail
SSIM. A budget gating on LPIPS alone would have admitted both; one gating on
SSIM alone would have rejected both without noticing they were perceptually
quite close. That is the argument for declaring both rather than the one that
is easiest to satisfy.

## Provenance: why these numbers can be trusted

Both experiments verify the *effective* backend before timing and record it
alongside the requested one. This is not ceremony. On this cluster
`sageattention`, `sageattn3` and `fastvideo_kernel` were **all absent from the
base container image** (SLURM 1031), and FastVideo substitutes FlashAttention
silently when an optional backend cannot be imported. An unguarded campaign
would have run FlashAttention in both arms and reported ~1.00x as a
SageAttention result -- a phantom "no regression" for a backend that never
executed.

Every receipt above records `effective` == `requested`, so both candidates
genuinely ran.

Baseline reproducibility across independent jobs, hours apart:
15.1683 s (1041) vs 15.2035 s (1046) -- **0.2% apart**.

## Scope

- SageAttention here is **v1** (PyPI `sageattention==1.0.6`), which builds and
  runs on sm100. v2 needs a source build and was **not** evaluated; its kernels
  differ and this is not a v2 result.
- One model, one resolution, one step count, one architecture. A backend that
  loses at 480p/49 frames on GB200 may win at 720p on a 14B model, where
  attention takes a larger share and sequences are longer. These results do not
  generalize beyond the row they were measured on -- which is the argument for
  the support matrix in Track E.
