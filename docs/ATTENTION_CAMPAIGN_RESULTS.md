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

---

# Round 2

## Step 1 — attention's share of end-to-end, measured before spending budget

The >=1.3x gate is an Amdahl bound. If attention is a share S of device time,
then even *infinitely fast* attention caps end-to-end speedup at `1/(1-S)`.
Below S ~= 0.23 the ceiling is under 1.3x and no backend can pass, however good
its kernels are. Measuring S first is the difference between a campaign and a
way of spending GPU hours to rediscover arithmetic.

Measured from our own profiler exports (`universal-profiler-wan-worker`,
`universal-profiler-ltx-worker`) with `scripts/attention_share.py`.

| workload | attention share | Amdahl ceiling | >=1.3x reachable? |
|---|---|---|---|
| `wan-t2v-1.3b-480p` | 27.17% - 34.95% | 1.3730x - 1.5372x | **yes**, with little margin |
| `ltx-480p` | 15.13% | 1.1783x | **no** |

**LTX is excluded from round 2 attention work.** At a 1.178x ceiling the 1.3x
gate cannot be met by any attention backend, including a hypothetical one that
takes zero time. Running candidates there would produce guaranteed rejections
at full GPU cost. This is a property of the workload, not of any candidate.

**Wan remains a valid target, but the margin is thin.** A ceiling of
1.373x-1.537x means a candidate must remove 70-85% of attention time just to
reach the gate. That is a demanding bar, and it should temper expectations for
step 2 rather than be discovered afterwards.

### How the share was attributed, and why it is a range

This is the part that is easy to get wrong, and this project has gotten it
wrong before. Profiler rows carry both `cuda_time_us` (inclusive of children)
and `self_cuda_time_us` (exclusive), and a single attention call appears in up
to four rows of one dispatch chain -- outer custom op, autograd Function, inner
op, and the CUDA kernel -- several sharing the same time.

Summing the inclusive column inflated the Wan share to 45.94% on the first
attempt. That is the identical defect recorded in `LTX_V1_R4_ROOT_CAUSE.md`
section 5: *"attributed 2864051.95us of a 2771790.13us total; inclusive ranges
summed against an exclusive total"*.

Self time alone is still not enough, because framework-operator rows and
device-kernel rows are two views of the same GPU work: for Wan,
`flash_attn::_flash_attn_forward` and `void flash::flash_fwd_kernel<...>` both
report 1.3673 s, and the two view totals sum exactly to the profiler's reported
total. So the share is computed independently in each population:

| workload | view | total | attention | share |
|---|---|---|---|---|
| wan | operators | 5.061 s | 1.375 s | 27.17% |
| wan | device kernels | 3.934 s | 1.375 s | 34.95% |
| ltx | operators | 2.771 s | 0.419 s | 15.13% |
| ltx | device kernels | (no kernel rows in this export) | | |

Both Wan views agree on the attention *time* (1.375 s); they differ only in
what they divide it by. The range is reported rather than a point estimate,
because picking whichever end supported the decision we wanted would be exactly
the failure this section exists to avoid.

LTX's export contains no device-kernel rows, so only the operator view is
available there. Its verdict does not depend on the choice: 15.13% is below the
threshold under any attribution.

### Provenance

Profiler exports predate the round-1 campaign and were produced by earlier
discovery runs, so the step counts may differ from the campaign workloads.
Attention share is a function of architecture and sequence length rather than
step count, so the ratio carries; the absolute seconds do not.
