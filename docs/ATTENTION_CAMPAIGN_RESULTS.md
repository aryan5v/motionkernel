# Attention backend campaign results

Wan 2.1 T2V 1.3B, 480x832, 49 frames, 20 steps, seed 1024, single GB200
(sm100), torch 2.8.0a0. 15 timed runs per arm, 1 warmup, both arms in one job
on one node. Baseline is `FLASH_ATTN` pinned explicitly in both experiments.

## Results

| candidate | speedup (median) | worst-frame SSIM | worst-frame LPIPS | verdict |
|---|---|---|---|---|
| `SAGE_ATTN` (SageAttention 1.0.6) | **0.8031x** | 0.9353 (floor 0.97) | 0.0378 (ceiling 0.05) | **rejected** |

Gate: >=1.3x end-to-end, SSIM >= 0.97, LPIPS <= 0.05.

## SAGE_ATTN (SLURM 1041, fidelity 1051)

| arm | backend | effective | median | stdev | min |
|---|---|---|---|---|---|
| native | FLASH_ATTN | FLASH_ATTN | 15.1683 s | 0.0054 | 15.1628 |
| optimized | SAGE_ATTN | SAGE_ATTN | 18.8880 s | 0.0747 | 18.8832 |

**0.8031x median, 0.8030x min-to-min.** SageAttention is 24% *slower* than
FlashAttention here.

Perceptual, 49 frames, worst-frame aggregation:

| metric | value | threshold | margin | passed |
|---|---|---|---|---|
| ssim | 0.935326 | 0.970 | **-0.034674** | no |
| lpips | 0.037841 | 0.050 | +0.012159 | yes |

Rejected on **both** axes independently: it is slower than the baseline *and*
outside the structural-similarity budget.

Note the two perceptual metrics disagree. LPIPS -- a learned perceptual
distance -- finds the output acceptable, while SSIM does not. Quantization
noise is distributed in a way LPIPS is relatively insensitive to. Gating on one
metric alone would have produced a different answer here, which is the argument
for declaring both.

### Why this is not a tuning failure

SageAttention v1 quantizes the attention product. On sm100 FlashAttention is
already well tuned, so the quantize/dequantize overhead exceeds the saving.
That is a property of the approach on this hardware, not a parameter that could
be searched, so no threshold sweep would rescue it.

This is SageAttention **v1** (PyPI `sageattention==1.0.6`), which builds and
runs on sm100. v2 requires a source build and has **not** been evaluated here;
its kernels differ and this result should not be read as a v2 result.

## Provenance

Both arms verify the *effective* backend before timing. FastVideo substitutes
FlashAttention silently when an optional backend cannot be imported, and on
this cluster `sageattention`, `sageattn3` and `fastvideo_kernel` were all
absent from the base image (SLURM 1031) -- so an unguarded campaign here would
have measured FlashAttention twice and reported it as a SageAttention result of
~1.00x. Every receipt records `effective` alongside `requested`; both runs above
confirm the candidate backend genuinely executed.

Baseline reproducibility across independent jobs: 15.1683 s (1041) vs 15.2035 s
(1046), 0.2% apart.
