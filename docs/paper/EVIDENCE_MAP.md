# Paper evidence map

For each claimed contribution: the exact committed records that back it. A
claim without a record here does not go in the paper.

The rule this file enforces is that prose follows evidence, not the reverse. It
is written before the prose so a gap shows up as an empty cell rather than as a
sentence that sounds supported.

## Contributions and their records

### (a) Isolated 10x kernel speedup vs end-to-end-neutral result

| item | record |
|---|---|
| isolated operator results | `docs/WAN_KERNEL_RESULTS.md` |
| end-to-end outcome | `docs/support-evidence/wan-t2v-1.3b-480p--sm100.json` -- `no_worthwhile_candidate`, 1.0073x against a 1.01x gate |

The point of the pairing: a kernel that is much faster in isolation moved
end-to-end by less than the gate. Both halves are needed or the claim is a
kernel benchmark.

### (b) R4 agent-failure taxonomy and gate catch-rates

| item | record |
|---|---|
| taxonomy and the failures themselves | `docs/LTX_V1_R4_ROOT_CAUSE.md` sections 1-5 |
| parity policy that closes the tolerance hole | `autokernel/verification/policy.py`, `tests/test_parity_policy.py` |
| the four quarantined artifacts and the arithmetic that disqualifies them | `docs/LTX_V1_R4_ROOT_CAUSE.md` section 4 |

**Gap:** "catch-rate" is not yet a measured quantity. We have the taxonomy and
the gates, not a denominator. Either compute it from the experiment store once
enough campaigns have run, or drop the word.

### (c) The 3.104 ms dispatch tax and its elimination

| item | record |
|---|---|
| the tax, measured | `docs/LTX_V1_R4_ROOT_CAUSE.md` section 7 |
| its elimination | `docs/DISPATCH_OVERHEAD.md`, `docs/dispatch-evidence/*.measurement.json` |
| variance controls and their honest limits | `docs/dispatch-overhead/VARIANCE_CONTROLS.md` |

**Caveat that must survive into the prose:** R4 section 8 records that
CUDA-graphing the block removes the *whole block's* host-side dispatch cost, so
the framework contributes more of the measured gain than the kernel does. And
per VARIANCE_CONTROLS.md, LTX dispatch A/Bs are currently ungateable on this
cluster -- the before/after numbers span 1.0274x-1.2459x across three runs of
the same measurement.

### (d) Attention rejections and the SSIM/LPIPS divergence

| item | record |
|---|---|
| both campaigns, full numbers | `docs/ATTENTION_CAMPAIGN_RESULTS.md` |
| SAGE_ATTN receipt | `/mnt/nfs/vlm-aryan/attn-ab-1041/campaign.json` (SLURM 1041, fidelity 1051) |
| VSA receipt | `/mnt/nfs/vlm-aryan/vsa-ab-1046/campaign.json` (SLURM 1046, fidelity 1053) -- **reconstructed from the run log**, marked as such in the record |
| attention share / Amdahl ceiling | `docs/ATTENTION_CAMPAIGN_RESULTS.md` round 2 step 1, `scripts/attention_share.py` |
| matrix cells | `docs/support-evidence/wan-t2v-1.3b-480p-attention*--sm100.json` |

Both candidates pass LPIPS and fail SSIM. That divergence is the interesting
claim and it is measured, not argued: gating on either metric alone reaches the
opposite conclusion in one direction or the other.

### (e) Tiered fidelity contracts

| item | record |
|---|---|
| the contract | `docs/TIERED_FIDELITY.md`, `autokernel/verification/fidelity.py` |
| the gate | `autokernel/artifact/finalizer.py` (`GenerationOutcome.decide`) |
| control tests | `tests/test_fidelity_tiers.py` -- lossy control rejected, known-good approximate passes with margins |
| perceptual harness, SSIM cross-checked against scikit-image | `autokernel/verification/perceptual.py`, `tests/test_perceptual_harness.py` |

## Open evidence gaps

| gap | what would close it | owner |
|---|---|---|
| **Tier-2 promotion** -- no artifact has yet been promoted under a tier-2 budget | the caching campaign; blocked on FastVideo's `enable_teacache` being an inert flag, so the cache must be implemented in the denoising loop first | Agent 1 |
| **Second architecture** -- every e2e number is sm100 | the sm90 LTX A/B; the bench ran but e2e is parked on the VAE-offload memory regime | was Agent 2, now Agent 1 |
| **Gate catch-rate** (contribution b) | a denominator from the experiment store once enough campaigns have run | — |
| **720p attention share** | measure it on `wan-t2v-1.3b-720p` before running candidates there, as was done at 480p | Agent 1 |

Nothing in this table may be written as a result until its record exists.
