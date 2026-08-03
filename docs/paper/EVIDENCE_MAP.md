# Evidence map

Rule: every number in the paper traces to a committed record listed here.
If a re-run changes a number, update this file first, then the prose.
Status: ✅ committed on main · 🔀 on an unmerged branch · ⏳ pending campaign.

## C2 — The promotion gap

| claim | value | record | status |
|---|---|---|---|
| Wan post-attn gated residual + LN, isolated weighted speedup | 8.638x | `docs/WAN_KERNEL_RESULTS.md` | ✅ |
| Wan modulated pre-attn LayerNorm, isolated | 10.449x | `docs/WAN_KERNEL_RESULTS.md` | ✅ |
| Wan post-MLP gated residual, isolated | 10.628x | `docs/WAN_KERNEL_RESULTS.md` | ✅ |
| Wan 50-step generation unchanged (~36.67 s both arms) | ~1.00x e2e | `docs/FASTVIDEO_UNIVERSAL_OPTIMIZATION_AGENT_PLAN.md` (progress section); FastVideo A/B script | ✅ (strengthen: re-run under variance controls) |
| Four LTX candidates: aggregate e2e share | 6.20% | `docs/LTX_V1_R4_ROOT_CAUSE.md` §4 | ✅ |
| Max achievable e2e from all four | 1.0064x (< 1.01 gate) | `docs/LTX_V1_R4_ROOT_CAUSE.md` §4 | ✅ |
| Amdahl estimate overstates a 1.115x kernel's return | >8x | `docs/LTX_V1_R4_ROOT_CAUSE.md` §4 | ✅ |

## C3 — Dispatch tax

| claim | value | record | status |
|---|---|---|---|
| Overhead per candidate call (eager FX replay) | 3.104 ms | `docs/LTX_V1_R4_ROOT_CAUSE.md` §3 | ✅ |
| Kernel saving per call | 124 µs (259.05→134.90 µs) | same | ✅ |
| Transformer: calls/generation, overhead share | 384 calls, 1.192 s, 32.4% e2e | same | ✅ |
| VAE contrast: overhead share | 14.5 ms, 0.39% e2e | same | ✅ |
| Tax eliminated (guarded compiled dispatch) | see records | `docs/dispatch-overhead/sm100-*.measurement.json` | 🔀 PR #21 |
| sm90 (H100) e2e A/B | — | Track D phase 3 | ⏳ Agent 2 |

## C4 — Failure taxonomy (agent-generated kernels)

| failure | evidence | record | status |
|---|---|---|---|
| Layout compiled in as constexpr (bit-exact, wrong contract) | stride 4096 vs 16384 assertion | `docs/LTX_V1_R4_ROOT_CAUSE.md` §1a | ✅ |
| Dynamo guard bypass (`_COMPILED_ENTRY` direct call) | unfair 2.557x | `docs/LTX_V1_R4_ROOT_CAUSE.md` §1b | ✅ |
| share_of_e2e computed as 1.0333 (inclusive vs exclusive) | metric audit | `docs/LTX_V1_R4_ROOT_CAUSE.md` §5 | ✅ |
| call count 195,661 vs 1,151 real | metric audit | same | ✅ |
| status `finalized` on quarantined artifacts | audit | same | ✅ |
| Per-gate catch rates across all campaigns | — | derive from experiment store | ⏳ Agent 2 phase 4 |

## C5 — Tiered fidelity contracts

| claim | record | status |
|---|---|---|
| Contract design (tiers, budgets, per-workload declaration) | `docs/TIERED_FIDELITY.md` | 🔀 PR #18 |
| Verification harness | tiered-fidelity-contracts branch (Modal harness) | 🔀 PR #18 |
| SSIM/LPIPS divergence: LPIPS admits both attention candidates, SSIM rejects both | `docs/ATTENTION_CAMPAIGN_RESULTS.md` | 🔀 PR #19 |

## C6 — Negative results (attention round 1)

| claim | value | record | status |
|---|---|---|---|
| SageAttention 1.0.6 on Wan 480p sm100 | 0.8031x, SSIM 0.9353 (floor 0.97) FAIL, LPIPS 0.0378 pass | `docs/ATTENTION_CAMPAIGN_RESULTS.md` (SLURM 1041/1051) | 🔀 PR #19 |
| VIDEO_SPARSE_ATTN zero-shot on vanilla Wan | 0.5169x, SSIM 0.9527 FAIL, LPIPS 0.0253 pass | same (SLURM 1046/1053) | 🔀 PR #19 |
| Regime analysis (why they lose here) | prose | same §"What the numbers say" | 🔀 PR #19 |

## C1 / §8 — End-to-end results

| claim | value | record | status |
|---|---|---|---|
| LTX2 promoted artifact, median A/B | 3.3646→3.0991 s, 1.0857x, 15 runs | README status; `docs/LTX_V1_R4_ROOT_CAUSE.md` §6 (SLURM 986) | ✅ |
| Independent replication | 1.2514x | README status | ✅ (locate + link the raw record) |
| Dispatch count / fallbacks / parity | 6,143 / 0 / byte-identical | README status; R4 §6 | ✅ |
| Support matrix (model × workload × arch) | — | `docs/SUPPORT_MATRIX.md` + `support_matrix.json` | 🔀 PR #22 |
| Native-arm variance problem + fix | 3.12–4.01 s spread → post-fix | `docs/dispatch-overhead/` before/after | ⏳ Agent 2 phase 2 |
| Tier-2 promoted artifact (caching) | target ≥1.3x + quality evidence | Track C campaign | ⏳ Agent 1 phase 2 |
| Attention round 2 (SageAttention2/2++, VSA-on-FastWan, 720p) | — | Track A phase 3 | ⏳ Agent 1 |

## §6 additions — self-corrections and audit-caught defects

| claim | value | record | status |
|---|---|---|---|
| Attention share estimate corrected (inclusive column, chain counted 4x) | 45.94% → 27–35% (wan-480p) | PR #23 / attention-share records | 🔀 verify after merge |
| LTX attention share → excluded on ceiling | 15.13% → 1.178x ceiling | same | 🔀 |
| Author-published VSA numbers corrected via store ingest | job-log vs ingested record | experiment-store PR | 🔀 |
| Amdahl ceiling for wan-480p attention | 1.373–1.537x | same records | 🔀 |

## §6 — null intervention incident (job 1078)

| claim | value | record | status |
|---|---|---|---|
| Paired A/B reported 0.9011x, CI [0.868, 0.920], conclusive, valid clock trace — with both arms native | job 1078 | paired-protocol measurement record + run log (zero candidate calls) | ✅ (commit the record + log excerpt) |
| Candidate arm env vars inert (checkout reads only FASTVIDEO_OPTIMIZATION_CAPTURE) | code inspection | FastVideo checkout identity in the record | ✅ |
| Guard added: arms_differentiated=false invalidates before timings are read | schema change | paired-protocol PR | 🔀 |
| Paired vs sequential agreement where CV was already low (gap 5) | 0.8106x [0.8091, 0.8111] vs 0.8031x | jobs 1077/1078-adjacent records | ✅ |
| Warmup reaches and holds max boost without clock locking | 2062 MHz held through timed runs; plateau 77 s / 93 s | clock traces in v3 records | ✅ |

## §8 additions — measurement validity

| claim | value | record | status |
|---|---|---|---|
| Exclusive nodes do not collapse spread | 5.57% vs 4.74%/7.37% CV | variance before/after records (SLURM 1055) | ✅ (merged via #21/#22 follow-ups) |
| Clock locking unavailable; idle 285 MHz vs 2062 MHz max | environment records | same | ✅ |
| Run 1055 marked invalid for gating; three previously reported numbers retracted | 1.0274x, 1.2459x, 1.1155x | same | ✅ |
| Native-arm CV in attention campaign (valid runs) | 0.04% / 0.46% | attention campaign records | ✅ |

## Open evidence gaps (blockers for submission)

1. **Tier-2 promotion** — the paper's frontier claim needs at least one
   (caching campaign, Agent 1). Without it, §7 is design + rejections only.
2. **Second architecture** — sm90 rows (Agent 2). Without it, scope claims
   shrink to "on GB200".
3. **Per-gate catch rates** — needs the experiment store backfill (Agent 2).
4. **Replication record for 1.2514x** — the number is in the README; locate
   the raw measurement and link it here or drop the claim.
5. **Wan e2e-neutral re-run under variance controls** — current record
   predates clock locking; cheap to redo, strengthens T1.
6. **720p attention share** — unmeasured; decides whether attention
   round 2 is worth its budget and appears in §7's regime analysis.
7. **V1 LTX headline re-validation** — STILL OPEN, now with a documented
   near-false-retraction: job 1078's 0.9011x is a null-intervention
   measurement (both arms native; see the §6 incident section above) and
   must never be cited as a re-validation result. Closing this gap
   requires the `agent/v1-r4-dispatch-fix` FastVideo checkout with
   `candidate_calls > 0` asserted before timings are read (the original
   proof's gate 3), under the paired protocol with
   `arms_differentiated=true`. Until then the 1.0857x/1.2514x numbers
   stay flagged in the abstract and are not presented as gated results.
8. **Gap 5 (Wan paired re-run) — CLOSED** for the attention A/B: paired
   0.8106x, CI [0.8091, 0.8111], agrees with sequential 0.8031x within
   1% at native CV 0.04%. Link the record in §8's protocol discussion as
   the "pairing changes nothing where there is no drift" data point.
