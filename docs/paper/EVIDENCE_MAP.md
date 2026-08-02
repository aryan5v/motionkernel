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
