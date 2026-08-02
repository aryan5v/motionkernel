# Paper outline

Working title (pick one before submission):

1. **The Promotion Gap: Verified End-to-End Deployment of Agent-Generated
   GPU Optimizations for Video Diffusion Inference**
2. Trust but Verify: A Promotion Pipeline for AI-Generated Kernels in
   Production Video Generation
3. From Isolated Speedups to Shipped Speedups: Fail-Closed Promotion of
   Agent-Generated Optimizations

Target: arXiv (cs.DC, cross-list cs.LG and cs.PF) first, then MLSys.
Format: arXiv has no page limit; write to MLSys style (~12 pages + appendix)
so the same source serves both.

## One-paragraph thesis

Coding agents can now produce GPU kernels that pass correctness suites and
show large isolated speedups, yet deliver nothing — or worse — in production.
We present a promotion pipeline that measures optimizations the way users
experience them (full-generation latency and output quality on declared
workloads), and we report what it caught: 8.6–10.6x isolated operator
speedups with zero end-to-end effect, a dispatch mechanism that charged 25x
more than a kernel saved, agent-generated kernels that were bit-exact yet
compiled one call's memory layout in as constants, and published attention
methods that lose to a tuned baseline in a regime their papers do not test.
The pipeline's fail-closed gates, tiered fidelity contracts, and
measured-impact accounting are, we argue, the missing infrastructure layer
for the era of agent-generated performance code.

## Contributions (each must map to committed evidence — see EVIDENCE_MAP.md)

C1. A fail-closed, end-to-end promotion pipeline for agent-generated
    optimizations: declarative workloads, metadata-only graph capture with
    export/Dynamo fallback, measured impact ranking, graph-derived spec
    generation, sandboxed autonomous search, independent validation,
    versioned artifacts, runtime dispatch, full-generation A/B, promotion.

C2. The promotion gap, quantified: isolated operator speedup is
    uncorrelated with shipped value. Three Wan fusions at 8.6–10.6x
    isolated → 50-step generation unchanged; four LTX candidates covering
    6.20% of end-to-end aggregate → max achievable 1.0064x, below a 1.01x
    gate.

C3. Dispatch-tax accounting: delivering a 124 µs/call saving cost
    3.104 ms/call (25:1) through eager FX graph replay — 32.4% of
    end-to-end time at 384 calls/generation — and its measured elimination
    via guarded compiled dispatch.

C4. A failure taxonomy of agent-generated kernels, with per-gate catch
    rates: bit-exact kernels with call-specific layouts compiled in as
    constexpr; guard-bypassing dispatch producing unfair speedups (2.557x
    reported, invalid); inflated impact estimates (share_of_e2e 1.0333;
    call counts 170x off).

C5. Tiered fidelity contracts (bitwise / numeric tolerance / perceptual
    budget) and honest campaign results under them, including the finding
    that SSIM and LPIPS systematically disagree on attention approximations
    (LPIPS admitted both candidates; SSIM rejected both).

C6. Negative results the field needs: SageAttention v1 at 0.8031x and
    training-free VSA at 0.5169x versus tuned FlashAttention on
    Wan 2.1 1.3B 480p on GB200/sm100 — published methods evaluated outside
    their reported regimes.

## Section plan

1. **Introduction** — the era of agent-generated performance code; the
   evaluation gap (KernelBench-style isolated metrics); thesis; contributions.
2. **Background and related work** — LLM/agent kernel generation and its
   evaluation practice; video DiT inference cost structure; acceleration
   families (sparse/quantized attention, caching, compilation/megakernels);
   why none of the existing evaluation practice measures shipped value.
3. **System** — the pipeline stage by stage, with the trust boundaries:
   metadata-only capture (what is never serialized), the pure-tensor
   allowlist and fail-closed rejection, sandboxed candidate search, the
   write-once run contract, independent validation, artifact bundles and
   compatibility-checked dispatch, finalization gates.
4. **The promotion gap** — Wan case study (C2), the arithmetic of
   sub-percent regions, measured- vs estimated-impact ranking.
5. **The dispatch tax** — accounting methodology, the 25:1 result, the fix,
   before/after measurements (C3).
6. **What agents get wrong** — R4 taxonomy with reproduced artifacts and
   per-gate catch rates (C4). This section is the paper's teeth.
7. **Fidelity tiers and approximate optimizations** — contract design (C5),
   attention campaign with rejections (C6), caching campaign
   [PENDING: Track C promotion], metric-divergence analysis.
8. **End-to-end results** — LTX2 promoted artifact (1.0857x median,
   1.2514x replication, 6,143 dispatches, zero fallbacks, byte-identical
   frames); support matrix across models and architectures
   [PENDING: sm90 results]; variance-control methodology.
9. **Limitations and discussion** — single-framework integration depth
   (FastVideo), what generalizes; the honest scope of "verified".
10. **Conclusion.**

## Figures and tables (target list)

- F1: pipeline diagram with trust boundaries and terminal states.
- T1: Wan isolated vs end-to-end (the 10x → 1.00x table).
- T2: LTX candidate arithmetic (share, speedup, realized gain, verdict).
- F2: dispatch-tax waterfall (saving vs overhead per call; before/after fix).
- T3: failure taxonomy × gate catch matrix.
- T4: attention campaign (speedup, SSIM, LPIPS, verdict per candidate).
- F3: SSIM vs LPIPS divergence scatter across candidates/frames.
- T5: support matrix snapshot (model × workload × arch → verdict).
- F4: variance before/after clock locking + exclusive nodes.

## arXiv logistics

- LaTeX (main.tex here compiles standalone; add figures as PDFs).
- Category cs.DC, cross-list cs.LG, cs.PF. Moderation checks topical fit,
  not novelty — a well-formed systems paper will not be rejected.
- First-time submitter in cs needs an **endorsement**: plan for it (a
  co-author or colleague with arXiv standing, or request endorsement via
  the submission flow — takes days, so start early).
- License: arXiv non-exclusive license is the safe default (keeps MLSys
  submission options open; check the venue's preprint policy — MLSys
  allows preprints).
- Reproducibility: link the repo tag + evidence records; every number in
  the paper must trace to a committed measurement record (this is our
  differentiator — lean into it with an artifact appendix).

## Authorship / integrity notes

- List human authors; acknowledge agent tooling in an acknowledgments or
  methods note (agents wrote code under human-directed campaigns; all
  measurements machine-recorded). Do not list AI systems as authors
  (arXiv policy).
- Numbers in prose must match EVIDENCE_MAP.md exactly. If a campaign
  re-run changes a number, update the map first, then the prose.
