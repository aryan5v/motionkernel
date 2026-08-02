# Paper outline

Working title: *Verified kernel optimization for video diffusion: what the
gates caught.*

The organizing claim is not "we made video generation faster". It is that an
optimization platform is only as good as the evidence it refuses to accept, and
that most of what we learned came from candidates the gates rejected.

1. **Introduction.** Video diffusion inference is expensive; the optimization
   literature reports isolated kernel speedups; the gap between an isolated
   speedup and an end-to-end one is where the difficulty lives.

2. **The gap, measured.** Contribution (a): a kernel much faster in isolation,
   end-to-end neutral. Amdahl bounds computed from our own profiles, including
   the attention-share analysis that excluded a workload before we spent
   budget on it.

3. **Failure taxonomy.** Contribution (b): what an autonomous agent actually
   gets wrong -- compiled-in strides, bypassed guards, inclusive ranges summed
   against exclusive totals, quarantine reasons that describe the wrong cause.
   Each with the gate that now catches it.

4. **Dispatch.** Contribution (c): the 3.104 ms per-call tax, its
   mechanism, its elimination via CUDA-graph capture, and the honest note on
   how much of the resulting gain is framework rather than kernel.

5. **Fidelity contracts.** Contribution (e): why widening numeric tolerance to
   admit approximate work is the wrong axis, and what a second (perceptual)
   axis buys. The three tiers and the control experiments.

6. **Attention: two rejections.** Contribution (d): SageAttention v1 and VSA,
   both slower and both outside the SSIM floor on Wan 480p/sm100, with the
   SSIM/LPIPS divergence. Includes the silent-fallback guard and why the
   numbers would otherwise be meaningless -- all three optional backends were
   absent from the cluster image.

7. **Measurement discipline.** Variance controls, the CV ceiling, and the
   finding that exclusive-node allocation did *not* collapse the spread.

8. **Limitations.** One architecture for end-to-end results; no tier-2
   promotion yet; catch-rate has no denominator. Stated as limitations, not
   omitted.

## Rule

Every number in the prose cites a row in `EVIDENCE_MAP.md`. A section that
cannot cite is a section that does not ship.
