# Paired protocol: first results

Schema v3, ABBA-interleaved, sustained-load warmup, GB200/sm100.

## The warmup works

Both runs reached a clock plateau and held it:

| job | warmup | clock during timing |
|---|---|---|
| 1077 | plateaued after 77.3 s | **2062 MHz sustained** (max) |
| 1078 | plateaued after 93.0 s | **2062 MHz sustained** (max) |

This is the control v2 could not achieve. Clock *locking* is still unavailable,
but driving sustained load until the boost ramp flattens reaches the same
steady state, and the recorded trace lets a reader verify it rather than take
it on trust.

## Gap 5 — Wan attention A/B, re-measured paired

| protocol | speedup | valid for gating |
|---|---|---|
| sequential (SLURM 1041) | 0.8031x | yes (native CV 0.04%) |
| **paired (SLURM 1077)** | **0.8106x** | yes |

Paired 95% CI [0.8091, 0.8111], Wilcoxon p = 0.0143, conclusive.

The two protocols agree to within 1%. That is the expected outcome on a
workload whose native arm was already at 0.04% CV: where there is no drift to
cancel, pairing changes nothing. **The SAGE_ATTN rejection stands, now under
both protocols.**

## Gap 7 — V1 LTX re-validation: NOT MEASURED

SLURM 1078 returned a paired speedup of 0.9011x with a tight CI [0.8680,
0.9196] that excludes 1.0. Read naively that is a retraction of the V1 headline
(1.0857x median, 1.2514x replication).

**It is not a retraction. It is not a measurement of the artifact at all.**

The candidate arm set `FASTVIDEO_OPTIMIZATION_ARTIFACT_DIR`,
`..._ARTIFACT_MODEL_ID` and `..._ARTIFACT_VALIDATION`. The FastVideo checkout
used (`fastvideo-wan-fusions`) contains only `optimization/artifacts.py` and
`optimization/capture.py`, and the single environment variable it reads is
`FASTVIDEO_OPTIMIZATION_CAPTURE`. The artifact-dispatch path lives on a
different branch. Every one of those variables was inert, and the run log
contains no artifact selection, no candidate calls and no dispatch
diagnostics.

Both arms ran native. The 0.9011x is the two-resident-generator asymmetry of
the harness, not an effect.

### Why this nearly became a published retraction

The number had every surface property of a good result: a tight confidence
interval, a significant p-value, `conclusive: true`, `valid_for_gating: true`,
and a clean plateaued clock trace. Every check the protocol had *did* pass.
None of them asked whether the intervention under test actually happened.

That is the identical failure mode this project already guards against one
level up: FastVideo silently substitutes FlashAttention when an optional
backend cannot be imported, so an attention campaign records the fallback under
the requested backend's name. The attention artifact contract closes it by
recording the *effective* backend and refusing when it differs. The paired
protocol had no equivalent check.

It does now: `summarize_paired(..., arms_differentiated=False)` marks a
measurement invalid with the reason recorded. A candidate arm whose
configuration did not take effect measures something else, however tight its
interval.

### Gap 7 remains open

To close it the re-validation must run against a FastVideo checkout that
actually implements artifact dispatch (`agent/v1-r4-dispatch-fix`), with
`candidate_calls > 0` asserted before the timings are interpreted -- the same
gate 3 the original V1 proof used.

Until then the V1 headline stands as measured under the sequential protocol,
with the caveat already recorded in EVIDENCE_MAP: sequential measurement is
ungateable on this cluster, and three sequential runs of the same dispatch
measurement reported 1.0274x, 1.2459x and 1.1155x.

**Neither confirmed nor retracted. Not measured.**
