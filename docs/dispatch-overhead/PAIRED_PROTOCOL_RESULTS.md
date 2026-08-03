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

---

## Gap 7, resolved: the headline reproduces, the replication figure does not

Six attempts. Two were real discoveries about the system, four were harness
plumbing; both counts are recorded in the cycle summary because the ratio is
the useful part.

### What blocked it

| attempt | failure | what it would have reported |
|---|---|---|
| 1078 | wrong FastVideo checkout -- reads only `FASTVIDEO_OPTIMIZATION_CAPTURE`, every artifact variable inert | **0.9011x** -- "headline retracted" |
| 1081 | `ARTIFACT_ENABLE` is an artifact *selection filter*, not a boolean; `"1"` asked for an artifact named `1` and admitted 0 of 1 bundles | **1.0426x** -- "reproduces, inconclusive" |
| 1082 | dispatched correctly; harness read diagnostics before the worker flushed them | 1.1550x marked invalid -- a **false negative** |
| 1083-1084 | in-process retry cannot work (flush happens after `main()` returns); then a path mismatch | inconclusive |
| **1085** | -- | **verified** |

1078 and 1081 produced *opposite* wrong answers from the same non-event.
Whichever had been run once would have been believed.

1082 matters in the other direction: a check that discards a real result
teaches people to disable it. `valid_for_gating: false` has to be both rare and
right.

### The verified record

SLURM 1085, `agent/v1-r4-dispatch-fix @ 7299cc9a`, artifact
`mk-2c92e356aa34bc0d-7df21b47-sm100`:

```
candidate_calls  3071
runtime_fallbacks   0
differentiated   True
valid_for_gating True
```

### Four paired sessions

| job | paired speedup | 95% CI | dispatch |
|---|---|---|---|
| 1082 | 1.1550x | [1.123, 1.187] | verified (3071 calls) |
| 1083 | 1.0784x | [1.051, 1.103] | not captured |
| 1084 | 1.0981x | [0.995, 1.201] | not captured |
| **1085** | **1.0604x** | [0.995, 1.232] | **verified (3071 calls)** |

median **1.0882x**, mean 1.0980x, range 1.0604-1.1550 (**8.9% spread**),
stdev 0.0410.

### The verdict

**The V1 headline reproduces. The replication figure does not.**

The published median was **1.0857x**, which sits almost exactly on the paired
median of **1.0882x**. That is a reproduction, under a protocol designed to
break sequential measurements.

The published replication of **1.2514x** is **above every one of the four
paired sessions**. It should be read as a high outlier of the sequential
protocol -- the same protocol that produced 1.0274x, 1.2459x and 1.1155x for
one dispatch measurement -- not as a second independent confirmation.

Recommended statement: *the promoted V1 LTX artifact delivers approximately
1.09x end-to-end (four paired sessions, 1.06-1.16x), with the artifact verified
to dispatch on every timed run.* Quoting 1.2514x is not supportable.

### A limitation the data forced, not a bug

Sessions 1082 and 1083 have **non-overlapping** confidence intervals for the
same artifact on the same workload -- both paired, both ABBA, both on plateaued
2062 MHz clocks.

The bootstrap CI quantifies sampling error *within* one session and is silent
about variation *between* them, which is evidently larger. A single session's
interval therefore understates real uncertainty, and 1082's tight
[1.123, 1.187] is misleading read alone.

Pairing still helped: sequential spread was ~15% (1.0857 vs 1.2514), paired is
8.9%. It roughly halved the variance without removing it.

**The protocol needs n sessions, not just n pairs.** That is a design change,
not a fix, and it is recorded here rather than applied mid-cycle -- this cycle
has already demonstrated what editing the harness between runs costs.
