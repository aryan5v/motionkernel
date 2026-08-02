# Variance controls: what they fixed, and what they did not

## Why

Native LTX arms on sm100 span 3.12 s to 4.01 s -- about 25%. A 1.01x gate
cannot survive that, and neither can a 1.3x one. Before these controls existed,
every dispatch A/B published a point estimate drawn from that distribution
without saying so.

## What the schema adds (measurement schema v1 -> v2)

Every measurement record now carries:

- `node_exclusive` -- whether the timed arms had the node to themselves;
- `gpu_clocks` -- current and maximum SM/memory clocks, and whether a lock was
  applied;
- `variance.native_cv` / `variance.candidate_cv` -- coefficient of variation
  per arm;
- `variance.cv_ceiling` -- declared ceiling, currently **0.03**;
- `variance.valid_for_gating` -- false when the native arm exceeds the ceiling,
  with the reason recorded.

A measurement over the ceiling is **recorded but unusable as promotion
evidence**. It is not deleted and the gate is not widened to admit it.

## Before and after

Same workload (`ltx-t2v-480p`), same artifact, same node class, 15 timed runs
per arm.

| job | schema | controls | native CV | candidate CV | valid for gating | reported speedup |
|---|---|---|---|---|---|---|
| 1030 | v1 | none | 4.74% | 1.69% | (no such field) | 1.0274x |
| 1040 | v1 | none | 7.37% | 1.03% | (no such field) | 1.2459x |
| 1055 | **v2** | exclusive node; clock lock unavailable | 5.57% | 1.58% | **false** | 1.1155x |

**The spread did not collapse.** Exclusive node allocation alone did not bring
the native arm under 3%.

**What did change is that the measurement now says so.** Run 1055 is marked
invalid for gating and cannot be used as promotion evidence. Runs 1030 and 1040
carried no such field and were publishable.

Look at the speedup column: **1.0274x, 1.2459x, 1.1155x** for the same
measurement -- a 21% swing. Under v1 any of the three could have been quoted as
"the" dispatch speedup, and the choice between them would have been made by
whichever ran last. That is the failure the CV gate exists to prevent, and it
is prevented whether or not the underlying variance is fixed.

## Why the spread survived: clock locking is unavailable

The launch path requests a clock lock and records the outcome. On this cluster
it fails:

```
clock lock unavailable; running with default clock state (recorded)
```

The recorded clock state shows why the residual variance is what it is: SM
clock **285 MHz against a 2062 MHz maximum** at query time -- the GPU is idling
between arms and boosting back up during them, so each timed run starts from a
different point on the ramp.

`nvidia-smi -lgc` / `-lmc` require privileges the Pyxis container does not
have. This is an infrastructure limitation, not a measurement bug, and the
honest consequence is:

> **LTX dispatch A/Bs on this cluster are currently ungateable.** They can be
> recorded, but not used as promotion evidence, until clocks can be pinned or
> the run design absorbs the ramp (long warmups, interleaved arms).

## What this means for measurements already published

The ceiling applies retroactively, so it is worth stating which existing
evidence survives it. The attention campaign arms
(`docs/ATTENTION_CAMPAIGN_RESULTS.md`) are far inside the ceiling:

| campaign | native median | native stdev | native CV | within 3% ceiling? |
|---|---|---|---|---|
| SAGE_ATTN A/B (SLURM 1041) | 15.1683 s | 0.0054 | **0.04%** | yes |
| VSA A/B (SLURM 1046) | 15.2035 s | 0.0704 | **0.46%** | yes |

Wan at 480p/20 steps is two orders of magnitude more stable than LTX on the
same cluster, so the attention rejections stand under the new schema. That is a
property of the workload -- longer generations, less idle time between runs --
not of the controls.

## Next steps that would actually collapse the spread

1. A privileged clock lock (needs cluster-side capability, not a code change).
2. Interleaving arms run-by-run instead of arm-by-arm, so clock drift affects
   both arms equally rather than accumulating in whichever ran second.
3. Longer warmups to reach a steady clock state before timing starts.

None of these are done. The ceiling stands at 3% until one of them is.
