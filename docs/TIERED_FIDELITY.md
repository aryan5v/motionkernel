# Tiered fidelity contracts

## Why a second axis

`parity.policy` answers "how closely must these tensors match?". For a fused
LayerNorm meant as a drop-in replacement the only honest answer is *exactly*,
and `byte_equal` is the right contract.

It is the wrong contract for the two families this project is moving toward. A
different attention backend changes the accumulation order and numeric format
of the attention product. A cross-step cache *skips recomputing* a step and
reuses a previous result. Neither is bit-exact, and neither can be made
bit-exact. Under `byte_equal` both are rejected before they are ever
benchmarked.

The tempting fix — widen the tolerance until they fit — is exactly the failure
this repository already paid for. In the R4 LTX run four VAE artifacts built on
`rcp.approx.ftz.f32` / `tanh.approx.f32` reached packaging carrying up to
**131072.0** of absolute error at `match=true`, because a tolerance wide enough
to admit them was wide enough to admit anything. A tolerance wide enough to
admit a cache hit is wider still.

So the numeric axis is not widened. A second axis is added: a workload declares
which axis its outputs are contracted on.

## The tiers

| tier | name | contract | who lives here |
|---|---|---|---|
| 1 | `exact` | bitwise identity, decided by the parity policy | fused kernels, CUDA-graph replay |
| 2 | `perceptual` | rendered frames perceptually indistinguishable at a fixed seed | attention backends, schedule transforms |
| 3 | `advisory` | quality measured and recorded, **never** gating | discovery campaigns |

A workload that declares nothing gets tier 1. Failing closed is the point: a
workload that never considered the question gets the strictest contract, not
the most convenient one.

Tier 3 never auto-promotes. It exists so an aggressive transform can leave
evidence behind, not so a promotion can be waved through; a pipeline able to
promote from it would make the distinction from tier 2 decorative.

## Declaring a budget

```yaml
parity:
  policy: byte_equal      # still describes the numeric axis

fidelity:
  tier: perceptual
  min_ssim: 0.98
  max_lpips: 0.02
  frame_set: wan-1.3b-fixed-seed-8
  seed: 1234
```

Thresholds are declared **per workload and never invented by the harness** —
the same rule `resolve_leaf_tolerance` enforces on tolerances, for the same
reason. A 1.3B model at 480p and a 14B model at 720p do not share a perceptual
noise floor, and a default that fits neither is a fail-open hole wearing a
number.

Two rules follow from that and are enforced at construction:

- a `perceptual` budget must declare at least one threshold and name its frame
  set — a tier with no bar is not a weaker contract, it is *no* contract;
- an `exact` budget must not declare perceptual thresholds — a perceptual bar
  under an exact tier could only ever weaken it.

## How the gate reads it

`GenerationOutcome.decide()` is the single promotion gate. It consults the
budget rather than `parity_passed` directly:

- **tier 1** — the two are the same question; behaviour is unchanged, down to
  the wording of the quarantine reason.
- **tier 2** — the fidelity verdict *replaces* the parity check. A tier-2
  candidate fails `byte_equal` by construction, so gating on parity first would
  quarantine every attention and caching candidate before its evidence was ever
  read.
- **tier 3** — never promotes.

Missing evidence holds the artifact. Evidence from a different frame set or
seed is *refused*, not discounted — it is evidence about a different question.
A metric with no backend installed is reported absent, and a budget gating on
it holds; substituting a default would rebuild the R4 hole with a different
number in it.

## What lands in the manifest

Above tier 1 the promotion record carries the budget, the verdict, and the
**signed margin** for every gated metric, on failures and passes alike:

```json
"fidelity": {
  "budget":  {"tier": "perceptual", "tier_number": 2, "min_ssim": 0.98, ...},
  "verdict": {"passed": true, "margins": [
      {"metric": "ssim",  "value": 0.9912, "threshold": 0.98, "margin":  0.0112, "passed": true},
      {"metric": "lpips", "value": 0.0071, "threshold": 0.02, "margin":  0.0129, "passed": true}]},
  "evidence": {"frame_set": "wan-1.3b-fixed-seed-8", "seed": 1234, "frames_compared": 8}
}
```

Margins are signed so **positive always means passing**, whichever direction
the metric runs (SSIM upward, LPIPS downward). A promotion that cleared its
floor by 0.0004 is a fact worth having; "passed" would not even be true for an
artifact that is not bit-exact.

## The harness

`autokernel.verification.perceptual` measures; `autokernel.verification.fidelity`
contracts and gates. The split keeps the gate testable without a GPU, an image
backend, or a pretrained network.

- **SSIM** is computed directly (numpy only, Wang et al. 2004: 11×11 Gaussian
  window, σ=1.5) so tier 2 has no optional dependency in its common path. It is
  cross-checked against scikit-image to ~1e-16 in the test suite.
- **LPIPS** needs a pretrained network and is loaded through a pluggable
  backend; absent when not installed.
- **VBench** runs in an isolated stage and its score is passed *in*, so nothing
  on the promotion path imports its model zoo.

Aggregation is **worst-frame, not mean**. A cache that is perfect on seven
frames and destroys the eighth averages to a comfortable pass; video artifacts
are temporally local, and a single broken frame is precisely what a schedule
transform produces. Frame-count, shape, and seed mismatches are errors, not low
scores — that is a broken run, and recording it as a score would be a lie about
what was measured.
