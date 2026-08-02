# Track C — schedule transforms: input-similarity caching

Goal: add a **schedule transform** artifact kind that wraps the *denoising
loop* rather than a region, and land TeaCache-style input-similarity caching as
its first family, promoted per-workload under Tier 2.

This is the cheapest 1.5–2.5× available and it proves the artifact model
generalizes beyond kernels. Every artifact this repository has promoted so far
replaces a *region*; none has ever wrapped the loop.

---

## 1. Repository and environment

**MotionKernel** (the optimizer): `https://github.com/aryan5v/motionkernel`
**FastVideo** (the runtime): `https://github.com/aryan5v/FastVideo`

Local checkouts live under `~/Fast video1/` (many sibling worktrees; do not
disturb them — make your own).

> **`gh` gotcha:** `gh repo view` in these checkouts resolves to the *upstream
> fork parent* `RightNow-AI/autokernel`, so `gh pr create` fails with a
> misleading `No commits between main and <branch>`. Always pass
> `--repo aryan5v/motionkernel`.

### Branching

Branch from **`attention-artifact-kind`** (Track A), not from `main` — see §2.
Branch name: `schedule-transform-caching`. No prefixes.

```bash
git fetch origin attention-artifact-kind
git worktree add -b schedule-transform-caching ~/mk-caching origin/attention-artifact-kind
```

If `attention-artifact-kind` does not exist on the remote yet, start from
`tiered-fidelity-contracts` (you need the fidelity tiers regardless) and do the
cache implementation first, wiring the artifact kind last.

---

## 2. The collision you must avoid

`autokernel/artifact/types.py` (~line 548) has:

```python
if target_kind not in {"module", "subgraph"}:
```

Track A is extending this to admit an `attention` kind and is landing a
*mechanism* for registering kinds. You need a `schedule_transform` kind.

**Do not edit that literal set.** Register your kind through the mechanism
Track A lands. If you race ahead and edit the set directly you will produce a
conflict that is genuinely painful to resolve, because the validation rules
attached to each kind differ (a `subgraph` target carries rewrite fields that a
`schedule_transform` must not).

---

## 3. What a schedule transform is, contractually

A region artifact answers "replace these ops with this kernel". A schedule
transform answers "wrap the denoising loop and decide, per step, whether to
recompute".

Consequences you must handle:

- **It has no graph fingerprint.** The existing dispatch matches artifacts by
  `graph_fingerprint` over a captured region. A loop wrapper has no such
  region. Your kind needs its own identity and its own compatibility check
  (model, scheduler, step count, resolution).
- **Its output is not comparable per-call.** There is no per-call reference to
  diff against, because the whole point is that a call *did not happen*. The
  only meaningful comparison is the final decoded frames. This is precisely
  what Tier 2 exists for.
- **It changes step count, not step cost.** Amdahl reasoning over
  `share_of_e2e` does not apply. Do not reuse `estimated_max_e2e_improvement`;
  §4 of `docs/LTX_V1_R4_ROOT_CAUSE.md` records that metric overstating return
  by more than 8× and two artifacts carrying `meets_promotion_target: false`
  into packaging anyway.

---

## 4. The searched parameter

TeaCache caches the transformer output and skips recomputation when the
*relative L1 distance* between the current and previous modulated input falls
below a threshold. The threshold is the searched parameter.

Requirements:

- The threshold must be **declared in the artifact manifest**, not baked into
  the payload as a constant. §1a of the R4 document is the cautionary tale: a
  candidate compiled `_MAIN_ROWS = 4680` and `SOURCE_ROW_STRIDE=16384` in as
  `constexpr` — right for one call, wrong as a property of the operation, and
  it took a full campaign to find out.
- Cache state must be **reset between generations**. A cache that survives into
  the next generation produces a first frame contaminated by the previous
  prompt. Assert this in a test with two different prompts back to back.
- **The first step can never be a hit.** There is no previous input. Make this
  structural, not a threshold accident.
- Skipping must be **counted and reported** (steps taken vs steps skipped). A
  transform that never fires and a transform that fires every step both look
  like "it ran"; the hit rate is what distinguishes a 1.8× from a no-op.

---

## 5. Fidelity tier — you are gated at Tier 2

Tiered fidelity contracts landed in PR #18. Read `docs/TIERED_FIDELITY.md`
before writing the gate; the summary:

- **Tier 1 `exact`** — bitwise. A cache can never pass this. Do not try.
- **Tier 2 `perceptual`** — SSIM/LPIPS/VBench against a fixed-seed frame set,
  thresholds declared per workload. **This is you.**
- **Tier 3 `advisory`** — recorded, never auto-promotes.

Your workload manifests must declare:

```yaml
fidelity:
  tier: perceptual
  min_ssim: 0.98        # pick per workload; do not copy blindly
  max_lpips: 0.02
  frame_set: <name>
  seed: <int>
```

The gate is `GenerationOutcome.decide()` in `autokernel/artifact/finalizer.py`
and it already handles Tier 2 — at that tier the fidelity verdict *replaces*
the parity check, so your cache failing `byte_equal` is expected and will not
quarantine it. You should not need to modify the gate. If you think you do,
that is a signal something is wrong with the design; say so rather than
loosening it.

Aggregation is **worst-frame, not mean** (`autokernel/verification/perceptual.py`).
This matters more for you than for anyone: a cache that is perfect on seven
frames and destroys the eighth is the characteristic caching failure, and a
mean would hide it.

---

## 6. GPU access

**Never run GPU workloads on the login node.** Two paths:

### SLURM — GB200, sm100, torch 2.8.0a0

The SLURM login host, account name and SSH key path are **not recorded in this
repository** — it is public. Get them from the project operator, along with the
key itself, and connect with:

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes \
    -i <path-to-key> <account>@<login-host>
```

Never display, upload, commit or copy the private key.

Partition `all`. Submit with `sbatch --export=NIL`. Working NFS root is
`<nfs-root>`. Two gotchas that will cost you a job each:

- **There is no `/home/<account>`** — set `#SBATCH --chdir=<nfs-root>`
  or the job errors `couldn't chdir ... going to /tmp instead`.
- **Bare `srun` has no CUDA tooling**, not even `nvidia-smi`. Everything runs
  under Pyxis:

```bash
srun --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
     --container-mounts=<nfs-root>:<nfs-root> \
     --container-workdir=<nfs-root>/<your-dir> \
     bash -lc "<command>"
```

A job pending on `Reason=Resources` with `StartTime=Unknown` is **not** proof
the cluster is full — jobs have scheduled within a minute of showing that.
Check `sacct -j <id>` before concluding there is no capacity.

### Modal — H100, sm90

`modal` CLI, workspace `hao-ai-lab` (`modal profile list` to confirm it is
active). Faster iteration, but **sm90 only** — it cannot validate sm100
behavior. See `modal/verify_tiered_fidelity.py` for a working app, including
a `baseline_suite` pattern that runs the same suite at the merge-base.

**Use both:** Modal for the fast loop, SLURM/sm100 for anything you will claim.

---

## 7. Measurement discipline

This is where the project has been burned repeatedly. Non-negotiable:

- **Minimum 15 timed runs per arm.** R4's `runs: 2` measurement produced a
  0.8327× that later reversed; one artifact swung **29.5%** between runs, and
  a native arm showed a **38.4% spread** within a single trial on a shared
  node. 5 runs is not enough. Report median *and* stdev *and* min-to-min.
- **A/B on the same node, same session.** Cross-node comparisons on this
  cluster are noise.
- **Report the hit rate alongside the speedup.** A 2× with a 90% skip rate and
  a 2× with a 10% skip rate are different claims about the same number.
- If your measurement is inside the noise band, **say so** rather than
  reporting the point estimate. §3 of the R4 document is an example of doing
  this correctly.

---

## 8. Exit criteria

A promoted caching artifact on **one Wan and one LTX workload**, each with:

1. `promotion: promoted` written by the real finalizer, not by hand;
2. Tier-2 perceptual evidence attached, with **signed margins** recorded in the
   manifest (positive = passing);
3. end-to-end speedup measured at ≥15 runs/arm with median, stdev and min-to-min
   reported;
4. cache hit rate reported;
5. a test proving cache state does not leak between generations;
6. a test proving an intentionally over-aggressive threshold is **rejected** at
   Tier 2 — the control. Mirror
   `tests/test_fidelity_tiers.py::test_intentionally_lossy_control_is_rejected_at_tier_2`.

## 9. Deliverable

Draft PR against `aryan5v/motionkernel`:

```bash
gh pr create --repo aryan5v/motionkernel --draft \
  --base main --head schedule-transform-caching --title "..." --body-file <file>
```

State plainly in the PR what you did **not** verify. A gap that is named is
useful; a gap that is implied to be covered is how this project lost a
campaign.
