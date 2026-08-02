# Track D — dispatch overhead: measure it, publish it, defend it

**Read this section before anything else. The original roadmap for this track
is based on two premises that are false, and following it verbatim wastes
days.**

---

## 0. What the roadmap said, and what is actually true

The roadmap said: *"Replace FX-replay delivery with torch.library custom-op
registration so torch.compile traces through and guards stay intact; target
<100µs/call overhead... Re-dispatch the existing promoted LTX2 artifact and the
four quarantined ones as the regression suite — some of those 'could never
pass' verdicts flip if the 25:1 tax disappears."*

Both halves are already settled, in `docs/LTX_V1_R4_ROOT_CAUSE.md` §7–9:

**The FX replay is already gone.** It was replaced with
`torch.cuda.CUDAGraph` capture, not `torch.library` custom ops. The choice was
contractual rather than performance-driven: a CUDA graph replays the same
kernels with the same parameters in the same order, so it is bitwise identical
*by construction*, whereas a compiler backend's fusion or reassociation would
put `byte_equal` at risk. Gate 5 now **passes at 1.2514× median** (15 runs/arm,
SLURM 999), 6143 calls, 0 fallbacks, byte-equal parity, +4.62% peak memory
against a 5% limit.

**The four quarantined artifacts will not flip.** §4 does the arithmetic:

| artifact | share of e2e | speedup | realized gain |
|---|---|---|---|
| `mk-bbfe15180d31bf50` | 2.334% | 1.1154× | 0.241% |
| `mk-a81e140d62ff170c` | 1.863% | 1.1104× | 0.185% |
| `mk-baecc3825d4a8c18` | 1.081% | 1.1212× | 0.117% |
| `mk-b6cb64f99049683b` | 0.922% | 1.1064× | 0.089% |
| **total** | **6.20%** | | **0.632% → 1.0064×** |

Target is 1.01×. **1.0064× is the ceiling at zero dispatch overhead.** They
additionally break `byte_equal` deterministically (`rcp.approx.ftz.f32`,
`tanh.approx.f32`). Both disqualifiers are independent of dispatch cost. Do not
spend GPU time re-dispatching them expecting a different verdict.

Also relevant: the per-artifact latency differences that made
`mk-b6cb64f99049683b` look like the offender were **node contention**. At
`runs: 5` it reversed from 0.8425× to 1.0912× — a 29.5% swing — with a native
baseline spanning 38.4% within one trial.

---

## 1. Re-scoped goal

The interesting question is no longer "how do we remove the tax" but **"what is
the tax now, measured the same way, and does it hold across shapes,
architectures and call volumes?"** The 3.1 ms figure is the most-cited number
in this project and it is now stale. Replace it with a current, defensible one.

Deliverables:

1. **A published per-call overhead number** for the CUDA-graph dispatch path,
   measured by the same method that produced 3.104 ms, so the two are
   comparable. The method is in §7 of the R4 document: profile in situ,
   attribute `shadow.native_forward` vs `subgraph.execute` vs
   flatten/validate/unflatten/shape_key.
2. **A regression harness** that recomputes it on demand, so the number does
   not go stale again.
3. **The break-even curve**: per-call saving required to clear the gate, as a
   function of call volume. The old answer was "3.1 ms, so a 124 µs kernel at
   384 calls/generation is 25× short". The new answer determines which regions
   are worth searching at all — this is the output other tracks actually need.
4. **A second architecture.** Everything to date is sm100 (GB200). Add sm90
   (H100, available on Modal) and report whether the overhead differs.

## 2. Honest framing you must preserve

§8 of the R4 document contains this, and any write-up you produce must not
quietly drop it:

> the artifact's kernel saves 124 us per call, which alone caps end-to-end at
> ~1.015x. The measured 1.086x is larger because CUDA-graphing the block also
> removes the *whole block's* host-side dispatch cost. That acceleration exists
> only on the artifact path — without an artifact the block is never dispatched
> and never captured — so the A/B is sound, but the framework contributes more
> of the gain than the kernel does.

If your measurements reproduce that, say so in the same terms. A support matrix
that implies the kernels are doing the work when the framework is would be a
credibility problem the first time someone reproduces it.

## 3. Where the code is

FastVideo `fastvideo/optimization/` holds the dispatch path. The CUDA-graph
capture, the pinning checks, and the bitwise-verification step are described in
R4 §7 — read it before touching anything. Note in particular that after capture
the runner replays once, runs the eager graph, and **refuses the capture unless
every output is bitwise equal**. That check is load-bearing; do not weaken it
for speed.

Known-subtle areas, all previously bugs, all currently fixed — do not
regress them:

- captured parameter addresses are re-checked per replay (FSDP2 `reshard` frees
  the all-gathered storage a capture recorded pointers into);
- pinning is on `(data_ptr, shape, stride, dtype, device, storage_offset)`, not
  `data_ptr` alone (a permuted view reusing the same allocator block passed
  every check while the captured kernels read elements in a different order);
- each capture gets its own `graph_pool_handle` (sharing one across 48 blocks
  trips the allocator's `use_count > 0` assert);
- non-tensor outputs are cloned unless `None` or scalar;
- warmup is not detected by comparing an exception message string.

## 4. GPU access

**Never run GPU workloads on the login node.**

### SLURM — GB200, sm100, torch 2.8.0a0

The SLURM login host, account name and SSH key path are **not recorded in this
repository** — it is public. Get them from the project operator, along with the
key itself, and connect with:

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes \
    -i <path-to-key> <account>@<login-host>
```

Never display, upload, commit or copy the private key.

Partition `all`, submit with `sbatch --export=NIL`, NFS root
`<nfs-root>`. Two gotchas:

- **No `/home/<account>`** — set `#SBATCH --chdir=<nfs-root>`.
- **Bare `srun` has no CUDA tooling**, not even `nvidia-smi`. Use Pyxis:

```bash
srun --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
     --container-mounts=<nfs-root>:<nfs-root> \
     --container-workdir=<nfs-root>/<your-dir> \
     bash -lc "<command>"
```

`Reason=Resources` / `StartTime=Unknown` is not proof of no capacity; jobs have
started within a minute of showing it. Check `sacct -j <id>`.

Prior evidence lives in `<nfs-root>/ltx-v1-*` — treat it as immutable.

### Modal — H100, sm90

`modal` CLI, workspace `hao-ai-lab`. This is your second architecture. See
`modal/verify_tiered_fidelity.py` on branch `tiered-fidelity-contracts` for a
working app.

## 5. Measurement discipline

- **≥15 timed runs per arm.** R4's `runs: 2` and `runs: 5` measurements both
  produced conclusions that later reversed.
- Report **median, stdev, and min-to-min**. The candidate arm being *more*
  reproducible than the baseline (stdev 0.0205 vs 0.1555) is itself a finding —
  it is a consequence of replaying a fixed graph.
- A/B on the same node in the same session.
- If a difference is inside the noise band, say so. Do not report a point
  estimate you cannot defend.

## 6. Branching and deliverable

Branch from `main`, name `dispatch-overhead-published`. No prefixes.

Draft PR against `aryan5v/motionkernel` (pass `--repo aryan5v/motionkernel` —
`gh` otherwise resolves to the upstream fork parent and fails with a misleading
"no commits between" error).

The headline deliverable is a **number with a method attached**, plus the
break-even curve. If the honest finding is "overhead is now negligible and the
constraint has moved elsewhere", that is a completely acceptable result — say
it plainly and identify what the new constraint is.
