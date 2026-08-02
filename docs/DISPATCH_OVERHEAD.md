# Dispatch overhead, measured and published

**Headline: the per-call dispatch tax is gone.** Measured the same way that
produced the stale 3.104 ms figure, the CUDA-graph dispatch path's overhead
is **−0.096 ms/call** (net cost −0.220 ms/call plus the artifact's 0.124
ms/call isolated kernel saving) on the LTX transformer stack at 384
calls/generation, sm100. Negative means the dispatched path is *faster* than
the native module forward it replaced: replaying the block from a captured
CUDA graph removes the whole block's host-side dispatch cost, not just the
artifact region's. The 3.104 ms number described the eager FX replay era and
no longer describes any code that runs.

| | eager FX replay (R4, 2026-08-01) | CUDA-graph dispatch (2026-08-02) |
|---|---|---|
| dispatch overhead / call | **+3.104 ms** | **−0.096 ms** |
| net cost / call (e2e) | +2.980 ms | −0.220 ms |
| e2e A/B (15 runs/arm) | 0.7629× | 1.0274× median, 1.0148× min-to-min |
| parity | byte_equal | byte_equal |
| candidate calls / fallbacks | 2303 / 0 | 6143 / 0 |

The constraint this tax imposed is gone with it. What limits subgraph
artifacts now is stated honestly below: parity, the decomposed export
graph's device-side cost, and call volume versus the gate — not dispatch.

## Method (identical to the one that produced 3.104 ms)

Two measurements, both from `python -m autokernel.dispatch measure`
(SLURM job 1030; harness in `autokernel/dispatch/`, runner script in
`benchmarks/dispatch_overhead_ltx_sm100.sbatch`):

1. **End-to-end A/B**, the arithmetic of R4 §7: native vs candidate, 15
   timed runs per arm, same node, same session, frames compared under the
   workload's `byte_equal` policy. Overhead = (candidate − native) /
   calls-per-generation + kernel saving. R4: (4.8226 − 3.6789) / 384 +
   0.124 = +3.104 ms. Now: (3.0835 − 3.1679) / 383.94 + 0.124 = **−0.096
   ms**.
2. **In-situ profiles** of the dispatched path
   (`FASTVIDEO_OPTIMIZATION_ARTIFACT_TIMING`): a shadow profile against the
   native forward on identical inputs, and an unsynced host profile
   measuring host-side cost only.

Artifact `mk-2c92e356aa34bc0d-7df21b47-sm100`, workload `ltx_480p.yaml`
(LTX2-Distilled, 480×768, 97 frames, 8 steps), FastVideo `7299cc9a`.

### sm100 run (SLURM 1030, 2026-08-02)

| arm | median | stdev | min | runs |
|---|---|---|---|---|
| native | 3.1679 s | 0.1502 | 3.0688 | 15 |
| candidate | 3.0835 s | 0.0522 | 3.0240 | 15 |

1.0274× median, 1.0148× min-to-min, byte_equal, 6143 candidate calls
(383.94/generation), 0 runtime fallbacks. The candidate arm is again more
reproducible than the baseline (stdev 0.052 vs 0.150) — a consequence of
replaying a fixed graph, as in R4 §9.

**Noise discipline.** The 84 ms median separation is real but small against
the baseline's spread; min-to-min separates by 45 ms. We report the point
estimate with the spread, not a single heroic digit: overhead ≤ 0 within
noise, and non-positive in every reading we have taken. The gate-5
confirmations (1.0857×–1.2514× across SLURM 994–999) measure the same
absence of a tax on other days and nodes.

## Where the time goes now

The shadow profile (synchronized, 1151 attributed calls) records:

| phase | mean/call | reading |
|---|---|---|
| `shadow.native_forward` | 8.84 ms | the block forward that was replaced |
| `dispatch.candidate_total` | 28.26 ms | see the caveat below before comparing |
| `subgraph.flatten` + `validate` + `unflatten` | 0.17 ms | plumbing |
| `dispatch.shape_key` | 0.11 ms | per-call signature |
| `subgraph.execute` (eager) | 12.64 ms ×163 calls | one scope's capture declined under shadow |
| `subgraph.execute_cuda_graph` | 19.84 ms ×1133 calls | sync-serialized, see caveat |

**The caveat that keeps this honest.** The shadow profile synchronizes
around every phase, serializing host and device time. That was the right
way to attribute the eager FX replay's tax, because eager replay *is* host
work: 621 `call_function` nodes of Python and dispatcher cost, identical
device kernels to the native forward, so the difference (11.57 vs 8.18 ms)
was host cost. A CUDA-graph replay is the opposite shape: ~10 µs of host
launch plus device time the synchronization then waits for. The 19.84 ms
figure is that serialized device time, inflated well past what real
unsynced runs pay — the candidate arm's real per-call amortized cost is
~5.5 ms (3.08 s/gen over 384 calls, minus non-block work), not 19.84 ms.
Comparing 28.26 against 8.84 ms as if it were the R4 reading would
"discover" a +19 ms overhead that the end-to-end A/B directly contradicts
(candidate is *faster*). We report the profile for structure — what runs
where, what declined — not for that difference.

The unsynced **host profile** measures what actually remains of the tax:
host-side cost per candidate call. (Filled in from SLURM 1040.)

One scope's capture declined under the shadow profile (`runtime input 1
moved or changed layout after capture`): the shadow forward's extra
allocations move addresses, which the pinning checks correctly refuse. In
the unshadowed arms all 48 scopes captured and stayed captured (0
fallbacks over 6143 calls).

## The break-even curve

Required per-call kernel saving to clear the 1.01× gate, from the measured
native e2e (3.1679 s) and overhead (−0.096 ms/call), LTX 480p:

| calls/generation | required saving/call |
|---|---|
| 1 | 31.27 ms |
| 5 | 6.18 ms |
| 10 | 3.04 ms |
| 25 | 1.16 ms |
| 50 | 531 µs |
| 100 | 218 µs |
| 200 | 61 µs |
| 384 | **−14 µs** |
| 500 | −33 µs |
| 1000 | −64 µs |

How to read this:

- **Negative required saving** means the gate clears on the framework's
  host-side win alone at that call volume — a zero-saving artifact that
  holds byte_equal would pass. That is exactly what gate 5 showed
  (1.0857×–1.2514× with a 124 µs kernel). High-frequency regions are now
  worth searching on parity grounds alone.
- **Low-volume regions (VAE-scale, ~5 calls/gen) still need real kernels:**
  ~6 ms/call to clear the gate. The framework's per-call win does not
  amortize there, and the gate term dominates. The quarantined VAE
  artifacts' ~1.11× kernels (~120 µs/call) remain 50× short — the R4 §4
  verdict stands, unchanged by the dispatch fix.
- The old curve's reading ("at 384 calls, a 124 µs kernel is 25× short")
  inverts: at 384 calls the framework's win *exceeds* the gate's demand.

**Honest framing, preserved from R4 §8.** The artifact's kernel saves 124
µs per call, which alone caps end-to-end at ~1.015×. Measured gains are
larger because CUDA-graphing the block also removes the whole block's
host-side dispatch cost. That acceleration exists only on the artifact
path — without an artifact the block is never dispatched and never
captured — so the A/B is sound, but **the framework contributes more of
the gain than the kernel does**. Any support claim derived from these
numbers must say so in the same terms.

## What the constraint is now

1. **Parity.** `byte_equal` is the binding constraint: the quarantined VAE
   artifacts fail it deterministically (`rcp.approx.ftz`, `tanh.approx`),
   and no dispatch improvement changes that.
2. **The decomposed export graph's device cost.** The export graph
   materializes slices, casts, and small ops the native module fuses. The
   graph replay removes all host cost but keeps that device work; closing
   it is a graph-construction problem, not a dispatch problem.
3. **Call volume vs the gate** for low-frequency regions (above).

## Architectures

| arch | GPU | status |
|---|---|---|
| sm100 | GB200 (SLURM) | **measured** — this page, SLURM 1030/1040 |
| sm90 | H100 (Modal) | pending — sm90 bundle packages from the same candidate with real H100 isolated evidence (`modal/dispatch_overhead_h100.py`) |

## Recomputing the number

The number goes stale the day the dispatch path changes. The harness is the
antidote:

```bash
# SLURM (sm100): submits the same measurement as SLURM 1030
sbatch --export=NIL benchmarks/dispatch_overhead_ltx_sm100.sbatch

# Any fastvideo timing report, ad hoc:
python -m autokernel.dispatch analyze timing.json
python -m autokernel.dispatch breakeven --native-e2e-s 3.1679 \
    --overhead-ms -0.096 --calls-per-generation 384
```

Each run writes a fresh immutable evidence directory
(`/mnt/nfs/vlm-aryan/mk-track-d-dispatch-runs/<arch>-<date>-<jobid>`) with
the per-arm results, profiles, diagnostics, and one `measurement.json`
record. History:

| date | arch | overhead/call | evidence |
|---|---|---|---|
| 2026-08-01 | sm100 | +3.104 ms (eager FX replay) | `docs/LTX_V1_R4_ROOT_CAUSE.md` §7 |
| 2026-08-02 | sm100 | **−0.096 ms** (CUDA graph) | SLURM 1030, `mk-track-d-dispatch-runs/sm100-20260802-181929-1030` |
