# MotionKernel + FastVideo LTX V1 — R4 root cause and targeted fix

Evidence base: `/mnt/nfs/vlm-aryan/ltx-v1-overnight-20260801-r4-sol` (SLURM 983,
immutable, unmodified). New evidence:
`/mnt/nfs/vlm-aryan/ltx-v1-r4-targeted-fix-20260801-203751` (SLURM 984, 985).

Branches: `aryan5v/motionkernel@agent/v1-r4-correctness-fix`,
`aryan5v/FastVideo@agent/v1-r4-dispatch-fix`. No PRs opened.

---

## 1. Transformer correctness failure

**Candidate** `2c92e356aa34bc0d3c49522bd1365c1b`, region
`transformer.model.transformer_blocks`, 22 selected nodes.

R4 recorded: smoke PASS (`max_abs_error=0.0`), shape sweep PASS
(`worst_err=0.00e+00`), determinism PASS (3 runs bitwise identical), and
numerical stability FAIL on all five cases with

```
AssertionError: expected size 4680==4680, stride 4096==16384 at dim=1
```

**This was never a numerical failure.** The kernel is bit-exact. It fails a
*layout contract*, for two compounding reasons.

### 1a. The input layout was compiled in as `constexpr`

```python
_MAIN_ROWS = 4680
_MAIN_WIDTH = 4096
...
SOURCE_ROW_STRIDE=16384,
CONTEXT_ROW_STRIDE=8192,
```

Those values are right for this workload — `input_2` is a 4096-wide slice of a
`1x4680x16384` timestep tensor, `input_7` a 2048-wide slice of `1x101x8192` —
but they are properties of *one call*, not of the operation. The
numerical-stability stage allocates fresh **contiguous** tensors of the same
shape (row stride 4096). The Triton kernel would have read at the wrong
addresses.

### 1b. The candidate bypassed Dynamo's guards

```python
_COMPILE_DISPATCH = torch.compile(_kernel_impl, backend=_capture_inductor_entry, ...)

def kernel_fn(...):
    if _COMPILED_ENTRY is None:
        return _COMPILE_DISPATCH(...)
    # "Calling the captured entry directly avoids a guard/cache
    #  lookup before every GPU schedule."
    outputs = _COMPILED_ENTRY(...)
```

After the first call, every invocation went straight to Inductor's raw compiled
entry — no shape, stride, or dtype guard. The violation was caught *only*
because Inductor happens to emit an internal stride assertion. The Triton
kernels have no such assertion; a slightly different violation would have
produced silently wrong output at full speed.

It also made the recorded **2.557×** an unfair comparison: the candidate skipped
guard evaluation the compiled baseline paid for.

### Repair and result (SLURM 984)

Shapes and strides are read from the tensors at run time; dispatch goes through
`torch.compile` with guards intact; intermediates are allocated contiguous
rather than with `empty_like` (which preserves a non-contiguous source layout
while the kernel indexes them contiguously). Arithmetic is untouched — no
approximate intrinsics — so bitwise parity remains reachable.

Measured under the workload's real `byte_equal` policy, tolerances **not**
loosened, verifier unchanged:

| candidate | correctness | stability | determinism | speedup |
|---|---|---|---|---|
| `transformer-orig` (control) | **FAIL** | FAIL (same stride assertion) | PASS | 2.395× |
| `transformer-fix` | **PASS** | **PASS** (`max_err=0.00e+00`) | PASS | **1.920×** |

The drop from 2.395× to 1.920× is the honest cost of restoring guards and
runtime strides.

---

## 2. VAE parity failure

All four packaged VAE artifacts use approximate hardware math:

```python
asm="rcp.approx.ftz.f32 $0, $1;"
asm="tanh.approx.f32 $0, $1;"
```

These cannot reproduce an eager reference bitwise, and `ftz` additionally
flushes denormals. The workload declares `parity.policy: byte_equal`. Full
generation parity failed exactly as it had to.

**Why validation accepted them:** the policy never reached the verifier.
`byte_equal` appeared nowhere in `autokernel/verification/` or `bench.py`; it
was consulted once, in `optimize/adapters.py`, to compare decoded frames. The
chain:

1. workload declares `byte_equal`;
2. generated manifests carry `tolerances: null`;
3. `_tolerance_for` falls back to `DEFAULT_TOLERANCE = 1e-2/1e-2`;
4. the stability stage multiplies that by `relax=10.0`;
5. against reference values near 3e5, an rtol of 1e-1 permits ~3e4 of error.

Recorded leaf metrics, all with `match=true, pct_within_tol=100`:

| artifact | stage | max_abs_error |
|---|---|---|
| `mk-bbfe15180d31bf50` | stability/mixed_scale | **131072.0** |
| `mk-baecc3825d4a8c18` | stability/mixed_scale | 65536.0 |
| `mk-a81e140d62ff170c` | stability/mixed_scale | 32768.0 |
| `mk-b6cb64f99049683b` | stability/mixed_scale | 32768.0 |

Even the realistic *smoke* case carried 0.25 error at
`pct_within_tol=99.99999%`, accepted as a match.

Three further comparator defects, independent of policy:

- `inf - inf` is NaN, so a candidate correctly reproducing the reference's
  infinities reported `max_abs_error=nan` while `allclose` returned `True`.
- A candidate NaN where the reference is finite passed under a large enough
  tolerance.
- Right values in the wrong layout satisfied `allclose`.

---

## 3. End-to-end regression

Native 3.2818s → optimized 3.9410s = **0.8327×**. Candidate runs were
[3.9301, 3.9519] — tight, so this is steady-state cost, not warm-up.

### Per-artifact isolation (SLURM 985, 986)

Running each artifact alone (native median 3.2207s, same workload, same node):

| artifact | dispatch | candidate | fallbacks | parity | median | Δ | speedup |
|---|---|---|---|---|---|---|---|
| `mk-a81e140d62ff170c` | 14 | 14 | 0 | **FAIL** | 3.1086 | −0.112 | 1.0360× |
| `mk-b6cb64f99049683b` | 14 | 14 | 0 | **FAIL** | 3.8226 | **+0.602** | **0.8425×** |
| `mk-baecc3825d4a8c18` | 14 | 14 | 0 | **FAIL** | 3.1008 | −0.120 | 1.0387× |
| `mk-bbfe15180d31bf50` | 14 | 14 | 0 | **FAIL** | 3.1112 | −0.110 | 1.0352× |

Three of the four were individually **faster** than native; one
(`mk-b6cb64f99049683b`) cost +0.60s. That invited the conclusion that
`mk-b6cb64f99049683b` was the latency offender.

**The repeat at `runs: 5` (SLURM 986) does not support that conclusion — it
reverses it:**

| artifact | run 985 (runs=2) | run 986 (runs=5) | swing |
|---|---|---|---|
| `mk-b6cb64f99049683b` | 0.8425× | **1.0912×** | 29.5% |
| `mk-baecc3825d4a8c18` | 1.0387× | **0.9177×** | 13.2% |

The native baselines explain why. Job 986's `vae-repeat` native runs were
[3.314, 3.342, 3.369, **4.587**, **4.347**] — a **38.4% spread** within one
trial on a shared node.

**The per-artifact VAE latency differences are node contention, not artifact
properties.** No reliable per-artifact latency attribution is possible for
effects of this size on this cluster at this sample count. R4's 0.8327×
combined figure, measured with `runs: 2`, is inside that noise band and should
not be treated as a precise quantity either.

What *is* reproducible and deterministic: **all four break `byte_equal` parity,
every time**. That is the disqualifying property, and it does not depend on
timing.

*(Checked and rejected: the two ragged-shape variants use `triton.cdiv` for
their grid, but they do mask the tail block correctly — there is no
out-of-bounds access.)*

### The dispatch overhead, measured exactly: 3.1 ms per call

The repaired transformer artifact gives a clean measurement, because its effect
is an order of magnitude larger than the noise and it is mechanistically
explained. Same scope, 384 invocations per generation:

```
native   3.6789 s
candidate 4.8226 s   = 0.7629x
2303 candidate calls over 6 generations = 384 per generation
delta per generation  = +1143.7 ms
net cost per call     = +2.980 ms
expected saving/call  =   0.124 ms   (259.05us -> 134.90us)
=> dispatch overhead  =   3.104 ms per call
```

This is measured **after** the two hot-path fixes below, so 3.1 ms is what
remains: the structural cost of `rewrite_exported_subgraph` replacing the whole
block's forward with an **eager FX GraphModule replay** in order to fuse a
subregion.

That single number explains both observations:

| call volume | overhead per generation | share of e2e |
|---|---|---|
| VAE, ~4.7 calls/generation | 14.5 ms | 0.39% — lost in the noise |
| transformer, 384 calls/generation | **1.192 s** | **32.4%** — fatal |

The VAE artifacts were never overhead-limited. The transformer is overhead-
limited and nothing else: its kernel saves 124 µs per call and the framework
charges 3104 µs to deliver it, a 25:1 loss.

### Framework overhead that was fixed

Two costs were paid per candidate call in `fastvideo/optimization/`. Both are
real and both are fixed, but the numbers above show they were not the dominant
term:

- `_dispatch` built **three** `_parameter_snapshot` dictionaries per call
  (walking module parameters, sorting hook names, listing tensor shapes) to
  diagnose FSDP materialization. A run that never fails never reads them.
- `_validate_runtime_inputs` walked **every node** of the rewritten graph on
  every call to rediscover its placeholders — a constant, re-derived inside the
  region it was meant to accelerate.

Beyond those, `rewrite_exported_subgraph` replaces the whole `res_blocks`
forward with an **eager FX GraphModule replay** in order to fuse 7 elementwise
nodes. That is a structural cost, not an oversight, and it is the dominant term.

---

## 4. The decisive arithmetic: these artifacts could never have passed

| artifact | share of e2e | measured speedup | realized e2e gain |
|---|---|---|---|
| `mk-bbfe15180d31bf50` | 2.334% | 1.1154× | 0.241% |
| `mk-a81e140d62ff170c` | 1.863% | 1.1104× | 0.185% |
| `mk-baecc3825d4a8c18` | 1.081% | 1.1212× | 0.117% |
| `mk-b6cb64f99049683b` | 0.922% | 1.1064× | 0.089% |
| **total** | **6.20%** | | **0.632% → 1.0064×** |

Campaign target: **1.01×**. Maximum achievable at zero dispatch overhead and
perfect parity: **1.0064×**. No combination of these four could have passed.

Packaging consulted `estimated_max_e2e_improvement` — an Amdahl bound of
`share × 0.9` assuming the region's cost drops to nearly zero — which for a
1.115× kernel overstates the return by more than 8×. Two of the four carried
`meets_promotion_target: false` into packaging anyway.

---

## 5. Metric defects found in the audit

| metric | R4 value | reality |
|---|---|---|
| `share_of_e2e` (transformer) | 1.0333 | attributed 2864051.95µs of a 2771790.13µs total; inclusive ranges summed against an exclusive total |
| `calls` (transformer) | 195661 | aten-event count across 48 blocks × 8 steps × 3 generations; the module was invoked **1151** times |
| search prompt | "Measured optimistic model impact: **None%**" | read `estimated_max_e2e_improvement_pct`, a key nothing writes |
| quarantine reason | "the end-to-end validation stage did not complete" | the stage completed (`status: ok`) and returned a definite negative; the real causes were parity failure and regression |
| candidate status | `finalized` (×4) | every artifact was `quarantined`, nothing promoted |

---

## 6. V1 gate status

Repaired transformer artifact `mk-2c92e356aa34bc0d-7df21b47-sm100`, packaged
from `transformer-fix`, dispatched alone (SLURM 986):

| # | gate | result |
|---|---|---|
| 1 | strict independent correctness | **PASS** — bit-exact under `byte_equal`, all stages, tolerances not loosened |
| 2 | packaged and hash verified | **PASS** — bundle re-verified from disk |
| 3 | selected and executed, `candidate_calls > 0` | **PASS** — 2303 calls, 0 fallbacks |
| 4 | `byte_equal` output policy preserved | **PASS** — "frame arrays are byte equal" |
| 5 | ≥1% end-to-end improvement | **FAIL** — 0.7629× |
| 6 | safely promoted | correctly **NOT** promoted; fail-closed held |

Gate 4 is the first byte-identical full-generation parity any MotionKernel
artifact has achieved on this workload. Gate 5 fails on framework dispatch
overhead (3.1 ms/call × 384 calls = 1.19 s), not on the kernel: the kernel is
1.920× and saves 124 µs per call.

**V1 is not proven, and the blocker is precisely located**: the export-subgraph
dispatch path costs ~3.1 ms per invocation because it replays an entire block
through an eager FX GraphModule to substitute a fused subregion. Until that
per-call cost falls below the per-call saving, no subgraph artifact on a
high-frequency region can pass gate 5, however good its kernel is.

### What would unblock it

1. Reduce the per-call cost of the rewritten-subgraph path — the FX replay is
   the term that matters, not the two hot-path costs already fixed.
2. Or dispatch high-frequency regions through a `module`-target artifact, which
   does not replay the parent graph.
3. Or select regions whose per-call saving exceeds ~3.1 ms. At 124 µs, this
   transformer subregion is 25× short.

---

## 7. Unblocking gate 5 (SLURM 987-998)

Gate 5 failed at 0.7629x because the dispatcher executed the rewritten export
graph in eager FX. Profiling attributed it precisely, in situ:

| phase | mean/call |
|---|---|
| `shadow.native_forward` (what we replaced) | 8.18 ms |
| `subgraph.execute` (eager replay) | **11.57 ms** |
| flatten + validate + unflatten + shape_key | 0.26 ms |

The plumbing was negligible. The op histogram explained the rest: the export
graph is **621 `call_function` nodes per call** (108 `aten.slice`, 68
`aten._assert_tensor_metadata`, 60 `aten.to.dtype`, ...). At ~5 us of Python
and dispatcher cost per op that is the whole 3.39 ms. **Attention was not the
problem** -- it survives capture as a single
`fastvideo._flash_attn_default_forward` op, so nothing was being silently
substituted.

### The fix

Replay the rewritten graph from a `torch.cuda.CUDAGraph` capture. A CUDA graph
runs the same kernels with the same parameters in the same order, so it is
bitwise identical **by construction** -- which is why it was chosen over a
compiler backend, whose fusion or reassociation would put `byte_equal` at risk.

Getting there took four real bugs, each found by measurement rather than
reasoning:

1. Every capture failed with `runtime input 10 is bool, not a tensor`. Export
   flattens Python scalars through as graph inputs; they are constants to bake
   in, not a reason to decline. **The fast path had never engaged once**, which
   means the 1.382 ms/call "improvement" measured before this was node
   variance.
2. Sharing one `graph_pool_handle` across the stack's 48 blocks tripped the
   allocator's `use_count > 0` assert: every graph keeps its outputs alive, so
   the pool is never free when the next capture starts. Each capture now gets
   its own pool.
3. A bool among the *outputs* needed the same treatment.
4. `aten._assert_tensor_metadata` nodes (68/call) are stripped -- they compute
   nothing.

### Correctness work after review

A review of the capture path found defects a `byte_equal` workload cannot
carry. All are fixed:

- **Captured parameter addresses were never re-checked.** FSDP2's `reshard`
  frees the all-gathered storage a capture recorded pointers into; the next
  replay would read freed memory. Every bound `get_attr` tensor's identity is
  now recorded and re-checked per replay.
- **Non-tensor outputs were returned uncloned** on the assumption that "not a
  Tensor" means "immutable". A list or dict leaf would have handed back the
  graph's own static buffers. Only `None` and scalars qualify now.
- **Pinned inputs were validated by `data_ptr` alone**, which folds in
  `storage_offset` but says nothing about strides -- a permuted view reusing
  the same allocator block passed every check while the captured kernels read
  elements in a different order. Pinning is now on
  `(data_ptr, shape, stride, dtype, device, storage_offset)`.
- An aborted capture never called `graph.reset()`, pinning its pool for the run.
- `"warming up"` was signalled by **comparing an exception message string**;
  rewording it would have turned every warmup into a permanent disable.

Most importantly, the argument that a capture *must* be bitwise identical is
now **checked rather than asserted**: the graph contains one node export did
not functionalize -- the artifact's own entry point -- so purity is an
assumption about third-party code. After capture the runner replays once, runs
the eager graph, and refuses the capture unless every output is bitwise equal.

---

## 8. V1 gate status: PROVEN

Artifact `mk-2c92e356aa34bc0d-7df21b47-sm100`, dispatched alone, 15 timed runs
per arm (SLURM 997), promoted by the real finalizer (SLURM 998).

| # | gate | result |
|---|---|---|
| 1 | strict independent correctness | **PASS** — bit-exact under `byte_equal`, tolerances not loosened |
| 2 | packaged and hash verified | **PASS** — verified from disk before and after finalization |
| 3 | selected and executed, `candidate_calls > 0` | **PASS** — 6143 calls, 0 fallbacks |
| 4 | `byte_equal` output policy preserved | **PASS** — "frame arrays are byte equal" |
| 5 | ≥1% end-to-end improvement | **PASS** — **1.0857×** median (1.0306× min-to-min) |
| 6 | safely promoted | **PASS** — `promotion: promoted`, decision derived from the measurement |

Native median 3.3646s (stdev 0.1675) → candidate median 3.0991s (stdev 0.0395).
Peak-memory regression 4.62% against the workload's 5% limit.

Honest note on where the win comes from: the artifact's kernel saves 124 us per
call, which alone caps end-to-end at ~1.015x. The measured 1.086x is larger
because CUDA-graphing the block also removes the *whole block's* host-side
dispatch cost. That acceleration exists only on the artifact path -- without an
artifact the block is never dispatched and never captured -- so the A/B is
sound, but the framework contributes more of the gain than the kernel does.

The candidate arm is markedly more reproducible than the baseline (stdev 0.0395
vs 0.1675), which is itself a consequence of replaying a fixed graph.

---

## 9. Review findings and final confirmation (SLURM 999)

Greptile flagged three further issues, all real and all fixed: a dead `if True:`
guard left by an earlier edit; the private memory pool being released on the
unexpected-error path but not when `_capture` itself refused (which mattered
because the bitwise-verification step ran *after* `self._graph` was assigned,
leaving a rejected capture published on the runner); and process-global timing
counters with no way to clear them between sessions. The graph is now published
only once verified.

Gate 5 re-confirmed with every guard in place, 15 timed runs per arm:

| | median | stdev | min |
|---|---|---|---|
| native | 3.7494 s | 0.1555 | 3.4689 |
| candidate | **2.9963 s** | **0.0205** | 2.9862 |

**1.2514× median, 1.1616× min-to-min.** Parity byte equal, 6143 calls, 0
fallbacks, peak memory +4.62%.

### Every paired A/B measurement taken

| run | timed runs/arm | speedup | parity |
|---|---|---|---|
| 994 | 5 | 1.0187× | byte equal |
| 995 | 5 | 1.0094× | byte equal |
| 996 | 15 | 1.1703× | byte equal |
| 997 | 15 | 1.0857× | byte equal |
| 999 | 15 | 1.2514× | byte equal |

Four of five clear the 1.01× gate; the fifth (1.0094×, a 5-run measurement) is
within noise of it. All three 15-run measurements clear it with margin. The
candidate arm's medians span 2.9963–3.1199s (4% spread) while the native arm's
span 3.1457–3.7494s (19%), so the residual variance is in the baseline, not in
the artifact path — itself a consequence of replaying a fixed graph.
