# Parallel track briefs

Five tracks. One is done, one is in progress, four can run in parallel **right
now** by four separate agents. Each brief in this directory is self-contained —
hand one whole file to one agent.

## Status

| track | what | status | owner |
|---|---|---|---|
| B | tiered fidelity contracts | **done** — PR #18, verified on sm90+sm100 | — |
| A | attention as a promotable artifact | **in progress** | do not assign |
| C | schedule transforms (caching) | ready | `TRACK_C_CACHING.md` |
| D | dispatch overhead, measured and published | ready | `TRACK_D_DISPATCH.md` |
| E | support matrix from evidence | ready | `TRACK_E_SUPPORT_MATRIX.md` |
| F | search farm + experiment store | ready | `TRACK_F_SEARCH_FARM.md` |

## Read this before assigning anything

Three premises in the original roadmap did not survive contact with the code.
Agents given the original wording will waste days re-deriving these:

1. **The 3.1 ms dispatch tax is already fixed.** Not with `torch.library`
   custom ops, but with `torch.cuda.CUDAGraph` capture — chosen deliberately
   because a graph replay is bitwise-identical *by construction*, while a
   compiler backend's fusion or reassociation would put `byte_equal` at risk.
   Gate 5 passes at **1.2514×** median (15 runs/arm, SLURM 999). See
   `docs/LTX_V1_R4_ROOT_CAUSE.md` §7–9. Track D is re-scoped accordingly.

2. **The four quarantined VAE artifacts cannot be rehabilitated.** §4 of that
   document does the arithmetic: 6.20% of end-to-end at ~1.11× each gives
   **1.0064× maximum** at *zero* dispatch overhead, against a 1.01× target.
   They also break `byte_equal` deterministically (`rcp.approx.ftz`,
   `tanh.approx`). Both disqualifiers are independent of dispatch cost. Do not
   assign "re-dispatch the quarantined artifacts" as if the verdict might flip.

3. **FastVideo already ships the attention backends.** `sage_attn.py`,
   `sage_attn3.py`, `video_sparse_attn.py`, `sla.py`, `nabla.py`, `bsa_attn.py`
   and more, with a full selection mechanism (`AttentionBackendEnum`,
   `FASTVIDEO_ATTENTION_BACKEND`, `get_attn_backend`). Track A is not an
   integration job; it is about making the backend *choice* a captured,
   searchable, promotable artifact. That is why Track A is not being farmed out.

## Ownership map — read this to avoid collisions

The one real collision risk is `target_kind` in
`autokernel/artifact/types.py` (~line 548), today `{"module", "subgraph"}`.
**Track A and Track C both need to extend it.**

Resolution: **Track A lands the extension mechanism first.** Track C must
branch from Track A's branch once it is pushed (`attention-artifact-kind`), not
from `main`, and must add its kind through that mechanism rather than editing
the literal set. Track C's agent should check whether
`attention-artifact-kind` exists on the remote before starting; if it does not
yet, do the parts of Track C that do not touch `target_kind` first (the cache
implementation and its tests) and wire the artifact kind last.

Otherwise:

| track | owns (exclusive write access) |
|---|---|
| A | `autokernel/artifact/types.py`, `autokernel/discovery/`, FastVideo `fastvideo/attention/` |
| C | `autokernel/optimize/` cache stage, new `autokernel/transforms/` |
| D | `fastvideo/optimization/` dispatch path, benchmark scripts |
| E | `workloads/`, `docs/SUPPORT_MATRIX.md`, CI workflow files |
| F | `autokernel/optimize/search.py`, new experiment store, `AGENTS.md` |

D, E and F touch disjoint files and can start immediately from `main`.

## Branching

| track | branch from | branch name |
|---|---|---|
| C | `attention-artifact-kind` (Track A) | `schedule-transform-caching` |
| D | `main` | `dispatch-overhead-published` |
| E | `main` | `support-matrix` |
| F | `main` | `search-farm-population` |

No prefixes (`claude/`, `feature/`) — repository convention.

Track C additionally needs the tiered-fidelity work, which is on PR #18
(`tiered-fidelity-contracts`) and not yet merged. Track A's branch already
includes it, so branching from Track A gets both.
