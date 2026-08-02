# Track F — search farm and experiment memory

Goal: parallel population-based candidate search plus a retrievable experiment
store, and a head-to-head night showing **population search beats single-agent
search on the same region and the same budget**.

The exit criterion is a comparison, not a feature. Build the smallest thing
that lets you run that comparison honestly.

---

## 1. Two halves

### 1a. SLURM-array parallel search with a population layer

N sandboxed candidates × M agents per region. The population layer is the
point: keep top-k lineages and feed each agent the **diagnostics and diffs of
its siblings' best attempts** (the EvoEngineer / MaxCode pattern). An agent that
cannot see why a sibling's kernel failed will rediscover the same failure.

Existing search lives in `autokernel/optimize/search.py`, with sandboxing in
`autokernel/optimize/isolation.py`. Read both before designing. Note there is
already an allowlist mechanism for what a candidate may read (see the
`agent/v1-search-trust-fix` and `agent/v1-safe-subregion-ranking` branches in
the history) — **do not widen it** to make sibling-sharing easier. Share
diagnostics through an explicit channel you control, not by relaxing isolation.

### 1b. The experiment store

Start boring: **SQLite or parquet**, one row per bench result, carrying kernel
source, fingerprint, shapes, arch, measured timings, and outcome. Then wire
retrieval into search prompts:

> "here are the 3 best prior kernels for similar fingerprints on sm100"

Fingerprint machinery already exists in `autokernel/discovery/fingerprint.py` —
reuse it rather than inventing a similarity key.

Also in scope if time allows: `AGENTS.md`, and an MCP server. **The MCP server
is the designated cut** — if anything is dropped, drop that.

---

## 2. Measurement discipline — this track is the most at risk

Your exit criterion is a comparison between two search strategies, and search
outcomes are high-variance. The failure mode is declaring victory on noise.

Rules:

- **Same region, same budget, same node class.** Budget means wall-clock GPU
  time or candidate count — fix which one and state it.
- **Report the distribution, not the winner.** Best-of-N is a biased statistic:
  with more agents you get a better max even from an identical generator. If
  population search only wins on max and not on median, that is the finding.
- **Run the comparison more than once.** A single night is one sample.
- Beware the baseline: this cluster has shown **38.4% spread** in native
  generation time within a single trial, and one artifact swung **29.5%**
  between two runs at different sample counts. Timing-derived fitness inherits
  all of that.
- ≥15 timed runs per arm for anything you report as a speedup.

If the honest result is "population search did not beat single-agent search at
this budget", that is a genuinely valuable finding and you should report it as
the headline. Do not tune until it wins and then report the tuned run.

---

## 3. Correctness gates apply to searched kernels

Everything the search produces still goes through the existing verification
chain, and the chain exists because of specific past failures. Do not bypass it
for throughput:

- `autokernel/verification/policy.py` propagates the workload's parity policy
  to *every* stage. Before it existed, a `byte_equal` workload's upstream gates
  compared with "whatever tolerance happened to be lying around", and four
  approximate kernels got packaged.
- `detect_approximate_math()` textually screens for `rcp.approx`, `tanh.approx`,
  `__expf`, `allow_tf32` and similar **before** spending GPU time. Keep that
  screen in the loop; it is cheap and it is the reason a hopeless candidate
  does not consume a benchmark slot.
- A candidate must not bypass Dynamo's guards. R4 §1b records a candidate that
  called Inductor's raw compiled entry directly to "avoid a guard/cache lookup",
  which both hid a layout violation and made its 2.557× an unfair comparison
  against a baseline that paid for guard evaluation.
- Shapes and strides must be read at run time, not compiled in as `constexpr`.
  R4 §1a is the cautionary tale.

---

## 4. GPU access

**Never run GPU workloads on the login node.** This track is the heaviest SLURM
user of the five — array jobs, many nodes.

### SLURM — GB200, sm100, torch 2.8.0a0

The SLURM login host, account name and SSH key path are **not recorded in this
repository** — it is public. Get them from the project operator, along with the
key itself, and connect with:

```bash
ssh -o BatchMode=yes -o IdentitiesOnly=yes \
    -i <path-to-key> <account>@<login-host>
```

Never display, upload, commit or copy the private key.

Partition `all`, `sbatch --export=NIL`, NFS root `<nfs-root>`.

- **No `/home/<account>`** — set `#SBATCH --chdir=<nfs-root>`.
- **Bare `srun` has no CUDA tooling**, not even `nvidia-smi`. Use Pyxis:

```bash
srun --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
     --container-mounts=<nfs-root>:<nfs-root> \
     --container-workdir=<nfs-root>/<your-dir> \
     bash -lc "<command>"
```

- `Reason=Resources` / `StartTime=Unknown` is not proof of no capacity — jobs
  have started within a minute of showing it. Check `sacct -j <id>`.
- **You share this cluster.** `squeue -o "%.8i %.9P %.20j %.8u %.2t %.10M %R"`
  shows other users' holds; one has held many nodes for days at a time. Size
  your arrays so a stuck array does not starve the other four tracks, and
  prefer `--array=...%N` to cap concurrency.
- There are existing array/campaign scripts in `<nfs-root>/*.sbatch`
  and in `~/Fast video1/*.sbatch` — read them before writing your own.

### Modal — H100, sm90

`modal` CLI, workspace `hao-ai-lab`. Better for quick functional iteration than
for the head-to-head (different arch, different noise profile). Working example:
`modal/verify_tiered_fidelity.py` on branch `tiered-fidelity-contracts`.

---

## 5. Exit criteria

1. A SLURM-array search that runs N candidates × M agents per region, sandboxed,
   with a population layer sharing sibling diagnostics and diffs.
2. An experiment store (SQLite/parquet) holding every bench result with source,
   fingerprint and outcome.
3. Retrieval wired into search prompts, demonstrably changing what agents see.
4. **A head-to-head run** comparing population vs single-agent search on the
   same region and budget, reported as a distribution with more than one
   sample — whichever way it comes out.
5. `AGENTS.md`. (MCP server optional; cut this first.)

## 6. Branching and deliverable

Branch from `main`, name `search-farm-population`. No prefixes.

Draft PR against `aryan5v/motionkernel` — pass `--repo aryan5v/motionkernel`;
`gh` resolves these checkouts to the upstream fork parent and otherwise fails
with a misleading `No commits between main and <branch>`.

Do not let the store's schema become a research project. One table, boring
columns, written on every bench result. It is more valuable complete and dull
than elegant and half-populated.
