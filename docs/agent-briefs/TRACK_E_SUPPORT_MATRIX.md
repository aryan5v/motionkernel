# Track E — the support matrix gets real

Goal: an auto-generated **model × workload × arch → outcome** page, published
from CI, derived from evidence rather than claims. This is the project's first
public credibility artifact, and it is the one most easily discredited, so the
bar is that every cell traces to a run.

---

## 1. Why this is delicate

The README currently says, correctly:

> These are isolated operator results; complete model packs still require
> end-to-end benchmark publication before support is claimed.

That sentence is the standard you are held to. A matrix that shows green for a
model whose only evidence is an isolated operator benchmark would be a
regression in honesty, not a feature. Every cell must carry a link to the run
that produced it and the date it was produced.

Cell vocabulary — use exactly these, and make "we never tried" visually
distinct from "we tried and it failed":

| outcome | meaning |
|---|---|
| `promoted` | an artifact was promoted by the real finalizer, with e2e evidence |
| `no_worthwhile_candidate` | discovery ran, found nothing above threshold |
| `capture_blocked` | capture failed — **must carry the reason** |
| `not_attempted` | no run exists. Not a failure, but never blank |

`capture_blocked` with no reason is worse than useless; it is the failure mode
R4 §5 records, where a quarantine reason said "the end-to-end validation stage
did not complete" when the stage had in fact completed and returned a definite
negative, suppressing the real cause.

---

## 2. Scope

Workload manifests for every FastVideo family the project claims to target:

- Wan 1.3B and 14B
- LTX / LTX2
- FastWan
- Hunyuan
- Cosmos

Plus **two architectures**: sm100 (GB200, via SLURM) and sm90 (H100, via
Modal). Adding the non-GB200 arch is an explicit exit criterion.

Existing manifests are in `workloads/` (`ltx_480p.yaml`,
`wan_t2v_1.3b_480p.yaml`) — follow their schema exactly. It is validated:
`autokernel/workload/types.py` rejects unknown fields, forbids embedded tensor
values/credentials/weights, and requires a `schema_version`.

Note the schema recently gained an optional `fidelity` block (PR #18,
`docs/TIERED_FIDELITY.md`). Absent means Tier 1 (`exact`), which is the correct
default; you do not need to add it unless a workload is genuinely Tier 2.

---

## 3. Nightly discovery-only campaigns

Discovery-only, not search — you are populating the matrix, not optimizing.
Requirements:

- **Idempotent and resumable.** A nightly that cannot resume will lose whole
  runs to preemption.
- **Immutable evidence directories.** Existing runs use timestamped roots under
  `<nfs-root>/` (e.g. `ltx-v1-overnight-20260801-r4-sol`). Never write
  into a previous run's directory; the R4 analysis depended on those being
  untouched.
- **A failure to capture is data.** Record it with its reason rather than
  retrying until it disappears.

---

## 4. Generation must be mechanical

The page is generated from run artifacts, never hand-edited. Concretely:

- a script reads run result JSON and emits the matrix;
- CI runs it and fails if the committed page differs from the generated one
  (otherwise it drifts and quietly becomes a claim again);
- every cell links to its evidence and carries a date;
- a cell whose evidence is older than some staleness window is visually marked.

Prefer boring formats. A committed Markdown page plus a JSON sidecar is
sufficient and reviewable; a dashboard is not needed.

---

## 5. GPU access

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

Partition `all`, `sbatch --export=NIL`, NFS root `<nfs-root>`.

- **No `/home/<account>`** — set `#SBATCH --chdir=<nfs-root>` or jobs
  error with `couldn't chdir ... going to /tmp instead`.
- **Bare `srun` has no CUDA tooling**, not even `nvidia-smi`. Use Pyxis:

```bash
srun --container-image=nvcr.io/nvidia/pytorch:25.06-py3 \
     --container-mounts=<nfs-root>:<nfs-root> \
     --container-workdir=<nfs-root>/<your-dir> \
     bash -lc "<command>"
```

`Reason=Resources` / `StartTime=Unknown` does not mean the cluster is full —
jobs have started within a minute of showing that. Check `sacct -j <id>`.

Useful: `squeue -o "%.8i %.9P %.20j %.8u %.2t %.10M %R"` to see who is holding
what. Long-running holds by other users are normal; plan around them rather
than waiting.

### Modal — H100, sm90

`modal` CLI, workspace `hao-ai-lab` (`modal profile list`). This is your second
architecture — you cannot get sm90 from SLURM. A working app with an image
build, a pinned commit and a GPU function is at
`modal/verify_tiered_fidelity.py` on branch `tiered-fidelity-contracts`.

Model weights: SLURM runs use an HF cache under `<nfs-root>`; Modal
uses a `fastvideo-hf-cache` volume with `HF_HUB_OFFLINE=1`. Check
`~/Fast video1/modal-wan-h100/motionkernel_wan_h100.py` for the established
image and volume pattern before building your own.

---

## 6. Exit criteria

1. Validated workload manifests for all six families, passing the existing
   schema validation.
2. A nightly discovery campaign that runs unattended, resumes, and writes
   immutable evidence.
3. A generated support matrix page covering model × workload × arch, including
   **both** sm100 and sm90.
4. CI regenerates and diffs it, failing on drift.
5. Every non-`not_attempted` cell links to evidence with a date; every
   `capture_blocked` carries a reason.

## 7. Branching and deliverable

Branch from `main`, name `support-matrix`. No prefixes.

Draft PR against `aryan5v/motionkernel` — pass `--repo aryan5v/motionkernel`,
because `gh repo view` resolves these checkouts to the upstream fork parent
`RightNow-AI/autokernel` and PR creation otherwise fails with a misleading
`No commits between main and <branch>`.

If a family cannot be captured at all, that is a legitimate and useful
result — record it as `capture_blocked` with the reason and move on. Do not
leave it blank, and do not let it block the other five.
