# Optimize production stage adapters

`optimize.py` runs each stage in a subprocess and exchanges schema-version 1
stage results under `RUN/stages/STAGE/result.json`. The built-in driver now
connects the following stages to the existing FastVideo and MotionKernel APIs:

| Stage | Production action | Durable output |
| --- | --- | --- |
| `baseline` | FastVideo native generation through `run_ab()` | `RUN/generation/native_result.json` |
| `profile` | Dedicated post-warmup native generation with FX/export capture enabled | `RUN/stages/profile/profiler.json` |
| `discover` | Validate, correlate, rank, and persist the profiler/capture export | `RUN/stages/discover/discovery.json` |
| `specgen` | Generate a graph-derived spec and corpus for every search-worthy region | `RUN/candidates/FINGERPRINT/` |
| `package` | Package hash-verified quarantined bundles from measured isolated results | `RUN/artifacts/ARTIFACT_ID/` |
| `end_to_end_validate` | Run FastVideo candidate mode, require dispatch, compare frames, and classify end-to-end timing | `RUN/stages/end_to_end_validate/` plus `RUN/generation/candidate_result.json` |
| `finalize` | Rewrite the selected quarantined bundles with measured generation evidence and a promoted/rejected decision | `RUN/artifacts/ARTIFACT_ID/artifact.json` |

`search` and `isolated_validate` remain external stage commands. Configure both
with `--stage-commands`; the control plane still applies its campaign and
per-candidate time budgets to those subprocesses.

## External isolated-validation handoff

The `isolated_validate` result must use the ordinary stage-result envelope and
add a non-empty `package_requests` list:

```json
{
  "schema_version": 1,
  "stage": "isolated_validate",
  "status": "ok",
  "metrics": {
    "isolated_correct": true,
    "isolated_speedup": 1.23
  },
  "package_requests": [
    {
      "source_dir": "/campaign/candidates/FINGERPRINT/payload",
      "sections": {
        "artifact_id": "candidate-id",
        "operation": {},
        "signature": {},
        "entry_point": {},
        "compatibility": {},
        "evidence": {
          "benchmark": {"passed": true},
          "generation": {"passed": false}
        },
        "promotion": {"decision": "quarantined"}
      }
    }
  ]
}
```

The abbreviated objects above must contain every field required by artifact
schema 1. The adapter does not infer benchmark numbers, compatibility ranges,
or promotion evidence. It rejects a failed isolated benchmark, a pre-validation
`promoted` decision, and generation evidence claiming to have passed before
the full-generation stage ran.

The end-to-end adapter loads quarantined bundles only in FastVideo's explicit
validation mode. A successful result requires all three conditions:

1. FastVideo diagnostics report at least one `artifact_selected` decision.
2. Saved output frames pass the workload's parity policy.
3. Native-versus-candidate wall time meets the campaign speedup threshold
   without exceeding the workload memory-regression limit.

## Artifact finalization

A packaged bundle is `quarantined` and carries pending generation evidence: it
has never run inside the model. The `finalize` stage is the only step allowed
to rewrite those two sections, and only from what `end_to_end_validate`
measured.

The stage reads the dispatch diagnostics FastVideo wrote during validation and
requires both halves of the record to agree:

```json
{
  "dispatch": {
    "reason_counts": {"artifact_selected": 1},
    "decisions": [
      {"artifact_id": "candidate-one", "reason": "artifact_selected"},
      {"artifact_id": "candidate-two", "reason": "fingerprint_mismatch"}
    ]
  }
}
```

`reason_counts.artifact_selected` says how many selections happened and
`decisions` says which artifacts they were. The adapter fails closed when the
two disagree, when a selection carries no `artifact_id`, or when a selected id
was never packaged: finalizing on ambiguous diagnostics would promote a bundle
that may never have executed.

For every packaged bundle the stage then:

- verifies the bundle, hash for hash, before changing anything;
- finalizes only the bundles dispatch actually selected, leaving every other
  bundle quarantined (it is still verified, so a tampered bundle sitting in the
  artifact root fails the stage);
- writes `promoted` only when output parity passed, dispatch selected the
  artifact, the end-to-end classification is `improved`, and the measured
  speedup meets the stricter of the campaign and workload thresholds;
- writes `rejected` when the run completed but the candidate is neutral,
  regressed, or below the speedup gate — the measured evidence is recorded with
  `passed: false`;
- leaves the bundle byte-for-byte untouched when parity failed, the artifact
  was not selected, or the measurement is incomplete.

`evidence.generation` is replaced with the measured record
(`metric: "end_to_end_speedup"`, the measured `value`, the configured
`threshold`, the workload id and step count, and the native/candidate result
paths as `baseline_ref`/`candidate_ref`). `evidence.benchmark`, the payload
files, and their digests are carried over unchanged; finalization refuses to
write a manifest that would alter either.

The manifest is written atomically through a temporary file in the artifact
root — never inside the bundle, where debris would show up as an undeclared
file — and the finished bundle is re-verified exactly the way a runtime does.
If the write or the re-verification fails, the packaged manifest is restored,
so an interrupted finalization always leaves a loadable bundle behind.

Finalization is write-once, which makes resume safe: re-running it against an
already finalized bundle re-derives the same decision and returns without
rewriting anything, and a run whose evidence would *change* a shipped decision
fails closed instead of weakening it.

The stage result reports the finalized bundle paths and every decision:

```json
{
  "schema_version": 1,
  "stage": "finalize",
  "status": "ok",
  "recommendation": "promoted",
  "metrics": {
    "artifacts_promoted": 1,
    "artifacts_rejected": 0,
    "artifacts_quarantined": 1,
    "artifacts_selected": 1
  },
  "artifacts": {
    "root": "/campaign/artifacts",
    "finalized": [
      {
        "artifact_id": "candidate-one",
        "bundle_dir": "/campaign/artifacts/candidate-one",
        "manifest": "/campaign/artifacts/candidate-one/artifact.json",
        "decision": "promoted",
        "reason": "promoted: parity (byte_equal) passed, ...",
        "changed": true
      }
    ],
    "quarantined": []
  },
  "decisions": [{"artifact_id": "candidate-one", "decision": "promoted"}]
}
```

## Current contract gaps

- Artifact schema 1 requires a generation evidence object before the candidate
  can be packaged for its first full-generation run. The external validation
  command must therefore provide an explicit pending/failed measurement record;
  the adapter never invents one.
- Artifact schema 1 has no dedicated parity field in `evidence.generation`.
  Finalization records the measured parity policy and its result in the
  promotion `reason` and reflects it in `passed`; a machine-readable parity
  record needs a schema 2 field.
- Finalization is write-once by design. A genuinely re-measured campaign that
  should overturn a shipped decision has no supported path yet: it fails closed
  and must be republished under a new artifact id.
- End-to-end evidence is per *run*, not per artifact. When one generation
  selects several bundles they all record the same measurement, because the
  campaign cannot attribute the wall-time delta to an individual artifact.
- The dispatch `decisions` list is the only link between a bundle and the run
  that exercised it. FastVideo must emit one entry per selection with the
  artifact id; a runtime that reports counts only cannot be finalized.
- Search has no universal invocation contract yet. Its command must write the
  candidate payload and the isolated validator must identify that payload in
  `package_requests`.
- The profile adapter currently assumes a single profiler export path. A
  multi-rank workload needs a future aggregation contract for rank-suffixed
  exports before discovery can rank the complete workload.

