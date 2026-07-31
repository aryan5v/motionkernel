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

## Current contract gaps

- Artifact schema 1 requires a generation evidence object before the candidate
  can be packaged for its first full-generation run. The external validation
  command must therefore provide an explicit pending/failed measurement record;
  the adapter never invents one.
- The linear pipeline ends at `end_to_end_validate`. Its stage result can make
  the campaign decision, but there is no post-validation artifact-finalization
  stage that rewrites the quarantined manifest with measured generation evidence
  and a promoted/rejected decision. Production publication must remain a
  separate explicit step until that contract exists.
- Search has no universal invocation contract yet. Its command must write the
  candidate payload and the isolated validator must identify that payload in
  `package_requests`.
- The profile adapter currently assumes a single profiler export path. A
  multi-rank workload needs a future aggregation contract for rank-suffixed
  exports before discovery can rank the complete workload.

