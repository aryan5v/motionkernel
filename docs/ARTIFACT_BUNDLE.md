# Artifact bundles

An **artifact bundle** is how an optimized graph kernel is packaged once and
loaded safely by a model runtime. The runtime integrates the contract a single
time; every future kernel arrives as data, never as a new code path.

```
store/
  fused-scale-add-sm90/
    artifact.json     # the manifest: identity, compatibility, evidence
    kernel.py         # the candidate entry point
    notes.txt         # anything else the bundle needs, all hashed
```

## What the manifest records

| Section | Contents |
| --- | --- |
| `operation` | operation name, 32-hex `graph_fingerprint`, parent module, canonical operation list |
| `signature` | input and output tensor signatures (shape, stride, dtype, device type, `requires_grad`) |
| `entry_point` | the candidate `file` and `symbol` |
| `files` | every bundled file with its SHA-256 and byte size |
| `compatibility` | model id and revision, GPU architectures, PyTorch/CUDA/Triton version ranges, execution modes, distributed modes |
| `evidence.benchmark` | isolated harness measurement: baseline/candidate microseconds, speedup, sample count, error bounds, pass flag |
| `evidence.generation` | full-generation validation: workload, steps, metric, value, threshold, pass flag |
| `promotion` | decision, reason, timestamp, and the source campaign |

`schema_version` is `1`. It is versioned independently of the discovery and
campaign schemas, so a manifest change never invalidates a profiler reader.

The manifest is metadata only. Tensor values, weights, prompts, activations and
credentials are rejected structurally: a forbidden-key walk runs over the whole
document before it is accepted.

## Packaging

```python
from autokernel.artifact import package_artifact

manifest = package_artifact(
    "workspace/generated_specs/wan",   # payload directory
    "workspace/artifacts/fused-scale-add-sm90",
    sections,                          # everything except files/schema_version
)
```

Every regular file in the payload is hashed and declared, so a bundle can never
reference content that was not measured at package time. `files`,
`schema_version` and `created_at` are computed by the packager and are refused
if the caller supplies them. The finished directory is then re-verified from
disk exactly the way a consuming runtime will verify it -- if that fails,
packaging fails.

## Matching

```python
from autokernel.artifact import DispatchRequest, RuntimeProfile, match_artifact

result = match_artifact(manifests, DispatchRequest(
    graph_fingerprint=fingerprint,
    inputs=live_inputs,
    outputs=live_outputs,
    runtime=RuntimeProfile.detect(model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
))
```

Selection is keyed on the captured **graph fingerprint** and the **tensor
signatures**, not on any environment switch naming a model. Checks run
cheapest-first -- graph identity, then tensor layout, then the declared
environment window -- and each failure returns a stable reason code:

`fingerprint_mismatch`, `input_signature_mismatch`, `output_signature_mismatch`,
`model_mismatch`, `model_revision_mismatch`, `gpu_architecture_mismatch`,
`torch_version_unsupported`, `cuda_version_unsupported`,
`triton_version_unsupported`, `execution_mode_unsupported`,
`distributed_mode_unsupported`, `not_promoted`, `evidence_incomplete`.

When several bundles qualify, the one with the strongest measured benchmark
speedup wins, breaking ties on artifact id so selection is deterministic.

`"*"` is accepted as a wildcard for `model_id`, `model_revision` and any entry
of `gpu_architectures`. A version range with no bounds accepts anything; a
range **with** a bound rejects a runtime whose version is missing or
unparsable, because a bound that cannot be evaluated must never be assumed to
hold.

## Trust

`load_entry_point` imports the candidate under three rules:

1. The bundle must resolve inside the explicitly configured trusted root.
   A symlink or crafted path that escapes it is refused.
2. Every declared file is re-hashed immediately before the import, and any
   undeclared file in the directory is a hard rejection -- otherwise an
   attacker who can drop an extra module beside a signed entry point could have
   it imported.
3. The bundle is never added to `sys.path`; it is loaded under a private
   `autokernel._artifacts.*` name so it cannot shadow a real module.

## Candidate calling convention

The entry point is called as `candidate(module, *args, **kwargs)`: the native
module first, then the exact arguments its `forward` received. Passing the
module is what lets one artifact serve every block in a repeated stack -- the
kernel reads the parameters it needs from the module it was handed, instead of
the loader having to know which parameters exist.

## Failure behavior

Nothing here is best-effort. A malformed manifest, a changed byte, an
unreadable entry point or an unsatisfied compatibility bound all produce an
`ArtifactError` with a located message. A runtime is expected to catch it and
run natively; `load_bundles` returns the manifests that verified alongside an
error string per bundle that did not, so one bad artifact cannot hide the good
ones.
