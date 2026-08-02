# Security policy

## Reporting a vulnerability

Report suspected vulnerabilities privately through
[GitHub Security Advisories](https://github.com/aryan5v/motionkernel/security/advisories/new).
Please do not open a public issue for a security report.

> **Before the first public release**, private vulnerability reporting must be
> enabled on the repository (*Settings → Security → Private vulnerability
> reporting*). It is currently disabled, and Issues are disabled too, so a
> reporter following this page has no working channel. Verified against the
> repository API; see `docs/RELEASE_CHECKLIST.md` §7.

Include the affected version or commit, what an attacker can achieve, and the
smallest reproduction you have. If a report involves an artifact bundle, attach
its `artifact.json` rather than the payload.

Expect an acknowledgement within 7 days and an assessment within 30. There is
no bounty programme.

## Supported versions

MotionKernel is pre-1.0 as a public project. Fixes land on `main`; there are no
maintained backport branches yet.

## Threat model

MotionKernel runs an autonomous agent that writes GPU kernels, and it loads
kernel code from artifact bundles. Both are code execution by design, so it is
worth being precise about what is defended and what is not.

### Trusted

- The repository, the fixed benchmark harness, specifications, shape corpora,
  and the verifier. The search agent may not write to any of them.
- Artifact bundles inside an explicitly configured trusted root.

### Untrusted

- **Generated kernel code.** The search agent may edit exactly one file, the
  candidate's `kernel.py`. Its own benchmark result is never the acceptance
  signal: every candidate is independently re-measured by a fixed harness in a
  separate process, and the artifact request is derived only from that
  measurement and the validated manifest.
- **Artifact bundles from anywhere else.** See below.

### Loading an artifact executes code

An artifact bundle contains Python that the runtime imports and runs on your
GPU, in your process. The protections are:

- executable code is imported only from inside the explicitly configured
  trusted root — no path from a manifest is ever followed;
- every declared file is hashed and compared before the entry point is
  imported, and undeclared files are rejected;
- an unknown schema version, a missing field, or a changed byte is a hard
  rejection, never a best-effort load;
- compatibility (model, revision, GPU architecture, framework versions,
  execution and distributed mode) must match, and the graph fingerprint is
  re-derived from the live model.

**These checks establish integrity, not trustworthiness.** They prove a bundle
is the one that was packaged and that it matches the model in front of it. They
do not prove the bundle was packaged by someone you should trust. Treat an
artifact directory exactly as you would treat a directory of Python you are
about to `import`: only point the runtime at bundles you or your organisation
produced, or received over a channel you trust.

There is no signature verification yet. Adding it is tracked on the roadmap;
until it exists, provenance is your responsibility.

### Not in scope

- Numerical differences that are within a workload's declared parity policy.
  Set `parity.policy: byte_equal` if you need bitwise output.
- Denial of service from a kernel that is merely slow. The end-to-end gate
  catches regressions; it is not a security boundary.
- The upstream AutoKernel project. Report issues in inherited code that also
  affect upstream to both projects.

## Handling of secrets

The optimize control plane records command configurations as SHA-256 digests
plus a program basename, so no credential, prompt, or raw argument is written
to `preflight.json` or campaign state. Dispatch diagnostics and profiler
exports are metadata-only: operation names, shapes, dtypes and timings, never
tensor values.

If you find a path where MotionKernel writes a secret, a prompt, or tensor
contents to disk or to a log, please report it as a vulnerability.
