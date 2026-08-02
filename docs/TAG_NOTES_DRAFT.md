# Draft tag notes — v1.0.0

> **Draft. Not published.** Publishing, tagging and detaching the fork are all
> held pending review.

## MotionKernel V1

Verified GPU kernel optimization for video generation models. V1 is the first
release where a generated kernel goes from discovery to promotion with evidence
at every gate.

### Proven

One artifact, end to end, on FastVideo LTX2:

- strict independent correctness, bit-exact under a `byte_equal` policy;
- packaged and hash-verified;
- dispatched 6,143 times with zero runtime fallbacks;
- byte-identical generated frames;
- **1.0857x** median end-to-end over 15 timed runs per arm on an NVIDIA GB200,
  replicated at 1.2514x;
- promoted by a decision derived from that measurement.

Scope is deliberately narrow and stated in
[SUPPORT_STATUS.md](SUPPORT_STATUS.md): one artifact, one workload
(`ltx_480p`), one GPU architecture (sm100), single-GPU inference. Other models,
resolutions and architectures are untested, not known-good.

An honest note carried in the evidence report: the artifact's kernel saves
~124 microseconds per call, which alone bounds end-to-end near 1.015x. The
larger figure comes from the runtime additionally replaying the block from a
CUDA graph. That acceleration exists only on the artifact path, so the A/B is
sound, but the framework contributes more of the gain than the kernel.

### Also in this release

- `motionkernel` canonical import namespace, alongside the `autokernel`
  compatibility namespace. Both supported, both type-check.
- Workload parity policy governs every correctness gate, not just the final
  frame comparison.
- Packaging gated on measured end-to-end impact rather than an optimistic
  upper bound.
- Per-artifact end-to-end isolation, so a parity or latency change can be
  attributed to a specific bundle.
- Provenance inventory generated from Git history.
- Security policy stating the threat model, including that loading an artifact
  executes code.

### Not claimed

Cosmos is a **candidate**: integration work exists, no authoritative end-to-end
evidence has been published. Wan has isolated operator results only. See the
support matrix.

### Provenance

MotionKernel is an independently maintained, MIT-licensed fork of
[RightNow-AI/AutoKernel](https://github.com/RightNow-AI/autokernel). The
upstream MIT notice and copyright are preserved; `PROVENANCE.md` records the
per-file split.
