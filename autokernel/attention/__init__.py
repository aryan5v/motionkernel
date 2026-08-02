"""Attention backends as a first-class, verifiable dispatch boundary.

FastVideo already ships the implementations -- SageAttention, SageAttention3,
Video Sparse Attention, SLA, NABLA, and more -- behind
``AttentionBackendEnum`` and ``get_attn_backend``. What it does not ship is a
way for a promotion campaign to *prove* which of them actually ran.

That gap matters because the selector falls back silently. See
``fastvideo/platforms/cuda.py``: selecting ``SAGE_ATTN`` when the
``sageattention`` package is missing logs

    "Sage Attention backend is not installed. Fall back to Flash Attention."

and returns the Flash Attention backend. The generation then succeeds, at
baseline speed, with baseline numerics -- and a campaign measuring it would
record FlashAttention's behaviour as SageAttention's result. R4 spent a night
on exactly this shape of mistake in a different guise: a candidate that
appeared to be doing the work it was credited with, and was not.

This package closes that hole. An attention artifact declares the backend it
was measured with; :func:`verify_effective_backend` checks the backend actually
resolved at run time and refuses the run when they differ. Fallback is not
treated as degraded service -- for a measurement it is a wrong answer.

Note that FastVideo itself already takes this position for two backends:
``ATTN_QAT_TRAIN`` and ``NABLA_ATTN`` raise rather than fall back, on the
reasoning that a silent substitution "would produce a non-QAT training run" or
be "orders of magnitude slower and diverge from the reference". This package
extends that reasoning to every backend a campaign measures.
"""

from __future__ import annotations

from .identity import (
    FALLBACK_BACKEND,
    KNOWN_BACKENDS,
    AttentionBackendIdentity,
    AttentionFallbackError,
    AttentionIdentityError,
    backend_identity,
    verify_effective_backend,
)

__all__ = [
    "FALLBACK_BACKEND",
    "KNOWN_BACKENDS",
    "AttentionBackendIdentity",
    "AttentionFallbackError",
    "AttentionIdentityError",
    "backend_identity",
    "verify_effective_backend",
]
