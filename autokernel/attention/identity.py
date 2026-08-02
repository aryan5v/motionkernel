"""Which attention backend a run actually used, and whether that is the one claimed.

This module never imports torch or FastVideo. It reasons over backend *names*
and resolved class paths, both of which the runtime can report cheaply, so the
check is usable from a validation stage, from a finalizer, and from a test.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "FALLBACK_BACKEND",
    "KNOWN_BACKENDS",
    "AttentionBackendIdentity",
    "AttentionFallbackError",
    "AttentionIdentityError",
    "backend_identity",
    "verify_effective_backend",
]


class AttentionIdentityError(ValueError):
    """An attention backend declaration is malformed or unknown."""


class AttentionFallbackError(RuntimeError):
    """The backend that ran is not the backend that was claimed.

    Raised rather than warned. A campaign that measures a fallback and records
    it under the requested backend's name produces a number that is not wrong
    by a little -- it is a measurement of a different system.
    """


#: What FastVideo silently falls back to when an optional backend fails to
#: import. Named so a fallback can be reported as such rather than as a
#: mysterious mismatch.
FALLBACK_BACKEND = "FLASH_ATTN"


@dataclass(frozen=True)
class AttentionBackendIdentity:
    """One selectable attention implementation.

    Args:
        name: the ``AttentionBackendEnum`` member name, e.g. ``SAGE_ATTN``.
        class_path: the fully-qualified backend class FastVideo resolves to.
        requires: import names that must be present for this backend to be
            selectable at all. Empty for backends that are always available.
        exact: whether this backend can reproduce an eager reference bitwise.
            False for anything that quantizes or sparsifies -- which is most of
            the interesting ones, and is why attention artifacts are gated at
            fidelity tier 2 rather than tier 1.
        notes: one line of why it is inexact, when it is.
    """

    name: str
    class_path: str
    requires: tuple[str, ...] = ()
    exact: bool = True
    notes: str = ""

    @property
    def optional(self) -> bool:
        """Whether this backend can fail to import and be silently replaced."""
        return bool(self.requires)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "class_path": self.class_path,
            "requires": list(self.requires),
            "exact": self.exact,
        }


def _backend(*args: Any, **kwargs: Any) -> tuple[str, AttentionBackendIdentity]:
    identity = AttentionBackendIdentity(*args, **kwargs)
    return identity.name, identity


#: The backends FastVideo can select, as of ``fastvideo/platforms/interface.py``
#: and ``fastvideo/platforms/cuda.py``. ``exact`` reflects whether the backend
#: is capable of bitwise agreement with an eager reference, which decides the
#: fidelity tier an artifact using it can be promoted at.
KNOWN_BACKENDS: Mapping[str, AttentionBackendIdentity] = dict(
    [
        _backend(
            "FLASH_ATTN",
            "fastvideo.attention.backends.flash_attn.FlashAttentionBackend",
        ),
        _backend(
            "TORCH_SDPA",
            "fastvideo.attention.backends.sdpa.SDPABackend",
        ),
        _backend(
            "SAGE_ATTN",
            "fastvideo.attention.backends.sage_attn.SageAttentionBackend",
            requires=("sageattention",),
            exact=False,
            notes="quantizes the attention product; cannot match eager bitwise",
        ),
        _backend(
            "SAGE_ATTN_THREE",
            "fastvideo.attention.backends.sage_attn3.SageAttention3Backend",
            requires=("sageattn3",),
            exact=False,
            notes="FP4/Blackwell attention; cannot match eager bitwise",
        ),
        _backend(
            "VIDEO_SPARSE_ATTN",
            "fastvideo.attention.backends.video_sparse_attn."
            "VideoSparseAttentionBackend",
            requires=("fastvideo_kernel",),
            exact=False,
            notes="skips attention blocks; output differs by construction",
        ),
        _backend(
            "SLA_ATTN",
            "fastvideo.attention.backends.sla.SlaAttentionBackend",
            requires=("fastvideo_kernel",),
            exact=False,
            notes="sparse-linear attention; output differs by construction",
        ),
        _backend(
            "NABLA_ATTN",
            "fastvideo.attention.backends.nabla.NablaAttentionBackend",
            requires=("torch.nn.attention.flex_attention",),
            exact=False,
            notes="block-sparse flex attention; output differs by construction",
        ),
    ]
)


def backend_identity(name: str) -> AttentionBackendIdentity:
    """Look up a backend by ``AttentionBackendEnum`` member name."""
    try:
        return KNOWN_BACKENDS[name]
    except KeyError:
        raise AttentionIdentityError(
            f"unknown attention backend {name!r}; expected one of "
            f"{sorted(KNOWN_BACKENDS)}"
        ) from None


def verify_effective_backend(
    declared: str,
    effective: str | None,
    *,
    effective_class_path: str | None = None,
) -> AttentionBackendIdentity:
    """Confirm the backend that ran is the backend the artifact claims.

    Args:
        declared: the backend name recorded on the artifact.
        effective: the backend name the runtime actually resolved. ``None``
            means the runtime did not report one, which is treated as a failure
            rather than as agreement -- an unreported backend is exactly the
            state a silent fallback leaves behind.
        effective_class_path: the resolved class path, checked when supplied.
            A name can match while the class does not if two enum members map
            to one implementation, so this is the stronger signal when present.

    Returns:
        The declared backend's identity, when everything agrees.

    Raises:
        AttentionIdentityError: the declared backend is not one we know.
        AttentionFallbackError: the run used a different backend. When the
            substitute is FastVideo's fallback the message says so explicitly,
            because "SAGE_ATTN was requested and FLASH_ATTN ran" is a far more
            actionable sentence than "backend mismatch".
    """
    identity = backend_identity(declared)

    if effective is None:
        raise AttentionFallbackError(
            f"artifact declares attention backend {declared!r} but the runtime "
            f"reported no effective backend; refusing to attribute this run "
            f"(an unreported backend is what a silent fallback leaves behind)"
        )

    if effective != declared:
        if effective == FALLBACK_BACKEND and identity.optional:
            raise AttentionFallbackError(
                f"attention backend {declared!r} was requested but "
                f"{FALLBACK_BACKEND!r} ran: FastVideo falls back silently when "
                f"{' / '.join(identity.requires)} cannot be imported. This run "
                f"measures {FALLBACK_BACKEND!r}, not {declared!r}"
            )
        raise AttentionFallbackError(
            f"artifact declares attention backend {declared!r} but {effective!r} "
            f"ran; this run does not measure the declared backend"
        )

    if (
        effective_class_path is not None
        and effective_class_path != identity.class_path
    ):
        raise AttentionFallbackError(
            f"attention backend {declared!r} resolved to "
            f"{effective_class_path!r}, not the expected "
            f"{identity.class_path!r}; the name matches but the implementation "
            f"does not"
        )

    return identity
