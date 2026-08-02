"""CPU tests for attention as a first-class artifact kind.

Two things are under test: that artifact kinds are now registered rather than
enumerated (so Track C can add a schedule transform without editing a
conditional every other kind reads), and that an attention artifact cannot be
credited with a run it did not perform.

The second is the one that matters. FastVideo's selector falls back to
FlashAttention silently when an optional backend cannot be imported --
``fastvideo/platforms/cuda.py`` logs "Sage Attention backend is not installed.
Fall back to Flash Attention." and returns the Flash backend. A campaign that
requested SAGE_ATTN and measured the fallback would record FlashAttention's
speed and numerics under SageAttention's name.
"""

from __future__ import annotations

import pytest

from autokernel.artifact.kinds import (
    ATTENTION,
    MODULE,
    SUBGRAPH,
    TargetKind,
    known_target_kinds,
    register_target_kind,
    target_kind_spec,
    validate_kind_fields,
)
from autokernel.artifact.types import ArtifactError, OperationIdentity
from autokernel.attention import (
    FALLBACK_BACKEND,
    KNOWN_BACKENDS,
    AttentionFallbackError,
    AttentionIdentityError,
    backend_identity,
    verify_effective_backend,
)

FINGERPRINT = "0123456789abcdef0123456789abcdef"


def _operation(**overrides) -> dict:
    values = {
        "name": "attention",
        "graph_fingerprint": FINGERPRINT,
        "parent_module": "transformer.blocks.0.attn",
        "operations": ["fastvideo._flash_attn_default_forward"],
    }
    values.update(overrides)
    return values


def _parse(**overrides) -> OperationIdentity:
    return OperationIdentity.from_dict(
        _operation(**overrides), source="<test>", location="operation"
    )


# -- the kind registry --------------------------------------------------


def test_built_in_kinds_are_registered() -> None:
    assert set(known_target_kinds()) >= {MODULE, SUBGRAPH, ATTENTION}


def test_an_absent_kind_defaults_to_the_most_constrained_one() -> None:
    assert _parse().target_kind == MODULE


def test_attention_is_not_a_region_replacing_kind() -> None:
    """It selects an implementation for an op that survives capture whole.

    Region-replacing kinds are matched by graph fingerprint; this one needs its
    own compatibility check and must not be dispatched by fingerprint alone.
    """
    assert target_kind_spec(ATTENTION).replaces_region is False
    assert target_kind_spec(SUBGRAPH).replaces_region is True


def test_registering_a_duplicate_kind_is_refused() -> None:
    # Silent replacement would let import order decide what validates.
    with pytest.raises(ValueError, match="already registered"):
        register_target_kind(TargetKind(name=ATTENTION))


def test_a_field_belonging_to_another_kind_is_named_as_such() -> None:
    """"unknown field" would send the reader hunting for a typo."""
    with pytest.raises(ArtifactError, match="belongs to a 'subgraph' target"):
        _parse(target_kind=ATTENTION, attention_backend="SAGE_ATTN",
               capture_mode="export")


def test_module_targets_still_reject_subgraph_rewrite_fields() -> None:
    with pytest.raises(ArtifactError, match="belongs to a 'subgraph' target"):
        _parse(target_kind=MODULE, selected_node_ids=["n0"])


def test_unknown_kind_lists_the_known_ones() -> None:
    # Deliberately not a plausible-but-unregistered name: "schedule_transform"
    # was used here until Track C registered it, and the test then passed for
    # the wrong reason right up until it failed for the right one.
    with pytest.raises(ArtifactError, match="unknown target_kind"):
        _parse(target_kind="definitely_not_a_registered_kind")


def test_a_new_kind_can_be_registered_without_touching_validation() -> None:
    """Track C must be able to add its kind as a registration."""
    register_target_kind(
        TargetKind(
            name="test_only_kind",
            required=frozenset({"threshold"}),
            replaces_region=False,
            description="a test kind",
        )
    )
    spec = validate_kind_fields("test_only_kind", {"threshold": 0.4})
    assert spec.replaces_region is False
    with pytest.raises(ValueError, match="requires"):
        validate_kind_fields("test_only_kind", {})


# -- the attention kind's own fields ------------------------------------


def test_attention_target_requires_a_backend() -> None:
    with pytest.raises(ArtifactError, match="requires"):
        _parse(target_kind=ATTENTION)


def test_attention_target_rejects_an_unknown_backend() -> None:
    with pytest.raises(ArtifactError, match="unknown attention backend"):
        _parse(target_kind=ATTENTION, attention_backend="TOTALLY_MADE_UP")


def test_attention_target_round_trips_backend_and_config() -> None:
    parsed = _parse(
        target_kind=ATTENTION,
        attention_backend="VIDEO_SPARSE_ATTN",
        attention_config={"sparsity": 0.75},
    )
    assert parsed.attention_backend == "VIDEO_SPARSE_ATTN"
    assert parsed.attention_config == {"sparsity": 0.75}
    rendered = parsed.as_dict()
    assert rendered["target_kind"] == ATTENTION
    assert rendered["attention_backend"] == "VIDEO_SPARSE_ATTN"
    assert rendered["attention_config"] == {"sparsity": 0.75}
    assert OperationIdentity.from_dict(
        rendered, source="<test>", location="operation"
    ) == parsed


def test_module_targets_do_not_emit_attention_fields() -> None:
    assert "attention_backend" not in _parse().as_dict()


# -- the silent fallback, which is the point ----------------------------


def test_the_declared_backend_running_is_accepted() -> None:
    identity = verify_effective_backend("SAGE_ATTN", "SAGE_ATTN")
    assert identity.name == "SAGE_ATTN"


def test_a_silent_fallback_to_flash_attention_is_refused_by_name() -> None:
    """The message must say which backend was wanted and which ran."""
    with pytest.raises(AttentionFallbackError) as caught:
        verify_effective_backend("SAGE_ATTN", FALLBACK_BACKEND)
    message = str(caught.value)
    assert "SAGE_ATTN" in message
    assert FALLBACK_BACKEND in message
    assert "sageattention" in message  # names the missing import
    assert "measures" in message  # says what the run actually measured


def test_an_unreported_backend_is_a_failure_not_agreement() -> None:
    """An unreported backend is exactly what a silent fallback leaves behind."""
    with pytest.raises(AttentionFallbackError, match="no effective backend"):
        verify_effective_backend("SAGE_ATTN", None)


def test_a_substitution_between_two_real_backends_is_refused() -> None:
    with pytest.raises(AttentionFallbackError, match="does not measure"):
        verify_effective_backend("SAGE_ATTN", "VIDEO_SPARSE_ATTN")


def test_a_matching_name_with_the_wrong_class_is_refused() -> None:
    with pytest.raises(AttentionFallbackError, match="implementation"):
        verify_effective_backend(
            "SAGE_ATTN",
            "SAGE_ATTN",
            effective_class_path="fastvideo.attention.backends.flash_attn."
            "FlashAttentionBackend",
        )


def test_an_unknown_declared_backend_is_an_identity_error() -> None:
    with pytest.raises(AttentionIdentityError, match="unknown attention backend"):
        verify_effective_backend("NOPE", "NOPE")


# -- which tier these backends can be promoted at -----------------------


def test_the_interesting_backends_are_all_inexact() -> None:
    """Which is why attention artifacts are gated at tier 2, not tier 1.

    If any of these were bit-exact it could be promoted at tier 1 and would not
    need perceptual evidence at all. None of them are: they quantize the
    attention product or skip blocks outright.
    """
    for name in ("SAGE_ATTN", "SAGE_ATTN_THREE", "VIDEO_SPARSE_ATTN", "SLA_ATTN"):
        identity = backend_identity(name)
        assert not identity.exact, name
        assert identity.notes, f"{name} must say why it is inexact"


def test_the_always_available_backends_are_exact_and_not_optional() -> None:
    for name in ("FLASH_ATTN", "TORCH_SDPA"):
        identity = backend_identity(name)
        assert identity.exact
        assert not identity.optional


def test_every_known_backend_declares_a_class_path() -> None:
    for name, identity in KNOWN_BACKENDS.items():
        assert identity.class_path.startswith("fastvideo.attention.backends."), name


# -- the promotion gate -------------------------------------------------


def _outcome(**overrides):
    """A GenerationOutcome that would otherwise promote."""
    from autokernel.artifact import GenerationOutcome

    values = {
        "workload_id": "wan-t2v-1.3b",
        "steps": 8,
        "parity_passed": False,  # an attention backend is never bit-exact
        "artifact_selected": True,
        "classification": "improved",
        "min_speedup": 1.01,
        "speedup": 1.42,
    }
    values.update(overrides)
    return GenerationOutcome(**values)


def _tier2():
    from autokernel.verification.fidelity import FidelityBudget

    return FidelityBudget(
        tier="perceptual",
        min_ssim=0.98,
        max_lpips=0.02,
        frame_set="wan-fixed-seed-8",
        seed=1234,
    )


def _evidence(**overrides):
    from autokernel.verification.fidelity import PerceptualEvidence

    values = {
        "frame_set": "wan-fixed-seed-8",
        "seed": 1234,
        "frames_compared": 8,
        "ssim": 0.9903,
        "lpips": 0.0089,
    }
    values.update(overrides)
    return PerceptualEvidence(**values)


def test_a_silent_fallback_quarantines_an_otherwise_promotable_run() -> None:
    """Everything else says promote: 1.42x, selected, improved, tier-2 clean.

    Only the backend check stands between a FlashAttention run and a
    SageAttention promotion.
    """
    decision, reason = _outcome(
        fidelity=_tier2(),
        perceptual=_evidence(),
        attention_declared="SAGE_ATTN",
        attention_effective=FALLBACK_BACKEND,
    ).decide()
    assert decision == "quarantined"
    assert "SAGE_ATTN" in reason and FALLBACK_BACKEND in reason


def test_an_unreported_backend_quarantines() -> None:
    decision, reason = _outcome(
        fidelity=_tier2(),
        perceptual=_evidence(),
        attention_declared="SAGE_ATTN",
        attention_effective=None,
    ).decide()
    assert decision == "quarantined"
    assert "no effective backend" in reason


def test_an_inexact_backend_cannot_be_promoted_at_tier_1() -> None:
    """Declaring tier 1 for a quantizing backend is a contract error.

    Without this the run would quarantine anyway on parity, but with a reason
    that blames the numerics rather than the mis-declared budget.
    """
    decision, reason = _outcome(
        attention_declared="SAGE_ATTN",
        attention_effective="SAGE_ATTN",
    ).decide()
    assert decision == "quarantined"
    assert "tier 1" in reason
    assert "tier 2" in reason  # tells the operator what to do instead


def test_the_declared_backend_running_at_tier_2_promotes() -> None:
    decision, reason = _outcome(
        fidelity=_tier2(),
        perceptual=_evidence(),
        attention_declared="SAGE_ATTN",
        attention_effective="SAGE_ATTN",
    ).decide()
    assert decision == "promoted"
    assert "tier 2 (perceptual)" in reason


def test_non_attention_artifacts_are_unaffected() -> None:
    from autokernel.artifact import GenerationOutcome

    decision, _ = GenerationOutcome(
        workload_id="w",
        steps=4,
        parity_passed=True,
        artifact_selected=True,
        classification="improved",
        min_speedup=1.01,
        speedup=1.2,
    ).decide()
    assert decision == "promoted"


def test_an_unknown_declared_backend_is_rejected_at_construction() -> None:
    from autokernel.artifact import ArtifactError

    with pytest.raises(ArtifactError, match="unknown attention backend"):
        _outcome(attention_declared="MADE_UP", attention_effective="MADE_UP")
