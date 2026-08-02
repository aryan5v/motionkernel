"""CPU tests for post-validation artifact finalization."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autokernel.artifact import (
    REASON_NOT_PROMOTED,
    ArtifactError,
    DispatchRequest,
    GenerationOutcome,
    RuntimeProfile,
    TensorSignature,
    check_compatibility,
    finalize_bundle,
    package_artifact,
    read_manifest,
    verify_bundle,
)
from autokernel.artifact import finalizer as finalizer_module
from autokernel.optimize.adapters import ProductionAdapterError, run_production_stage

FINGERPRINT = "0123456789abcdef0123456789abcdef"

FAKE_KERNEL = '''"""CPU fake kernel used by the finalization tests."""


def fused_scale_add(hidden, residual, scale=2.0):
    return hidden * scale + residual
'''


def _tensor(shape, name=""):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= max(dim, 1)
    stride.reverse()
    return {
        "shape": list(shape),
        "stride": stride,
        "dtype": "float32",
        "device_type": "cuda",
        "requires_grad": False,
        **({"name": name} if name else {}),
    }


def _sections(artifact_id: str = "candidate-one") -> dict:
    """The exact quarantined sections the package stage writes."""
    return {
        "artifact_id": artifact_id,
        "operation": {
            "name": "generated_blocks_scale_add",
            "graph_fingerprint": FINGERPRINT,
            "parent_module": "transformer.blocks",
            "operations": ["aten::mul", "aten::add"],
        },
        "signature": {
            "inputs": [
                _tensor((2, 4), name="input_0"),
                _tensor((2, 4), name="input_1"),
            ],
            "outputs": [_tensor((2, 4), name="output_0")],
        },
        "entry_point": {"file": "kernel.py", "symbol": "fused_scale_add"},
        "compatibility": {
            "model_id": "fake/model",
            "model_revision": "main",
            "gpu_architectures": ["sm90"],
            "torch": {"min": "2.4.0", "max_exclusive": "3.0.0"},
            "cuda": {"min": "12.0"},
            "triton": {"min": "3.0.0"},
            "execution_modes": ["inference"],
            "distributed_modes": ["single"],
        },
        "evidence": {
            "benchmark": {
                "harness": "motionkernel-bench",
                "device": "cuda:0",
                "samples": 50,
                "baseline_us": 120.0,
                "candidate_us": 80.0,
                "speedup": 1.5,
                "max_abs_error": 1e-4,
                "max_rel_error": 1e-3,
                "atol": 2e-2,
                "rtol": 2e-2,
                "passed": True,
                "result_ref": "results/bench-001.json",
            },
            "generation": {
                "workload_id": "pending",
                "steps": 1,
                "metric": "pending_full_generation",
                "value": 0.0,
                "threshold": 1.0,
                "passed": False,
            },
        },
        "promotion": {
            "decision": "quarantined",
            "reason": "packaged before full-generation validation",
            "decided_at": "2026-01-01T00:00:00+00:00",
            "campaign": {
                "campaign_id": "campaign-001",
                "source": "motionkernel-optimize",
                "target_name": "transformer.blocks",
            },
        },
    }


def _bundle(tmp_path: Path, artifact_id: str = "candidate-one") -> Path:
    """Package one real quarantined bundle and return its directory."""
    payload = tmp_path / "payload" / artifact_id
    payload.mkdir(parents=True)
    (payload / "kernel.py").write_text(FAKE_KERNEL, encoding="utf-8")
    output = tmp_path / "artifacts" / artifact_id
    package_artifact(payload, output, _sections(artifact_id))
    return output


def _outcome(**overrides) -> GenerationOutcome:
    values = {
        "workload_id": "fake-workload",
        "steps": 4,
        "parity_passed": True,
        "artifact_selected": True,
        "classification": "improved",
        "min_speedup": 1.01,
        "speedup": 1.21,
        "parity_policy": "byte_equal",
        "baseline_ref": "generation/native_result.json",
        "candidate_ref": "generation/candidate_result.json",
    }
    values.update(overrides)
    return GenerationOutcome(**values)


def _request() -> DispatchRequest:
    inputs = tuple(
        TensorSignature.from_dict(item, source="test", location="input")
        for item in _sections()["signature"]["inputs"]
    )
    outputs = tuple(
        TensorSignature.from_dict(item, source="test", location="output")
        for item in _sections()["signature"]["outputs"]
    )
    return DispatchRequest(
        graph_fingerprint=FINGERPRINT,
        inputs=inputs,
        outputs=outputs,
        runtime=RuntimeProfile(
            model_id="fake/model",
            model_revision="main",
            gpu_architecture="sm90",
            torch_version="2.6.0",
            cuda_version="12.4",
            triton_version="3.2.0",
        ),
    )


# --- finalizer API -------------------------------------------------------


def test_promotion_records_measured_evidence_and_keeps_the_payload(tmp_path: Path):
    bundle = _bundle(tmp_path)
    packaged = read_manifest(bundle)
    payload_before = (bundle / "kernel.py").read_bytes()

    result = finalize_bundle(bundle, _outcome())

    assert result.decision == "promoted"
    assert result.changed is True
    manifest = verify_bundle(bundle)
    assert manifest.promotion.decision == "promoted"
    assert "byte_equal" in manifest.promotion.reason
    assert manifest.evidence.generation.passed is True
    assert manifest.evidence.generation.workload_id == "fake-workload"
    assert manifest.evidence.generation.metric == "end_to_end_speedup"
    assert manifest.evidence.generation.value == pytest.approx(1.21)
    assert manifest.evidence.generation.threshold == pytest.approx(1.01)
    assert manifest.evidence.generation.candidate_ref.endswith("candidate_result.json")
    # Payload bytes and isolated benchmark evidence survive untouched.
    assert (bundle / "kernel.py").read_bytes() == payload_before
    assert manifest.files == packaged.files
    assert manifest.evidence.benchmark == packaged.evidence.benchmark


@pytest.mark.parametrize(
    "classification,speedup",
    [("neutral", 1.0), ("regressed", 0.92), ("improved", 1.001)],
)
def test_completed_but_unimproved_candidate_is_rejected(
    tmp_path: Path, classification: str, speedup: float
):
    bundle = _bundle(tmp_path)

    result = finalize_bundle(
        bundle, _outcome(classification=classification, speedup=speedup)
    )

    assert result.decision == "rejected"
    assert result.changed is True
    manifest = verify_bundle(bundle)
    assert manifest.promotion.decision == "rejected"
    # A rejection still records what was measured, and never claims a pass.
    assert manifest.evidence.generation.passed is False
    assert manifest.evidence.generation.value == pytest.approx(speedup)


def test_parity_failure_leaves_the_bundle_quarantined(tmp_path: Path):
    bundle = _bundle(tmp_path)
    before = (bundle / "artifact.json").read_bytes()

    result = finalize_bundle(bundle, _outcome(parity_passed=False))

    assert result.decision == "quarantined"
    assert result.changed is False
    assert result.reason == "packaged before full-generation validation"
    assert (bundle / "artifact.json").read_bytes() == before
    assert verify_bundle(bundle).promotion.decision == "quarantined"


def test_unselected_artifact_leaves_the_bundle_quarantined(tmp_path: Path):
    bundle = _bundle(tmp_path)

    result = finalize_bundle(bundle, _outcome(artifact_selected=False))

    assert result.decision == "quarantined"
    assert result.reason == "packaged before full-generation validation"
    assert verify_bundle(bundle).promotion.decision == "quarantined"


@pytest.mark.parametrize(
    "overrides",
    [
        {"classification": "failed", "speedup": None},
        {"speedup": None},
        {"stage_status": "failed"},
    ],
)
def test_incomplete_or_failed_validation_leaves_the_bundle_quarantined(
    tmp_path: Path, overrides: dict
):
    bundle = _bundle(tmp_path)

    result = finalize_bundle(bundle, _outcome(**overrides))

    assert result.decision == "quarantined"
    assert result.changed is False
    assert verify_bundle(bundle).evidence.generation.passed is False


def test_tampered_bundle_is_rejected_before_anything_is_written(tmp_path: Path):
    bundle = _bundle(tmp_path)
    # Same byte count, different content: only the hash can catch this.
    (bundle / "kernel.py").write_text(
        FAKE_KERNEL.replace("scale + residual", "scale - residual"), encoding="utf-8"
    )

    with pytest.raises(ArtifactError, match="does not match manifest"):
        finalize_bundle(bundle, _outcome())

    assert read_manifest(bundle).promotion.decision == "quarantined"


def test_finalization_is_idempotent_and_never_weakens_a_decision(tmp_path: Path):
    bundle = _bundle(tmp_path)
    first = finalize_bundle(bundle, _outcome())
    manifest_bytes = (bundle / "artifact.json").read_bytes()

    second = finalize_bundle(bundle, _outcome())

    assert second.decision == "promoted"
    assert second.changed is False
    assert second.reason == first.reason
    assert (bundle / "artifact.json").read_bytes() == manifest_bytes

    # A later run that would demote the shipped bundle fails closed instead.
    with pytest.raises(ArtifactError, match="already finalized as 'promoted'"):
        finalize_bundle(bundle, _outcome(classification="regressed", speedup=0.9))
    assert (bundle / "artifact.json").read_bytes() == manifest_bytes


def test_interrupted_finalization_restores_the_original_valid_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle = _bundle(tmp_path)
    before = (bundle / "artifact.json").read_bytes()
    real_verify = finalizer_module.verify_bundle
    calls = {"count": 0}

    def flaky_verify(path):
        calls["count"] += 1
        if calls["count"] > 1:
            raise OSError("interrupted before the finalized bundle was verified")
        return real_verify(path)

    monkeypatch.setattr(finalizer_module, "verify_bundle", flaky_verify)

    with pytest.raises(OSError, match="interrupted"):
        finalize_bundle(bundle, _outcome())

    monkeypatch.undo()
    assert (bundle / "artifact.json").read_bytes() == before
    assert verify_bundle(bundle).promotion.decision == "quarantined"
    # No temporary debris may be left inside the bundle: an undeclared file
    # would make an otherwise intact bundle fail verification.
    assert sorted(p.name for p in bundle.iterdir()) == ["artifact.json", "kernel.py"]


def test_promoted_bundle_is_accepted_by_normal_production_dispatch(tmp_path: Path):
    bundle = _bundle(tmp_path)
    request = _request()

    quarantined = verify_bundle(bundle)
    rejection = check_compatibility(quarantined, request)
    assert rejection is not None and rejection.reason == REASON_NOT_PROMOTED

    finalize_bundle(bundle, _outcome())

    assert check_compatibility(verify_bundle(bundle), request) is None


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"workload_id": " "}, "workload_id"),
        ({"steps": 0}, "steps"),
        ({"min_speedup": float("inf")}, "min_speedup"),
        ({"speedup": float("nan")}, "speedup"),
        ({"stage_status": "unknown"}, "stage_status"),
    ],
)
def test_generation_outcome_rejects_unusable_measurements(overrides: dict, match: str):
    with pytest.raises(ArtifactError, match=match):
        _outcome(**overrides)


# --- optimize stage adapter ---------------------------------------------


def _stage_input(run_dir: Path, **overrides) -> None:
    checkout = run_dir / "FastVideo"
    checkout.mkdir(parents=True, exist_ok=True)
    workload = run_dir / "workload.yaml"
    workload.write_text("schema_version: 1\n", encoding="utf-8")
    config = {
        "fastvideo_checkout": str(checkout),
        "workload": str(workload),
        "model": "fake/model",
        "baseline": "compile",
        "min_e2e_speedup": 1.01,
        "artifact_dir_name": "artifacts",
    }
    config.update(overrides)
    path = run_dir / "stages" / "finalize" / "input.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"stage": "finalize", "config": config, "state_snapshot": {}}),
        encoding="utf-8",
    )


def _prior(run_dir: Path, stage: str, **extra) -> None:
    path = run_dir / "stages" / stage / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 1, "stage": stage, "status": "ok", **extra}),
        encoding="utf-8",
    )


def _diagnostics(run_dir: Path, decisions, count: int | None = None) -> Path:
    path = run_dir / "stages" / "end_to_end_validate" / "dispatch.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    dispatch: dict = {
        "reason_counts": {
            "artifact_selected": (
                len(
                    [
                        item
                        for item in decisions
                        if isinstance(item, dict)
                        and item.get("reason") == "artifact_selected"
                    ]
                )
                if count is None
                else count
            )
        }
    }
    payload = {"dispatch": dispatch}
    if decisions is not None:
        payload["decisions"] = decisions
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _wire(
    run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    bundles,
    decisions,
    count: int | None = None,
    metrics: dict | None = None,
    recommendation: str = "promoted",
) -> None:
    _stage_input(run_dir)
    _prior(
        run_dir,
        "package",
        artifacts={"root": str(run_dir / "artifacts"), "bundles": bundles},
    )
    diagnostics = _diagnostics(run_dir, decisions, count)
    _prior(
        run_dir,
        "end_to_end_validate",
        metrics={
            "classification": "improved",
            "parity_passed": True,
            "artifact_selected": True,
            "end_to_end_speedup": 1.2,
            **(metrics or {}),
        },
        recommendation=recommendation,
        parity={"passed": True, "policy": "byte_equal"},
        artifacts={
            "native_result": str(run_dir / "generation" / "native_result.json"),
            "candidate_result": str(run_dir / "generation" / "candidate_result.json"),
            "dispatch_diagnostics": str(diagnostics),
        },
    )
    monkeypatch.setattr(
        "autokernel.optimize.adapters.load_workload",
        lambda _path: SimpleNamespace(
            workload_id="fake-workload",
            sampling=SimpleNamespace(num_inference_steps=4),
            performance=SimpleNamespace(min_end_to_end_speedup=1.01),
        ),
    )


def _entry(bundle: Path) -> dict:
    return {"artifact_id": bundle.name, "bundle_dir": str(bundle)}


def test_finalize_stage_promotes_only_the_selected_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    selected = _bundle(tmp_path, "candidate-one")
    other = _bundle(tmp_path, "candidate-two")
    _wire(
        tmp_path,
        monkeypatch,
        bundles=[_entry(selected), _entry(other)],
        decisions=[
            {"artifact_id": "candidate-one", "reason": "artifact_selected"},
            {"artifact_id": "candidate-two", "reason": "fingerprint_mismatch"},
        ],
    )

    result = run_production_stage("finalize", tmp_path)

    assert result["recommendation"] == "promoted"
    assert result["metrics"] == {
        "artifacts_promoted": 1,
        "artifacts_rejected": 0,
        "artifacts_quarantined": 1,
        "artifacts_selected": 1,
    }
    assert result["artifacts"]["finalized"][0]["artifact_id"] == "candidate-one"
    assert result["artifacts"]["finalized"][0]["manifest"].endswith("artifact.json")
    assert result["decisions"] == [
        {"artifact_id": "candidate-one", "decision": "promoted"},
        {"artifact_id": "candidate-two", "decision": "quarantined"},
    ]
    assert verify_bundle(selected).promotion.decision == "promoted"
    # The bundle that never ran keeps its quarantine.
    assert verify_bundle(other).promotion.decision == "quarantined"


def test_finalize_stage_rejects_a_neutral_end_to_end_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle = _bundle(tmp_path)
    _wire(
        tmp_path,
        monkeypatch,
        bundles=[_entry(bundle)],
        decisions=[{"artifact_id": "candidate-one", "reason": "artifact_selected"}],
        metrics={"classification": "neutral", "end_to_end_speedup": 1.0},
        recommendation="no_worthwhile_candidate",
    )

    result = run_production_stage("finalize", tmp_path)

    assert result["recommendation"] == "no_worthwhile_candidate"
    assert result["metrics"]["artifacts_rejected"] == 1
    assert verify_bundle(bundle).promotion.decision == "rejected"


def test_finalize_stage_holds_quarantine_when_validation_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle = _bundle(tmp_path)
    _wire(
        tmp_path,
        monkeypatch,
        bundles=[_entry(bundle)],
        decisions=[{"artifact_id": "candidate-one", "reason": "artifact_selected"}],
        metrics={"classification": "failed", "parity_passed": False},
        recommendation="failed",
    )

    result = run_production_stage("finalize", tmp_path)

    assert result["recommendation"] == "failed"
    assert result["metrics"]["artifacts_quarantined"] == 1
    assert verify_bundle(bundle).promotion.decision == "quarantined"


def test_finalize_stage_is_safe_to_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle = _bundle(tmp_path)
    _wire(
        tmp_path,
        monkeypatch,
        bundles=[_entry(bundle)],
        decisions=[{"artifact_id": "candidate-one", "reason": "artifact_selected"}],
    )
    first = run_production_stage("finalize", tmp_path)
    manifest_bytes = (bundle / "artifact.json").read_bytes()

    second = run_production_stage("finalize", tmp_path)

    assert first["recommendation"] == second["recommendation"] == "promoted"
    assert second["artifacts"]["finalized"][0]["changed"] is False
    assert (bundle / "artifact.json").read_bytes() == manifest_bytes


@pytest.mark.parametrize(
    "decisions,count,match",
    [
        (None, 1, "carry no decisions list"),
        ([], 2, "ambiguous"),
        (
            [{"artifact_id": "candidate-one", "reason": "artifact_selected"}],
            2,
            "ambiguous",
        ),
        ([{"reason": "artifact_selected"}], None, "without an id"),
        (["not-an-object"], None, "must be an object"),
    ],
)
def test_finalize_stage_fails_closed_on_ambiguous_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decisions,
    count,
    match: str,
):
    bundle = _bundle(tmp_path)
    _wire(
        tmp_path,
        monkeypatch,
        bundles=[_entry(bundle)],
        decisions=decisions,
        count=count,
    )

    with pytest.raises(ProductionAdapterError, match=match):
        run_production_stage("finalize", tmp_path)

    assert verify_bundle(bundle).promotion.decision == "quarantined"


def test_finalize_stage_fails_closed_on_missing_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle = _bundle(tmp_path)
    _wire(
        tmp_path,
        monkeypatch,
        bundles=[_entry(bundle)],
        decisions=[{"artifact_id": "candidate-one", "reason": "artifact_selected"}],
    )
    (tmp_path / "stages" / "end_to_end_validate" / "dispatch.json").unlink()

    with pytest.raises(
        ProductionAdapterError, match="cannot read dispatch diagnostics"
    ):
        run_production_stage("finalize", tmp_path)


def test_finalize_stage_fails_closed_when_a_selected_bundle_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle = _bundle(tmp_path)
    _wire(
        tmp_path,
        monkeypatch,
        bundles=[_entry(bundle)],
        decisions=[{"artifact_id": "candidate-ghost", "reason": "artifact_selected"}],
    )

    with pytest.raises(ProductionAdapterError, match="candidate-ghost"):
        run_production_stage("finalize", tmp_path)

    assert verify_bundle(bundle).promotion.decision == "quarantined"


def test_finalize_stage_rejects_a_tampered_unselected_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    selected = _bundle(tmp_path, "candidate-one")
    other = _bundle(tmp_path, "candidate-two")
    (other / "kernel.py").write_text(
        FAKE_KERNEL.replace("scale + residual", "scale - residual"), encoding="utf-8"
    )
    _wire(
        tmp_path,
        monkeypatch,
        bundles=[_entry(selected), _entry(other)],
        decisions=[{"artifact_id": "candidate-one", "reason": "artifact_selected"}],
    )

    with pytest.raises(ProductionAdapterError, match="does not match manifest"):
        run_production_stage("finalize", tmp_path)
