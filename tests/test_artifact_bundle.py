"""CPU tests for the versioned artifact bundle contract."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest

from autokernel.artifact import (
    ANY,
    ARTIFACT_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    REASON_ARCHITECTURE_MISMATCH,
    REASON_CUDA_VERSION,
    REASON_DISTRIBUTED_MODE,
    REASON_EVIDENCE_INCOMPLETE,
    REASON_EXECUTION_MODE,
    REASON_FINGERPRINT_MISMATCH,
    REASON_INPUT_SIGNATURE_MISMATCH,
    REASON_MODEL_MISMATCH,
    REASON_NOT_PROMOTED,
    REASON_OUTPUT_SIGNATURE_MISMATCH,
    REASON_TORCH_VERSION,
    REASON_TRITON_VERSION,
    ArtifactError,
    ArtifactManifest,
    DispatchRequest,
    RuntimeProfile,
    TensorSignature,
    VersionRange,
    build_manifest,
    check_compatibility,
    load_bundles,
    load_entry_point,
    match_artifact,
    package_artifact,
    parse_version,
    verify_bundle,
)

FINGERPRINT = "0123456789abcdef0123456789abcdef"
OTHER_FINGERPRINT = "fedcba9876543210fedcba9876543210"

# A CPU-only fake kernel: no CUDA, no Triton, no model code.
FAKE_KERNEL = '''"""CPU fake kernel used by the artifact bundle tests."""


def fused_scale_add(hidden, residual, scale=2.0):
    return hidden * scale + residual
'''


def _tensor(shape, dtype="float32", device="cuda", requires_grad=False, name=""):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= max(dim, 1)
    stride.reverse()
    return {
        "shape": list(shape),
        "stride": stride,
        "dtype": dtype,
        "device_type": device,
        "requires_grad": requires_grad,
        **({"name": name} if name else {}),
    }


def _sections(**overrides):
    sections = {
        "artifact_id": "fused-scale-add-sm90",
        "operation": {
            "name": "generated_blocks_scale_add",
            "graph_fingerprint": FINGERPRINT,
            "parent_module": "transformer.blocks",
            "operations": ["aten::mul", "aten::add"],
        },
        "signature": {
            "inputs": [_tensor((2, 4), name="input_0"), _tensor((2, 4), name="input_1")],
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
                "workload_id": "fake-workload",
                "steps": 4,
                "metric": "max_abs_latent_diff",
                "value": 3e-3,
                "threshold": 1e-2,
                "passed": True,
                "baseline_ref": "runs/baseline",
                "candidate_ref": "runs/candidate",
            },
        },
        "promotion": {
            "decision": "promoted",
            "reason": "1.5x with full-generation parity",
            "decided_at": "2026-07-31T00:00:00+00:00",
            "campaign": {
                "campaign_id": "campaign-7",
                "source": "overnight-runner",
                "target_name": "blocks_scale_add",
            },
        },
    }
    for key, value in overrides.items():
        sections[key] = value
    return sections


def _subgraph_operation():
    return {
        "name": "generated_blocks_scale_add",
        "graph_fingerprint": FINGERPRINT,
        "parent_module": "transformer.blocks",
        "operations": ["aten::mul", "aten::add"],
        "target_kind": "subgraph",
        "capture_mode": "export",
        "selected_node_ids": ["n4", "n5"],
        "boundary_refs": ["p0", "n3"],
        "output_node_ids": ["n5"],
    }


def test_subgraph_rewrite_contract_round_trips(tmp_path):
    source = _payload(tmp_path)
    sections = _sections(operation=_subgraph_operation())
    document = build_manifest(source, sections)

    assert document["operation"] == _subgraph_operation()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("capture_mode", "dynamo", "requires 'export'"),
        ("selected_node_ids", [], "non-empty"),
        ("boundary_refs", ["weight.path"], "canonical executable-IR"),
        ("output_node_ids", ["n9"], "must be selected nodes"),
    ],
)
def test_subgraph_rewrite_contract_fails_closed(tmp_path, field, value, message):
    source = _payload(tmp_path)
    operation = _subgraph_operation()
    operation[field] = value

    with pytest.raises(ArtifactError, match=message):
        build_manifest(source, _sections(operation=operation))


def _payload(
    tmp_path: Path,
    *,
    kernel_source: str = FAKE_KERNEL,
    name: str = "payload",
) -> Path:
    source = tmp_path / name
    source.mkdir(parents=True, exist_ok=True)
    (source / "kernel.py").write_text(kernel_source, encoding="utf-8")
    (source / "notes.txt").write_text("graph-derived candidate\n", encoding="utf-8")
    return source


def _bundle(tmp_path: Path, *, sections=None, kernel_source=FAKE_KERNEL, name="bundle"):
    source = _payload(tmp_path, kernel_source=kernel_source, name=f"payload-{name}")
    output = tmp_path / "store" / name
    manifest = package_artifact(source, output, sections or _sections())
    return output, manifest


def _runtime(**overrides) -> RuntimeProfile:
    profile = {
        "model_id": "fake/model",
        "model_revision": "main",
        "gpu_architecture": "sm90",
        "torch_version": "2.8.0+cu128",
        "cuda_version": "12.8",
        "triton_version": "3.3.0",
        "execution_mode": "inference",
        "distributed_mode": "single",
    }
    profile.update(overrides)
    return RuntimeProfile(**profile)


def _request(manifest: ArtifactManifest, **overrides) -> DispatchRequest:
    fields = {
        "graph_fingerprint": manifest.graph_fingerprint,
        "inputs": manifest.signature.inputs,
        "outputs": manifest.signature.outputs,
        "runtime": _runtime(),
    }
    fields.update(overrides)
    return DispatchRequest(**fields)


# -- packaging and validation -------------------------------------------------


def test_package_artifact_hashes_every_payload_file(tmp_path):
    output, manifest = _bundle(tmp_path)

    assert manifest.schema_version == ARTIFACT_SCHEMA_VERSION
    assert manifest.file_paths() == ("kernel.py", "notes.txt")
    assert (output / MANIFEST_FILENAME).is_file()
    document = json.loads((output / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert document["operation"]["graph_fingerprint"] == FINGERPRINT
    assert MANIFEST_FILENAME not in [item["path"] for item in document["files"]]
    # Round-tripping the written document must reproduce the same manifest.
    assert ArtifactManifest.from_dict(document).as_dict() == manifest.as_dict()


def test_manifest_records_every_required_fact(tmp_path):
    _, manifest = _bundle(tmp_path)

    assert manifest.operation.operations == ("aten::mul", "aten::add")
    assert manifest.signature.inputs[0].shape == (2, 4)
    assert manifest.entry_point.symbol == "fused_scale_add"
    assert manifest.compatibility.gpu_architectures == ("sm90",)
    assert manifest.compatibility.torch.contains("2.8.0+cu128")
    assert manifest.compatibility.execution_modes == ("inference",)
    assert manifest.compatibility.distributed_modes == ("single",)
    assert manifest.evidence.benchmark.speedup == pytest.approx(1.5)
    assert manifest.evidence.generation.workload_id == "fake-workload"
    assert manifest.promotion.decision == "promoted"
    assert manifest.promotion.campaign.campaign_id == "campaign-7"


def test_packager_refuses_caller_supplied_file_digests(tmp_path):
    source = _payload(tmp_path)
    sections = _sections()
    sections["files"] = [{"path": "kernel.py", "sha256": "0" * 64, "bytes": 1}]

    with pytest.raises(ArtifactError, match="computed by the packager"):
        build_manifest(source, sections)


def test_packager_refuses_overwriting_without_permission(tmp_path):
    output, _ = _bundle(tmp_path)
    source = _payload(tmp_path, name="second")

    with pytest.raises(ArtifactError, match="already exists"):
        package_artifact(source, output, _sections())

    replaced = package_artifact(source, output, _sections(), overwrite=True)
    assert replaced.artifact_id == "fused-scale-add-sm90"


@pytest.mark.parametrize("relation", ["same", "output_inside_source", "source_inside_output"])
def test_packager_refuses_overlapping_source_and_output(tmp_path, relation):
    source = _payload(tmp_path)
    if relation == "same":
        output = source
    elif relation == "output_inside_source":
        output = source / "bundle"
    else:
        output = tmp_path / "outer"
        output.mkdir()
        source.rename(output / "payload")
        source = output / "payload"

    with pytest.raises(ArtifactError, match="must not overlap"):
        package_artifact(source, output, _sections(), overwrite=True)


def test_tampered_kernel_is_rejected(tmp_path):
    output, _ = _bundle(tmp_path)
    # A same-length edit: the size check cannot catch this, only the hash can.
    tampered = FAKE_KERNEL.replace("hidden * scale", "hidden + scale")
    assert len(tampered) == len(FAKE_KERNEL)
    (output / "kernel.py").write_text(tampered, encoding="utf-8")

    with pytest.raises(ArtifactError, match="does not match manifest"):
        verify_bundle(output)


def test_truncated_kernel_is_rejected_on_size(tmp_path):
    output, _ = _bundle(tmp_path)
    (output / "kernel.py").write_text("", encoding="utf-8")

    with pytest.raises(ArtifactError, match="bytes"):
        verify_bundle(output)


def test_missing_declared_file_is_rejected(tmp_path):
    output, _ = _bundle(tmp_path)
    (output / "notes.txt").unlink()

    with pytest.raises(ArtifactError, match="is missing"):
        verify_bundle(output)


def test_undeclared_file_is_rejected(tmp_path):
    output, _ = _bundle(tmp_path)
    (output / "sneaky.py").write_text("SECRET = 1\n", encoding="utf-8")

    with pytest.raises(ArtifactError, match="undeclared file"):
        verify_bundle(output)


def test_manifest_rejects_unknown_schema_version(tmp_path):
    output, _ = _bundle(tmp_path)
    path = output / MANIFEST_FILENAME
    document = json.loads(path.read_text(encoding="utf-8"))
    document["schema_version"] = ARTIFACT_SCHEMA_VERSION + 1
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ArtifactError, match="unsupported version"):
        verify_bundle(output)


def test_manifest_rejects_forbidden_content_keys(tmp_path):
    source = _payload(tmp_path)
    sections = _sections()
    sections["operation"] = dict(sections["operation"])
    sections["operation"]["weights"] = "checkpoint.safetensors"

    with pytest.raises(ArtifactError, match="forbidden"):
        build_manifest(source, sections)


def test_manifest_rejects_entry_point_outside_declared_files(tmp_path):
    source = _payload(tmp_path)
    sections = _sections(entry_point={"file": "other.py", "symbol": "fused_scale_add"})

    with pytest.raises(ArtifactError, match="not a declared file"):
        build_manifest(source, sections)


def test_manifest_rejects_traversal_paths():
    document = ArtifactManifest.from_dict(_valid_document()).as_dict()
    document["files"][0]["path"] = "../escape.py"
    document["entry_point"]["file"] = "../escape.py"

    with pytest.raises(ArtifactError, match="relative POSIX path"):
        ArtifactManifest.from_dict(document)


def test_load_bundles_reports_bad_bundles_without_hiding_good_ones(tmp_path):
    store = tmp_path / "store"
    good_source = _payload(tmp_path, name="good")
    package_artifact(good_source, store / "good", _sections())
    broken = store / "broken"
    broken.mkdir(parents=True)
    (broken / MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")

    manifests, errors = load_bundles(store)

    assert [item.artifact_id for item in manifests] == ["fused-scale-add-sm90"]
    assert len(errors) == 1
    assert "invalid JSON" in errors[0]


# -- trusted loading ----------------------------------------------------------


def test_entry_point_loads_and_runs_from_trusted_root(tmp_path):
    output, manifest = _bundle(tmp_path)

    candidate = load_entry_point(output, trusted_root=tmp_path / "store")

    assert candidate(3.0, 1.0) == pytest.approx(7.0)
    assert manifest.entry_point.symbol == "fused_scale_add"


def test_entry_point_module_names_do_not_collide_after_id_sanitizing(tmp_path):
    first_source = _payload(tmp_path, name="first")
    second_source = _payload(tmp_path, name="second")
    store = tmp_path / "store"
    first = package_artifact(
        first_source,
        store / "first",
        _sections(artifact_id="my-kernel-v1"),
    )
    second = package_artifact(
        second_source,
        store / "second",
        _sections(artifact_id="my_kernel_v1"),
    )

    first_candidate = load_entry_point(
        store / "first", trusted_root=store, manifest=first
    )
    second_candidate = load_entry_point(
        store / "second", trusted_root=store, manifest=second
    )

    assert first_candidate.__module__ != second_candidate.__module__


def test_entry_point_refuses_bundle_outside_trusted_root(tmp_path):
    output, _ = _bundle(tmp_path)
    elsewhere = tmp_path / "untrusted"
    elsewhere.mkdir()

    with pytest.raises(ArtifactError, match="outside the trusted root"):
        load_entry_point(output, trusted_root=elsewhere)


def test_entry_point_verifies_hashes_before_import(tmp_path):
    output, _ = _bundle(tmp_path)
    # A tampered kernel that would raise on import must never be imported.
    (output / "kernel.py").write_text(
        'raise SystemExit("payload executed")\n', encoding="utf-8"
    )

    with pytest.raises(ArtifactError, match="kernel.py"):
        load_entry_point(output, trusted_root=tmp_path / "store")


def test_entry_point_missing_symbol_is_reported(tmp_path):
    sections = _sections(entry_point={"file": "kernel.py", "symbol": "absent"})
    output, _ = _bundle(tmp_path, sections=sections)

    with pytest.raises(ArtifactError, match="is not defined"):
        load_entry_point(output, trusted_root=tmp_path / "store")


# -- compatibility matching ---------------------------------------------------


def test_compatible_artifact_is_selected(tmp_path):
    _, manifest = _bundle(tmp_path)

    result = match_artifact([manifest], _request(manifest))

    assert result.matched
    assert result.manifest is not None
    assert result.manifest.artifact_id == "fused-scale-add-sm90"
    assert result.rejections == ()


def test_fingerprint_mismatch_is_rejected(tmp_path):
    _, manifest = _bundle(tmp_path)

    rejection = check_compatibility(
        manifest, _request(manifest, graph_fingerprint=OTHER_FINGERPRINT)
    )

    assert rejection is not None
    assert rejection.reason == REASON_FINGERPRINT_MISMATCH


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("dtype", "bfloat16", REASON_INPUT_SIGNATURE_MISMATCH),
        ("device_type", "cpu", REASON_INPUT_SIGNATURE_MISMATCH),
        ("requires_grad", True, REASON_INPUT_SIGNATURE_MISMATCH),
    ],
)
def test_input_signature_mismatches_are_rejected(tmp_path, field, value, reason):
    _, manifest = _bundle(tmp_path)
    changed = dataclasses.replace(manifest.signature.inputs[0], **{field: value})

    rejection = check_compatibility(
        manifest, _request(manifest, inputs=(changed, manifest.signature.inputs[1]))
    )

    assert rejection is not None
    assert rejection.reason == reason


def test_shape_mismatch_is_rejected(tmp_path):
    _, manifest = _bundle(tmp_path)
    first = manifest.signature.inputs[0]
    changed = TensorSignature(
        shape=(4, 4),
        stride=first.stride,
        dtype=first.dtype,
        device_type=first.device_type,
    )

    rejection = check_compatibility(
        manifest, _request(manifest, inputs=(changed, manifest.signature.inputs[1]))
    )

    assert rejection is not None
    assert rejection.reason == REASON_INPUT_SIGNATURE_MISMATCH
    assert "4x4" in rejection.detail


def test_output_signature_mismatch_is_rejected(tmp_path):
    _, manifest = _bundle(tmp_path)
    output_signature = manifest.signature.outputs[0]
    changed = TensorSignature(
        shape=(2, 8),
        stride=(8, 1),
        dtype=output_signature.dtype,
        device_type=output_signature.device_type,
    )

    rejection = check_compatibility(manifest, _request(manifest, outputs=(changed,)))

    assert rejection is not None
    assert rejection.reason == REASON_OUTPUT_SIGNATURE_MISMATCH


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"model_id": "other/model"}, REASON_MODEL_MISMATCH),
        ({"gpu_architecture": "sm80"}, REASON_ARCHITECTURE_MISMATCH),
        ({"torch_version": "2.1.0"}, REASON_TORCH_VERSION),
        ({"torch_version": "3.1.0"}, REASON_TORCH_VERSION),
        ({"cuda_version": "11.8"}, REASON_CUDA_VERSION),
        ({"cuda_version": None}, REASON_CUDA_VERSION),
        ({"triton_version": None}, REASON_TRITON_VERSION),
        ({"execution_mode": "training"}, REASON_EXECUTION_MODE),
        ({"distributed_mode": "tensor_parallel"}, REASON_DISTRIBUTED_MODE),
    ],
)
def test_environment_mismatches_are_rejected(tmp_path, override, reason):
    _, manifest = _bundle(tmp_path)

    rejection = check_compatibility(
        manifest, _request(manifest, runtime=_runtime(**override))
    )

    assert rejection is not None
    assert rejection.reason == reason


def test_unpromoted_artifact_is_never_selected(tmp_path):
    sections = _sections()
    promotion = copy.deepcopy(sections["promotion"])
    promotion["decision"] = "quarantined"
    promotion["reason"] = "awaiting a second generation run"
    _, manifest = _bundle(tmp_path, sections=_sections(promotion=promotion))

    result = match_artifact([manifest], _request(manifest))

    assert not result.matched
    assert [item.reason for item in result.rejections] == [REASON_NOT_PROMOTED]


def test_failed_evidence_is_never_selected(tmp_path):
    sections = _sections()
    evidence = copy.deepcopy(sections["evidence"])
    evidence["generation"]["passed"] = False
    _, manifest = _bundle(tmp_path, sections=_sections(evidence=evidence))

    rejection = check_compatibility(manifest, _request(manifest))

    assert rejection is not None
    assert rejection.reason == REASON_EVIDENCE_INCOMPLETE


def test_fastest_compatible_artifact_wins(tmp_path):
    slow_sections = _sections()
    fast_sections = _sections()
    fast_sections["artifact_id"] = "fused-scale-add-fast"
    fast_evidence = copy.deepcopy(fast_sections["evidence"])
    fast_evidence["benchmark"]["speedup"] = 2.4
    fast_evidence["benchmark"]["candidate_us"] = 50.0
    fast_sections["evidence"] = fast_evidence
    _, slow = _bundle(tmp_path, sections=slow_sections, name="slow")
    _, fast = _bundle(tmp_path, sections=fast_sections, name="fast")

    result = match_artifact([slow, fast], _request(slow))

    assert result.manifest is not None
    assert result.manifest.artifact_id == "fused-scale-add-fast"
    assert [item.reason for item in result.rejections] == ["not_selected"]


def test_wildcard_compatibility_accepts_any_model(tmp_path):
    sections = _sections()
    compatibility = copy.deepcopy(sections["compatibility"])
    compatibility["model_id"] = ANY
    compatibility["model_revision"] = ANY
    compatibility["gpu_architectures"] = [ANY]
    _, manifest = _bundle(tmp_path, sections=_sections(compatibility=compatibility))

    rejection = check_compatibility(
        manifest,
        _request(
            manifest,
            runtime=_runtime(model_id="different/model", gpu_architecture="sm80"),
        ),
    )

    assert rejection is None


# -- version handling ---------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2.8.0", (2, 8, 0)),
        ("2.8.0+cu128", (2, 8, 0)),
        ("2.8.0a0", (2, 8, 0)),
        ("12.4", (12, 4)),
        ("unknown", None),
        ("", None),
    ],
)
def test_parse_version(text, expected):
    assert parse_version(text) == expected


def test_unbounded_range_accepts_missing_versions():
    unbounded = VersionRange()

    assert unbounded.contains(None)
    assert unbounded.contains("anything")
    assert unbounded.describe() == "any"


def test_bounded_range_rejects_unparseable_versions():
    bounded = VersionRange(minimum="2.4.0")

    assert not bounded.contains(None)
    assert not bounded.contains("nightly")
    assert bounded.contains("2.4")


def _valid_document() -> dict:
    """A minimal manifest document that parses, for negative-path edits."""
    sections = _sections()
    sections.update(
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "created_at": "2026-07-31T00:00:00+00:00",
            "producer": {"name": "motionkernel", "version": "1.0.0"},
            "files": [
                {"path": "kernel.py", "sha256": "a" * 64, "bytes": 10},
            ],
        }
    )
    return sections
