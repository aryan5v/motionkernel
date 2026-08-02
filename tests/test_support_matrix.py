"""CPU tests for the support-matrix evidence chain and generator."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from autokernel.support import (
    RunRecord,
    SupportEvidenceError,
    generate_matrix,
    load_run_record,
    record_from_receipt,
    write_run_record,
)
from autokernel.workload import load_workload

REPO_ROOT = Path(__file__).resolve().parents[1]


def _record_payload(**overrides):
    payload = {
        "schema": "motionkernel.support-run",
        "schema_version": 1,
        "workload_id": "ltx-t2v-480p",
        "model_id": "FastVideo/LTX2-Distilled-Diffusers",
        "family": "ltx",
        "arch": "sm100",
        "outcome": "promoted",
        "recorded_utc": "2026-08-02T01:42:30+00:00",
        "evidence": "/mnt/nfs/vlm-aryan/ltx-v1-r4-targeted-fix-20260801-203751",
    }
    payload.update(overrides)
    return payload


class TestRunRecordValidation:
    def test_round_trip(self, tmp_path: Path) -> None:
        record = RunRecord.from_dict(_record_payload())
        path = tmp_path / "record.json"
        write_run_record(record, path)
        loaded = load_run_record(path)
        assert loaded == record

    def test_outcome_vocabulary_is_closed(self) -> None:
        with pytest.raises(SupportEvidenceError):
            RunRecord.from_dict(_record_payload(outcome="green"))

    def test_capture_blocked_requires_reason(self) -> None:
        with pytest.raises(SupportEvidenceError, match="reason"):
            RunRecord.from_dict(_record_payload(outcome="capture_blocked"))

    def test_capture_blocked_with_reason_validates(self) -> None:
        record = RunRecord.from_dict(
            _record_payload(outcome="capture_blocked", reason="export failed: dynamic shape")
        )
        assert record.outcome == "capture_blocked"
        assert record.reason == "export failed: dynamic shape"

    def test_arch_must_look_like_an_arch(self) -> None:
        with pytest.raises(SupportEvidenceError):
            RunRecord.from_dict(_record_payload(arch="gb200"))

    def test_forbidden_metadata_keys_rejected(self) -> None:
        with pytest.raises(SupportEvidenceError):
            RunRecord.from_dict(_record_payload(weights="/secret/weights"))

    def test_unknown_fields_rejected(self) -> None:
        with pytest.raises(SupportEvidenceError):
            RunRecord.from_dict(_record_payload(confidence="high"))


class TestRecordFromReceipt:
    def test_promoted(self) -> None:
        record = record_from_receipt(
            {"terminal": "promoted", "message": "promoted with e2e 1.08"},
            workload_id="w", model_id="m", family="f", arch="sm100",
            evidence="/evidence", recorded_utc="2026-08-02T00:00:00+00:00",
        )
        assert record is not None and record.outcome == "promoted"

    def test_discovery_complete_with_candidates_is_candidates_found(self) -> None:
        record = record_from_receipt(
            {
                "terminal": "discovery_complete",
                "candidates": [{"fingerprint": "fp1"}],
                "completed_stages": ["baseline", "profile", "discover"],
            },
            workload_id="w", model_id="m", family="f", arch="sm90",
            evidence="/evidence", recorded_utc="2026-08-02T00:00:00+00:00",
        )
        assert record is not None
        assert record.outcome == "candidates_found"
        assert "1 candidate(s)" in (record.reason or "")

    def test_discovery_complete_without_candidates(self) -> None:
        record = record_from_receipt(
            {"terminal": "discovery_complete", "candidates": [], "message": "nothing above floor"},
            workload_id="w", model_id="m", family="f", arch="sm90",
            evidence="/evidence", recorded_utc="2026-08-02T00:00:00+00:00",
        )
        assert record is not None and record.outcome == "no_worthwhile_candidate"

    def test_failed_maps_to_capture_blocked_with_real_reason(self) -> None:
        record = record_from_receipt(
            {"terminal": "failed", "message": "profile stage failed: export refused dynamic shape"},
            workload_id="w", model_id="m", family="f", arch="sm100",
            evidence="/evidence", recorded_utc="2026-08-02T00:00:00+00:00",
        )
        assert record is not None
        assert record.outcome == "capture_blocked"
        assert record.reason == "profile stage failed: export refused dynamic shape"

    def test_budget_exhausted_produces_no_record(self) -> None:
        record = record_from_receipt(
            {"terminal": "budget_exhausted", "message": "budget exhausted"},
            workload_id="w", model_id="m", family="f", arch="sm100",
            evidence="/evidence", recorded_utc="2026-08-02T00:00:00+00:00",
        )
        assert record is None


class TestMatrixGeneration:
    def test_repo_manifests_all_validate(self) -> None:
        paths = sorted((REPO_ROOT / "workloads").glob("*.yaml"))
        assert len(paths) == 6
        families = set()
        for path in paths:
            manifest = load_workload(path)
            families.add(manifest.tags[0])
        assert families == {"ltx", "wan", "fastwan", "hunyuan", "cosmos25"}

    def test_matrix_from_repo_state(self) -> None:
        markdown, sidecar = generate_matrix(
            workloads_dir=REPO_ROOT / "workloads",
            evidence_dir=REPO_ROOT / "docs" / "support-evidence",
            today=date(2026, 8, 2),
        )
        # Six rows, two arches.
        assert len(sidecar["rows"]) == 6
        assert sidecar["arches"] == ["sm100", "sm90"]
        cells = {
            (row["workload_id"], arch): row["cells"][arch]
            for row in sidecar["rows"]
            for arch in sidecar["arches"]
        }
        assert cells[("ltx-t2v-480p", "sm100")]["outcome"] == "promoted"
        assert cells[("ltx-t2v-480p", "sm90")]["outcome"] == "not_attempted"
        assert cells[("wan-t2v-1.3b-480p", "sm100")]["outcome"] == "no_worthwhile_candidate"
        assert "1.0073x" in cells[("wan-t2v-1.3b-480p", "sm100")]["reason"]
        # Every non-not_attempted cell links to evidence with a date.
        for cell in cells.values():
            if cell["outcome"] != "not_attempted":
                assert cell["evidence"]
                assert cell["recorded_utc"]
        # The page carries the same claims.
        assert "[promoted](/mnt/nfs/vlm-aryan/ltx-v1-r4-targeted-fix-20260801-203751) 2026-08-02" in markdown
        assert markdown.count("*not_attempted*") == 9

    def test_not_attempted_never_blank(self) -> None:
        markdown, _ = generate_matrix(
            workloads_dir=REPO_ROOT / "workloads",
            evidence_dir=REPO_ROOT / "docs" / "support-evidence",
            today=date(2026, 8, 2),
        )
        for line in markdown.splitlines():
            if line.startswith("|") and "`" in line and "workload" not in line and "outcome" not in line:
                cells = [part.strip() for part in line.strip("|").split("|")]
                assert all(cells), f"blank cell in row: {line}"

    def test_staleness_marks_old_cells(self) -> None:
        markdown, sidecar = generate_matrix(
            workloads_dir=REPO_ROOT / "workloads",
            evidence_dir=REPO_ROOT / "docs" / "support-evidence",
            today=date(2026, 10, 1),
            stale_days=30,
        )
        cells = {
            (row["workload_id"], arch): row["cells"][arch]
            for row in sidecar["rows"]
            for arch in sidecar["arches"]
        }
        assert cells[("wan-t2v-1.3b-480p", "sm100")]["stale"] is True
        assert "(stale)" in markdown
        # not_attempted cells are never stale.
        assert cells[("ltx-t2v-480p", "sm90")]["stale"] is False

    def test_generation_is_deterministic(self) -> None:
        first = generate_matrix(
            workloads_dir=REPO_ROOT / "workloads",
            evidence_dir=REPO_ROOT / "docs" / "support-evidence",
            today=date(2026, 8, 2),
        )
        second = generate_matrix(
            workloads_dir=REPO_ROOT / "workloads",
            evidence_dir=REPO_ROOT / "docs" / "support-evidence",
            today=date(2026, 8, 2),
        )
        assert first == second

    def test_latest_record_wins(self, tmp_path: Path) -> None:
        evidence = tmp_path / "evidence"
        evidence.mkdir()
        old = RunRecord.from_dict(_record_payload(outcome="capture_blocked", reason="old"))
        new = RunRecord.from_dict(_record_payload(outcome="promoted", recorded_utc="2026-08-03T00:00:00+00:00"))
        write_run_record(old, evidence / "a.json")
        write_run_record(new, evidence / "b.json")
        _, sidecar = generate_matrix(
            workloads_dir=REPO_ROOT / "workloads",
            evidence_dir=evidence,
            today=date(2026, 8, 4),
        )
        cells = {
            (row["workload_id"], arch): row["cells"][arch]
            for row in sidecar["rows"]
            for arch in sidecar["arches"]
        }
        assert cells[("ltx-t2v-480p", "sm100")]["outcome"] == "promoted"
