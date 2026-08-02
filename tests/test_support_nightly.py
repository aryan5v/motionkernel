"""CPU tests for the discovery-only nightly runner (simulated stages)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autokernel.support import NightlyConfig, NightlyTarget, load_run_record, run_nightly
from conftest import make_fastvideo_checkout, make_workload


def _config(
    tmp_path: Path,
    repo_root: Path,
    *,
    workload_name: str = "workload.json",
    date: str = "2026-08-02",
) -> NightlyConfig:
    checkout = make_fastvideo_checkout(tmp_path)
    workload = make_workload(tmp_path / workload_name)
    return NightlyConfig(
        fastvideo_checkout=checkout,
        targets=(NightlyTarget(workload=workload),),
        nightly_root=tmp_path / "nightly",
        records_dir=tmp_path / "records",
        arch="sm100",
        budget_hours=1.0,
        date=date,
        repo_root=repo_root,
    )


def _simulate(monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    monkeypatch.setenv("MOTIONKERNEL_SIMULATE", "1")
    monkeypatch.setenv("MOTIONKERNEL_SIMULATE_OUTCOME", outcome)


def test_nightly_writes_candidates_found_record(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _simulate(monkeypatch, "promoted")
    config = _config(tmp_path, repo_root)

    report = run_nightly(config)

    (result,) = report["results"]
    assert result["terminal"] == "discovery_complete"
    record_path = Path(result["record_written"])
    record = load_run_record(record_path)
    assert record.outcome == "candidates_found"
    assert record.arch == "sm100"
    assert "discovery-only" in (record.reason or "")
    # Evidence layout: one immutable campaign directory for the night.
    assert result["campaign_dir"] == str(tmp_path / "nightly" / "2026-08-02" / record.workload_id)
    assert (tmp_path / "nightly" / "2026-08-02" / "nightly_report.json").is_file()


def test_nightly_is_idempotent_within_a_night(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _simulate(monkeypatch, "promoted")
    config = _config(tmp_path, repo_root)
    first = run_nightly(config)
    second = run_nightly(config)

    assert first["results"][0]["terminal"] == "discovery_complete"
    assert second["results"][0]["terminal"] == "discovery_complete"
    # The rerun returned the terminal receipt; no stage ran past discover.
    campaign = Path(first["results"][0]["campaign_dir"])
    assert not (campaign / "stages" / "specgen" / "result.json").exists()


def test_nightly_failure_records_capture_blocked_with_reason(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _simulate(monkeypatch, "fail_at:profile")
    config = _config(tmp_path, repo_root)

    report = run_nightly(config)

    (result,) = report["results"]
    record = load_run_record(result["record_written"])
    assert record.outcome == "capture_blocked"
    assert record.reason
    assert "profile" in record.reason


def test_nightly_no_worthwhile_candidate(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _simulate(monkeypatch, "no_worthwhile_candidate")
    config = _config(tmp_path, repo_root)

    report = run_nightly(config)

    (result,) = report["results"]
    record = load_run_record(result["record_written"])
    assert record.outcome == "no_worthwhile_candidate"


def test_nightly_new_night_gets_new_immutable_dir(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _simulate(monkeypatch, "promoted")
    first_config = _config(tmp_path, repo_root, date="2026-08-02")
    run_nightly(first_config)
    second_config = _config(tmp_path, repo_root, date="2026-08-03")
    run_nightly(second_config)

    assert (tmp_path / "nightly" / "2026-08-02").is_dir()
    assert (tmp_path / "nightly" / "2026-08-03").is_dir()
    # Both nights keep their own campaign evidence; neither was rewritten.
    assert (tmp_path / "nightly" / "2026-08-02" / "nightly_report.json").is_file()
    assert (tmp_path / "nightly" / "2026-08-03" / "nightly_report.json").is_file()


def test_nightly_requires_targets(tmp_path: Path, repo_root: Path) -> None:
    from autokernel.support import NightlyError

    config = NightlyConfig(
        fastvideo_checkout=tmp_path,
        targets=(),
        nightly_root=tmp_path / "nightly",
        records_dir=tmp_path / "records",
        arch="sm100",
    )
    with pytest.raises(NightlyError):
        run_nightly(config)
