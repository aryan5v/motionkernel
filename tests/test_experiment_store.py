"""CPU tests for the experiment store.

The store's job is to make published numbers traceable. Each test below
corresponds to a way it could hold something that looks like evidence and is
not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autokernel.store import (
    ExperimentRow,
    StoreError,
    connect,
    ingest_row,
    read_document,
    row_digest,
)


def _row(**overrides) -> ExperimentRow:
    values = {
        "kind": "attention_campaign",
        "source_path": "/evidence/run-1/campaign.json",
        "raw": {"workload_id": "w", "speedup_median": 0.8},
        "workload_id": "w",
        "arch": "sm100",
    }
    values.update(overrides)
    return ExperimentRow(**values)


# -- provenance is mandatory --------------------------------------------


def test_a_row_without_a_source_path_is_refused() -> None:
    """An untraceable row is worse than a missing one: it looks like evidence."""
    with pytest.raises(StoreError, match="source_path is required"):
        _row(source_path="")


def test_a_row_without_a_kind_is_refused() -> None:
    with pytest.raises(StoreError, match="kind"):
        _row(kind="")


def test_raw_must_be_the_source_mapping() -> None:
    with pytest.raises(StoreError, match="raw"):
        _row(raw="not a mapping")


# -- idempotent ingest ---------------------------------------------------


def test_reingesting_the_same_document_is_a_no_op(tmp_path: Path) -> None:
    """A backfill must be safe to re-run and a nightly need not track state."""
    connection = connect(tmp_path / "s.sqlite")
    row = _row()
    assert ingest_row(connection, row) is True
    assert ingest_row(connection, row) is False
    count = connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    assert count == 1


def test_the_same_record_from_two_paths_is_two_rows(tmp_path: Path) -> None:
    """Two pieces of provenance, not one -- collapsing them hides a move."""
    connection = connect(tmp_path / "s.sqlite")
    ingest_row(connection, _row(source_path="/a/campaign.json"))
    ingest_row(connection, _row(source_path="/b/campaign.json"))
    count = connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    assert count == 2


def test_a_changed_document_is_a_new_row(tmp_path: Path) -> None:
    """Re-measuring must not silently overwrite the earlier measurement."""
    connection = connect(tmp_path / "s.sqlite")
    ingest_row(connection, _row(raw={"speedup_median": 0.80}))
    ingest_row(connection, _row(raw={"speedup_median": 0.52}))
    count = connection.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    assert count == 2


def test_digest_is_stable_across_key_order() -> None:
    a = row_digest({"b": 2, "a": 1}, "/p")
    b = row_digest({"a": 1, "b": 2}, "/p")
    assert a == b


# -- the raw document survives ------------------------------------------


def test_the_source_document_is_preserved_verbatim(tmp_path: Path) -> None:
    """Extraction is lossy and these schemas still move.

    A row whose source is kept can be re-extracted when the reader improves;
    one whose source was discarded cannot.
    """
    connection = connect(tmp_path / "s.sqlite")
    raw = {"workload_id": "w", "nested": {"kept": [1, 2, 3]}, "unknown_future": 7}
    ingest_row(connection, _row(raw=raw))
    stored = connection.execute("SELECT raw FROM experiments").fetchone()[0]
    assert json.loads(stored) == raw


# -- readers -------------------------------------------------------------


def _write(tmp_path: Path, name: str, doc: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_reads_a_dispatch_measurement(tmp_path: Path) -> None:
    path = _write(tmp_path, "m.json", {
        "schema": "motionkernel.dispatch-measurement",
        "workload_id": "ltx-t2v-480p", "arch": "sm100",
        "e2e": {"speedup_median": 1.1155, "speedup_min_to_min": 1.0276},
        "variance": {"native_cv": 0.0574, "candidate_cv": 0.0158,
                     "valid_for_gating": False},
    })
    row = read_document(path)
    assert row is not None and row.kind == "dispatch_measurement"
    assert row.speedup_median == pytest.approx(1.1155)
    assert row.native_cv == pytest.approx(0.0574)
    assert row.valid_for_gating is False


def test_reads_an_attention_campaign_and_derives_the_verdict(tmp_path: Path) -> None:
    path = _write(tmp_path, "campaign.json", {
        "workload_id": "wan-t2v-1.3b-480p-attention",
        "arms": {
            "native": {"median": 15.1683, "stdev": 0.0054},
            "optimized": {"median": 18.8880, "stdev": 0.0747},
        },
        "speedup_median": 0.8031, "min_end_to_end_speedup": 1.3,
        "budget": {"tier": "perceptual"},
        "fidelity": {"evidence": {"ssim": 0.9353, "lpips": 0.0378},
                     "verdict": {"passed": False}},
    })
    row = read_document(path)
    assert row is not None and row.kind == "attention_campaign"
    # Slower than the gate and outside the quality budget: rejected.
    assert row.verdict == "rejected"
    assert row.ssim == pytest.approx(0.9353)
    # CV derived from the arm, so the variance ceiling can be applied later.
    assert row.native_cv == pytest.approx(0.0054 / 15.1683)


def test_reads_a_support_run(tmp_path: Path) -> None:
    path = _write(tmp_path, "s.json", {
        "schema": "motionkernel.support-run",
        "workload_id": "wan-t2v-1.3b-480p", "family": "wan", "arch": "sm100",
        "outcome": "no_worthwhile_candidate", "evidence": "/e/run-1041",
    })
    row = read_document(path)
    assert row is not None and row.verdict == "no_worthwhile_candidate"
    assert "1041" in row.slurm_job_ids


def test_an_unrecognized_document_returns_none_rather_than_raising(
    tmp_path: Path,
) -> None:
    """Intermediates are not results; ingest reports them, it does not crash."""
    path = _write(tmp_path, "timing.json", {"phases": {"a": 1}})
    assert read_document(path) is None


def test_malformed_json_does_not_abort_a_backfill(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    assert read_document(path) is None
