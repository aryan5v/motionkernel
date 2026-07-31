"""CPU contract tests for the resumable V1 optimize control plane."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from autokernel.optimize import OptimizeConfig, OptimizeError, PIPELINE_STAGES, run_optimize
from autokernel.optimize.runner import _decide_terminal
from autokernel.optimize.stages import _load_stage_result


def _config(tmp_path: Path, repo_root: Path, **overrides) -> OptimizeConfig:
    checkout = tmp_path / "FastVideo"
    checkout.mkdir(exist_ok=True)
    workload = tmp_path / "workload.yaml"
    workload.write_text("schema_version: 1\nname: cpu-contract\n", encoding="utf-8")
    values = {
        "fastvideo_checkout": checkout,
        "model": "test/model",
        "workload": workload,
        "output": tmp_path / "run",
        "budget_hours": 1.0,
        "repo_root": repo_root,
    }
    values.update(overrides)
    return OptimizeConfig(**values)


def _simulate(monkeypatch: pytest.MonkeyPatch, outcome: str) -> None:
    monkeypatch.setenv("MOTIONKERNEL_SIMULATE", "1")
    monkeypatch.setenv("MOTIONKERNEL_SIMULATE_OUTCOME", outcome)


def test_promoted_campaign_writes_receipt_state_report_and_preserves_candidate(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch, "promoted")
    config = _config(tmp_path, repo_root)

    receipt = run_optimize(config)

    assert receipt["terminal"] == "promoted"
    assert receipt["completed_stages"] == list(PIPELINE_STAGES)
    assert receipt["candidates"][0]["fingerprint"] == "fp_toy_001"
    assert receipt["candidates"][0]["status"] == "promoted"
    assert (config.output / "receipt.json").is_file()
    assert (config.output / "morning_report.md").is_file()
    assert (config.output / "artifacts" / "manifest.json").is_file()
    report = (config.output / "morning_report.md").read_text(encoding="utf-8")
    assert "end-to-end" in report
    assert "Isolated operator speedup alone **never** promotes" in report


def test_no_worthwhile_candidate_stops_after_discovery(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch, "no_worthwhile_candidate")
    config = _config(tmp_path, repo_root)

    receipt = run_optimize(config)

    assert receipt["terminal"] == "no_worthwhile_candidate"
    assert receipt["completed_stages"] == ["baseline", "profile", "discover"]
    assert not (config.output / "stages" / "search").exists()


def test_isolated_speedup_cannot_promote_without_e2e_improvement():
    state = {"candidates": [{"fingerprint": "candidate"}]}
    terminal, message = _decide_terminal(
        state,
        {
            "isolated_validate": {"metrics": {"isolated_speedup": 20.0}},
            "end_to_end_validate": {
                "recommendation": "promoted",
                "metrics": {
                    "end_to_end_speedup": 1.0,
                    "classification": "neutral",
                },
            },
        },
        min_e2e_speedup=1.01,
    )
    assert terminal == "no_worthwhile_candidate"
    assert "isolated speedup=20.0 is not sufficient" in message


@pytest.mark.parametrize("speedup", [None, "bad", float("nan"), float("inf")])
def test_non_finite_or_invalid_e2e_metrics_never_promote(speedup):
    terminal, message = _decide_terminal(
        {"candidates": [{"fingerprint": "candidate"}]},
        {
            "end_to_end_validate": {
                "recommendation": "promoted",
                "metrics": {
                    "end_to_end_speedup": speedup,
                    "classification": "improved",
                },
            }
        },
        min_e2e_speedup=1.01,
    )
    assert terminal == "no_worthwhile_candidate"
    assert "promotion blocked" in message


def test_resume_skips_durable_completed_stages(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch, "promoted")
    config = _config(tmp_path, repo_root)
    run_optimize(config)
    baseline_result = config.output / "stages" / "baseline" / "result.json"
    original = baseline_result.read_bytes()

    # Model a process interruption after discovery: durable early-stage state
    # survives, while later stage records are absent and the campaign is live.
    state_path = config.output / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "running"
    state["terminal"] = None
    state["completed_stages"] = list(PIPELINE_STAGES[:3])
    state["stage_records"] = {
        name: state["stage_records"][name] for name in PIPELINE_STAGES[:3]
    }
    state["candidates"] = [
        {
            "name": "toy.elementwise",
            "fingerprint": "fp_toy_001",
            "status": "discovered",
        }
    ]
    state_path.write_text(json.dumps(state), encoding="utf-8")
    (config.output / "receipt.json").unlink()

    receipt = run_optimize(config)

    assert receipt["terminal"] == "promoted"
    assert receipt["completed_stages"] == list(PIPELINE_STAGES)
    assert baseline_result.read_bytes() == original


def test_resume_rejects_campaign_identity_drift(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch, "promoted")
    config = _config(tmp_path, repo_root)
    run_optimize(config)

    changed = _config(tmp_path, repo_root, model="different/model")
    with pytest.raises(OptimizeError, match="changed campaign configuration: model"):
        run_optimize(changed)


def test_no_resume_replaces_a_previous_terminal_campaign(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch, "promoted")
    run_optimize(_config(tmp_path, repo_root))

    _simulate(monkeypatch, "no_worthwhile_candidate")
    receipt = run_optimize(_config(tmp_path, repo_root, resume=False))

    assert receipt["terminal"] == "no_worthwhile_candidate"
    persisted = json.loads(
        (tmp_path / "run" / "receipt.json").read_text(encoding="utf-8")
    )
    assert persisted["terminal"] == "no_worthwhile_candidate"


def test_per_candidate_timeout_is_terminal_and_receipted(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch, "promoted")
    config = _config(
        tmp_path,
        repo_root,
        per_candidate_budget_seconds=0.05,
        stage_commands={
            "search": [sys.executable, "-c", "import time; time.sleep(2)"]
        },
    )

    receipt = run_optimize(config)

    assert receipt["terminal"] == "budget_exhausted"
    assert "per-candidate budget exhausted" in receipt["message"]
    assert receipt["failed_stages"]["search"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_campaign_budgets_are_rejected(
    tmp_path: Path, repo_root: Path, value: float
):
    config = _config(tmp_path, repo_root, budget_hours=value)
    with pytest.raises(OptimizeError, match="budget_hours must be finite and positive"):
        run_optimize(config)


def test_stage_command_placeholders_are_expanded(
    tmp_path: Path, repo_root: Path, monkeypatch: pytest.MonkeyPatch
):
    _simulate(monkeypatch, "promoted")
    script = (
        "import json,os,pathlib; "
        "p=pathlib.Path(os.environ['MOTIONKERNEL_RUN_DIR'])/'stages'/'baseline'/'result.json'; "
        "p.write_text(json.dumps({'schema_version':1,'stage':'baseline','status':'ok',"
        "'message':os.environ['MOTIONKERNEL_MODEL']}))"
    )
    config = _config(
        tmp_path,
        repo_root,
        stage_commands={"baseline": [sys.executable, "-c", script, "{model}"]},
    )
    receipt = run_optimize(config)
    command = json.loads(
        (config.output / "commands" / "baseline.json").read_text(encoding="utf-8")
    )
    assert command["command"][-1] == "test/model"
    assert receipt["terminal"] == "promoted"


@pytest.mark.parametrize(
    "payload,match",
    [
        (
            {"schema_version": 1, "stage": "profile", "status": "ok"},
            "identity mismatch",
        ),
        ({"schema_version": 1, "stage": "baseline"}, "invalid stage result status"),
        (
            {
                "schema_version": 1,
                "stage": "baseline",
                "status": "ok",
                "metrics": [],
            },
            "metrics must be an object",
        ),
    ],
)
def test_stage_result_contract_fails_closed(
    tmp_path: Path, payload: dict, match: str
):
    result = tmp_path / "result.json"
    result.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OptimizeError, match=match):
        _load_stage_result(result, expected_stage="baseline")
