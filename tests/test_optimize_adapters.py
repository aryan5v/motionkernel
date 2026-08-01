"""CPU tests for production optimize stage adapters."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autokernel.optimize.adapters import ProductionAdapterError, run_production_stage
from autokernel.optimize.search import _agent_command
from autokernel.workload.launcher import build_launcher_command


def _input(run_dir: Path, stage: str, **overrides) -> dict:
    checkout = run_dir / "FastVideo"
    checkout.mkdir(parents=True, exist_ok=True)
    workload = run_dir / "workload.yaml"
    workload.write_text("schema_version: 1\n", encoding="utf-8")
    config = {
        "fastvideo_checkout": str(checkout),
        "workload": str(workload),
        "model": "test/model",
        "baseline": "compile",
        "min_e2e_speedup": 1.01,
        "artifact_dir_name": "artifacts",
    }
    config.update(overrides)
    payload = {"stage": stage, "config": config, "state_snapshot": {}}
    path = run_dir / "stages" / stage / "input.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return config


def _prior(run_dir: Path, stage: str, **extra) -> None:
    path = run_dir / "stages" / stage / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": stage,
                "status": "ok",
                **extra,
            }
        ),
        encoding="utf-8",
    )


def test_launcher_command_supports_dedicated_profile_output(tmp_path: Path):
    command = build_launcher_command(
        python="python",
        launcher=tmp_path / "launcher.py",
        workload=tmp_path / "workload.yaml",
        mode="native",
        output_dir=tmp_path / "output",
        profile_output=tmp_path / "profile.json",
    )

    assert command[-2:] == ["--profile-output", str(tmp_path / "profile.json")]


def test_default_search_agent_sandboxes_the_editable_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo_root = tmp_path / "motionkernel"
    run_dir = tmp_path / "run"
    candidate_dir = run_dir / "candidates" / ("a" * 32)
    candidate_dir.mkdir(parents=True)
    prompt = candidate_dir / "prompt.md"
    prompt.write_text("optimize the candidate", encoding="utf-8")
    last_message = candidate_dir / "last.md"
    monkeypatch.setattr(
        "autokernel.optimize.search.shutil.which",
        lambda name: "/usr/bin/codex" if name == "codex" else None,
    )

    command = _agent_command(
        None,
        repo_root=repo_root,
        run_dir=run_dir,
        candidate_dir=candidate_dir,
        prompt_path=prompt,
        last_message=last_message,
    )

    assert command[command.index("-C") + 1] == str(candidate_dir)
    assert command[command.index("-s") + 1] == "workspace-write"
    assert str(repo_root) not in command


def test_baseline_adapter_translates_generation_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _input(tmp_path, "baseline")
    calls = []

    def fake_run_ab(**kwargs):
        calls.append(kwargs)
        return {
            "results": {
                "native": {
                    "median_wall_seconds": 4.25,
                    "runs": 3,
                }
            }
        }

    monkeypatch.setattr("autokernel.optimize.adapters.run_ab", fake_run_ab)

    result = run_production_stage("baseline", tmp_path)

    assert result["metrics"]["median_wall_seconds"] == 4.25
    assert result["metrics"]["baseline_mode"] == "compile"
    assert calls[0]["modes"] == ("native",)
    assert calls[0]["model_override"] == "test/model"


def test_profile_adapter_enables_capture_and_requires_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _input(tmp_path, "profile")
    calls = []

    def fake_run_mode(**kwargs):
        calls.append(kwargs)
        Path(kwargs["profile_output"]).write_text("{}", encoding="utf-8")

    monkeypatch.setattr("autokernel.optimize.adapters.run_mode", fake_run_mode)
    monkeypatch.setattr(
        "autokernel.optimize.adapters.load_workload",
        lambda _path: SimpleNamespace(mode_env=None),
    )
    monkeypatch.setattr(
        "autokernel.optimize.adapters.load_generation_result",
        lambda _path: SimpleNamespace(status="ok", median_wall_seconds=2.0, runs=2),
    )

    result = run_production_stage("profile", tmp_path)

    assert result["artifacts"]["profiler_export"].endswith("profiler.json")
    assert calls[0]["env"]["FASTVIDEO_OPTIMIZATION_PROFILE_CAPTURE_FX"] == "1"
    assert calls[0]["env"]["FASTVIDEO_OPTIMIZATION_PROFILE_FX_TRACER"] == "auto"


def test_discover_adapter_correlates_and_persists_ranked_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _input(tmp_path, "discover")
    profiler = tmp_path / "profiler.json"
    profiler.write_text(json.dumps({"rows": [{"name": "aten::add"}]}), encoding="utf-8")
    _prior(tmp_path, "profile", artifacts={"profiler_export": str(profiler)})
    report = SimpleNamespace(
        regions=[object()],
        total_cuda_time_us=100.0,
        graph_breaks=[],
    )
    candidate = SimpleNamespace(
        search_worthy=True,
        as_dict=lambda: {
            "name": "region",
            "fingerprint": "a" * 32,
            "search_worthy": True,
        },
    )
    written = []
    monkeypatch.setattr(
        "autokernel.optimize.adapters.load_profiler_export", lambda _path: report
    )
    monkeypatch.setattr(
        "autokernel.optimize.adapters.correlate_discovery_report",
        lambda rows, loaded: report if rows and loaded is report else None,
    )
    monkeypatch.setattr(
        "autokernel.optimize.adapters.rank_regions",
        lambda *_args, **_kwargs: (candidate,),
    )
    monkeypatch.setattr(
        "autokernel.optimize.adapters.write_discovery_report",
        lambda value, path: written.append((value, path)),
    )

    result = run_production_stage("discover", tmp_path)

    assert result["candidates"][0]["fingerprint"] == "a" * 32
    assert result["metrics"]["search_worthy_regions"] == 1
    assert written[0][0] is report


def test_specgen_adapter_generates_every_discovered_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _input(tmp_path, "specgen")
    discovery = tmp_path / "discovery.json"
    discovery.write_text("{}", encoding="utf-8")
    fingerprint = "b" * 32
    _prior(
        tmp_path,
        "discover",
        candidates=[{"name": "region", "fingerprint": fingerprint}],
        artifacts={"discovery_report": str(discovery)},
    )

    def fake_write(_report, output, *, fingerprint):
        return {"spec": Path(output) / f"{fingerprint}.py"}

    monkeypatch.setattr(
        "autokernel.optimize.adapters.write_generated_artifacts", fake_write
    )

    result = run_production_stage("specgen", tmp_path)

    assert result["metrics"]["specs_generated"] == 1
    assert result["candidates"][0]["generated"]["spec"].endswith(f"{fingerprint}.py")


def _package_request(tmp_path: Path, *, decision: str = "quarantined") -> dict:
    source = tmp_path / "payload"
    source.mkdir(exist_ok=True)
    return {
        "source_dir": str(source),
        "sections": {
            "artifact_id": "candidate-one",
            "evidence": {
                "benchmark": {"passed": True},
                "generation": {"passed": False},
            },
            "promotion": {"decision": decision},
        },
    }


def test_package_adapter_consumes_external_evidence_without_promoting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _input(tmp_path, "package")
    request = _package_request(tmp_path)
    _prior(tmp_path, "isolated_validate", package_requests=[request])
    calls = []

    def fake_package(source, output, sections, *, overwrite):
        calls.append((source, output, sections, overwrite))
        return SimpleNamespace(artifact_id=sections["artifact_id"])

    monkeypatch.setattr("autokernel.optimize.adapters.package_artifact", fake_package)

    result = run_production_stage("package", tmp_path)

    assert result["metrics"]["artifacts_packaged"] == 1
    assert calls[0][2]["promotion"]["decision"] == "quarantined"
    assert calls[0][3] is True


def test_package_adapter_rejects_premature_promotion(tmp_path: Path):
    _input(tmp_path, "package")
    request = _package_request(tmp_path, decision="promoted")
    _prior(tmp_path, "isolated_validate", package_requests=[request])

    with pytest.raises(ProductionAdapterError, match="must be 'quarantined'"):
        run_production_stage("package", tmp_path)


def test_e2e_adapter_requires_dispatch_and_real_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _input(tmp_path, "end_to_end_validate")
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    _prior(tmp_path, "package", artifacts={"root": str(artifact_root)})
    generation = tmp_path / "generation"
    generation.mkdir()
    (generation / "native_result.json").write_text("{}", encoding="utf-8")
    native = SimpleNamespace(frames_path="native.npy")
    candidate = SimpleNamespace(frames_path="candidate.npy")
    calls = []

    def fake_run_mode(**kwargs):
        calls.append(kwargs)
        diagnostics = Path(kwargs["env"]["FASTVIDEO_OPTIMIZATION_ARTIFACT_DIAGNOSTICS"])
        diagnostics.write_text(
            json.dumps({"dispatch": {"reason_counts": {"artifact_selected": 2}}}),
            encoding="utf-8",
        )

    monkeypatch.setattr("autokernel.optimize.adapters.run_mode", fake_run_mode)
    monkeypatch.setattr(
        "autokernel.optimize.adapters.load_generation_result",
        lambda path: candidate if Path(path).name.startswith("candidate") else native,
    )
    monkeypatch.setattr(
        "autokernel.optimize.adapters.load_workload",
        lambda _path: SimpleNamespace(performance=None, parity=None, mode_env=None),
    )
    monkeypatch.setattr(
        "autokernel.optimize.adapters.classify_end_to_end",
        lambda *_args, **_kwargs: {
            "classification": "improved",
            "end_to_end_speedup": 1.05,
        },
    )
    monkeypatch.setattr(
        "autokernel.optimize.adapters.compare_frame_outputs",
        lambda *_args, **_kwargs: {"passed": True, "policy": "byte_equal"},
    )

    result = run_production_stage("end_to_end_validate", tmp_path)

    assert result["recommendation"] == "promoted"
    assert result["metrics"]["artifact_selected"] is True
    assert calls[0]["mode"] == "candidate"
    assert calls[0]["env"]["FASTVIDEO_OPTIMIZATION_ARTIFACT_VALIDATION"] == "1"


def test_search_adapter_runs_builtin_agent_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _input(tmp_path, "search")
    candidate = {"fingerprint": "c" * 32}
    _prior(tmp_path, "specgen", candidates=[candidate])
    calls = []

    def fake_search(run_dir, candidates, received_config):
        calls.append((run_dir, candidates, received_config))
        return {
            "candidates": [{**candidate, "search": {"speedup": 1.2}}],
            "failures": [],
        }

    monkeypatch.setattr("autokernel.optimize.adapters.search_candidates", fake_search)

    result = run_production_stage("search", tmp_path)

    assert result["metrics"]["faster_candidates"] == 1
    assert result["candidates"][0]["search"]["speedup"] == 1.2
    assert calls == [(tmp_path, [candidate], config)]


def test_isolated_adapter_emits_measured_package_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _input(tmp_path, "isolated_validate")
    candidate = {"fingerprint": "d" * 32, "search": {"speedup": 1.2}}
    _prior(tmp_path, "search", candidates=[candidate])
    request = _package_request(tmp_path)

    monkeypatch.setattr(
        "autokernel.optimize.adapters.validate_candidates",
        lambda run_dir, candidates, received_config: {
            "candidates": [
                {
                    **candidate,
                    "validation": {
                        "speedup": 1.15,
                        "artifact_id": "candidate-one",
                    },
                }
            ],
            "package_requests": [request],
            "failures": [],
        },
    )

    result = run_production_stage("isolated_validate", tmp_path)

    assert result["metrics"]["isolated_correct"] is True
    assert result["metrics"]["isolated_speedup"] == pytest.approx(1.15)
    assert result["package_requests"] == [request]
