from __future__ import annotations

import json
from pathlib import Path

from autokernel import cli


def test_top_level_help(capsys):
    assert cli.main(["--help"]) == 0
    assert "optimize" in capsys.readouterr().out


def test_version(capsys):
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip()


def test_unknown_command_fails(capsys):
    assert cli.main(["unknown"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_packaged_workloads_are_listed_and_parseable(capsys):
    assert cli.main(["workload", "list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert set(listed) == {"ltx_480p", "wan_t2v_1.3b_480p"}
    assert all(Path(path).is_file() for path in listed.values())

    assert cli.main(["workload", "path", "ltx_480p"]) == 0
    assert Path(capsys.readouterr().out.strip()).is_file()


def test_doctor_is_machine_readable_without_requiring_cuda(capsys):
    assert cli.main(["doctor"]) in {0, 1}
    report = json.loads(capsys.readouterr().out)
    assert report["schema_version"] == 1
    assert report["checks"]["python"]["ok"] is True


def test_optimize_dispatches_to_installed_command(monkeypatch):
    seen = []
    monkeypatch.setattr("autokernel.optimize.cli.main", lambda argv: seen.append(argv) or 17)
    assert cli.main(["optimize", "--help"]) == 17
    assert seen == [["--help"]]
