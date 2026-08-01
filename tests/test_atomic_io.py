"""Atomic workspace writes fail cleanly."""

from __future__ import annotations

import pytest

from autokernel import _io


def test_atomic_text_replaces_destination_without_temporary_files(tmp_path):
    destination = tmp_path / "state.json"
    destination.write_text("old", encoding="utf-8")

    written = _io.write_text_atomic(destination, "new")

    assert written == destination
    assert destination.read_text(encoding="utf-8") == "new"
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_text_cleans_up_after_replace_failure(tmp_path, monkeypatch):
    destination = tmp_path / "state.json"
    destination.write_text("old", encoding="utf-8")

    def fail_replace(*args, **kwargs):
        raise OSError("replace failed")

    monkeypatch.setattr(_io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        _io.write_text_atomic(destination, "new")

    assert destination.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))
