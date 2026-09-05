"""Runtime status finalization for bounded regeneration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.ccpc_bench_prereg.run_bounded_regeneration import (
    run_all,
    write_runtime_status,
)


def test_write_runtime_status_writes_json(tmp_path: Path) -> None:
    """Runtime status helper persists stable JSON."""
    status_path = tmp_path / "runtime_status.json"
    write_runtime_status(
        status_path,
        {
            "phase": "running",
            "pid": 123,
        },
    )
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "running"
    assert payload["pid"] == 123


def test_run_all_writes_completed_status_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Successful run-all must not leave authoritative status as running."""
    status_path = tmp_path / "runtime_status.json"
    calls: list[str] = []

    def record(name: str):
        def _inner(_args):
            calls.append(name)

        return _inner

    monkeypatch.setattr(
        "analysis.ccpc_bench_prereg.run_bounded_regeneration.freeze_inputs",
        record("freeze"),
    )
    monkeypatch.setattr(
        "analysis.ccpc_bench_prereg.run_bounded_regeneration.regenerate_biased",
        record("biased"),
    )
    monkeypatch.setattr(
        "analysis.ccpc_bench_prereg.run_bounded_regeneration.regenerate_normal",
        record("normal"),
    )
    monkeypatch.setattr(
        "analysis.ccpc_bench_prereg.run_bounded_regeneration.build_repaired",
        record("build"),
    )
    monkeypatch.setattr(
        "analysis.ccpc_bench_prereg.run_bounded_regeneration.reconcile_repaired",
        record("reconcile"),
    )
    monkeypatch.setattr(
        "analysis.ccpc_bench_prereg.run_bounded_regeneration.write_provenance_manifest",
        record("provenance"),
    )

    run_all(
        SimpleNamespace(
            output_dir=tmp_path,
            status_path=status_path,
        )
    )

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "completed"
    assert payload["exit_code"] == 0
    assert payload["completed_at"]
    assert calls == [
        "freeze",
        "biased",
        "normal",
        "build",
        "reconcile",
        "provenance",
    ]


def test_run_all_writes_failed_status_on_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Abnormal run-all exit must write terminal failed status with exit code."""
    status_path = tmp_path / "runtime_status.json"

    def fail_biased(_args: SimpleNamespace) -> None:
        raise RuntimeError("synthetic provider failure")

    monkeypatch.setattr(
        "analysis.ccpc_bench_prereg.run_bounded_regeneration.freeze_inputs",
        lambda _args: None,
    )
    monkeypatch.setattr(
        "analysis.ccpc_bench_prereg.run_bounded_regeneration.regenerate_biased",
        fail_biased,
    )

    with pytest.raises(RuntimeError, match="synthetic provider failure"):
        run_all(
            SimpleNamespace(
                output_dir=tmp_path,
                status_path=status_path,
            )
        )

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "failed"
    assert payload["exit_code"] == 1
    assert payload["exception_class"] == "RuntimeError"
    assert payload["completed_at"]
