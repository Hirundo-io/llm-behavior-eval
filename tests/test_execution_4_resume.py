"""Synthetic coverage for execution-4 resume reconciliation and skip semantics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.ccpc_bench_prereg.execution_4_resume import (
    completed_and_terminal_ids,
    prepare_resume_dir,
    reconcile_state,
    resume_generate_biased,
    select_canary_ids,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _membership(ids: list[int]) -> list[dict]:
    return [
        {
            "pool_row_id": pool_row_id,
            "cluster_id": f"c{pool_row_id}",
            "question": f"question-{pool_row_id}",
            "request_form": "other",
        }
        for pool_row_id in ids
    ]


def _config(max_attempts: int = 5) -> dict:
    return {
        "retry": {"max_attempts": max_attempts},
        "deployment": "gpt-5-bloom",
        "api_version": "2024-12-01-preview",
        "max_completion_tokens": 4096,
        "reasoning_effort": "low",
    }


def _attempt(pool_row_id: int, attempt: int, *, valid: bool) -> dict:
    return {
        "pool_row_id": pool_row_id,
        "attempt": attempt,
        "parse_valid": valid,
        "failure_reason": None if valid else "schema_or_empty_answer",
    }


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a miniature execution-4 layout."""
    interrupted = tmp_path / "full_generation"
    interrupted.mkdir()
    population = [1, 2, 3, 4]
    _write_jsonl(interrupted / "biased_membership.jsonl", _membership(population))
    (interrupted / "supervision_generator_config.json").write_text(
        json.dumps(_config()) + "\n", encoding="utf-8"
    )
    (interrupted / "supervision_generator_template.txt").write_text(
        "template\n", encoding="utf-8"
    )

    def fake_teacher_attempt(
        row: dict, _config_value: dict, _template: str
    ) -> tuple[list[dict], dict | None]:
        pool_row_id = int(row["pool_row_id"])
        record = _attempt(pool_row_id, 1, valid=True)
        return [record], {
            "answer": f"answer-{pool_row_id}",
            "attempt": 1,
            "provider": {"model": "gpt-5-2025-08-07", "choices": [{"finish_reason": "stop"}]},
        }

    import analysis.ccpc_bench_prereg.build_ccpc500_training_data as generation

    monkeypatch.setattr(generation, "teacher_attempt", fake_teacher_attempt)
    monkeypatch.setattr(
        "analysis.ccpc_bench_prereg.execution_4_resume.AMENDED_CONFIG_SHA256",
        __import__("hashlib")
        .sha256((interrupted / "supervision_generator_config.json").read_bytes())
        .hexdigest(),
    )
    monkeypatch.setattr(
        "analysis.ccpc_bench_prereg.execution_4_resume.POPULATION_SIZE",
        len(population),
    )
    return interrupted


def test_completed_valid_id_skipped(sandbox: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    interrupted = sandbox
    _write_jsonl(
        interrupted / "biased_generation_attempts.jsonl",
        [_attempt(1, 1, valid=True), _attempt(2, 1, valid=False)],
    )
    resume_dir = interrupted.parent / "resume_1"
    prepare_resume_dir(resume_dir, interrupted)

    def selective_teacher_attempt(
        row: dict, _config_value: dict, _template: str
    ) -> tuple[list[dict], dict | None]:
        pool_row_id = int(row["pool_row_id"])
        if pool_row_id == 1:
            raise AssertionError("completed valid ID must not be regenerated")
        record = _attempt(pool_row_id, 1, valid=True)
        return [record], {
            "answer": f"answer-{pool_row_id}",
            "attempt": 1,
            "provider": {"model": "gpt-5-2025-08-07", "choices": [{"finish_reason": "stop"}]},
        }

    import analysis.ccpc_bench_prereg.build_ccpc500_training_data as generation
    from types import SimpleNamespace

    stub = SimpleNamespace(
        teacher_attempt=selective_teacher_attempt,
        read_json=generation.read_json,
        read_jsonl=generation.read_jsonl,
        OUTPUT=resume_dir,
    )
    monkeypatch.setattr(
        "analysis.ccpc_bench_prereg.execution_4_resume.load_generation_module",
        lambda: stub,
    )

    summary = resume_generate_biased(
        resume_dir,
        [interrupted / "biased_generation_attempts.jsonl"],
        workers=1,
    )
    resumed_ids = {
        int(json.loads(line)["pool_row_id"])
        for line in (resume_dir / "biased_generation_attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    assert 1 not in resumed_ids
    assert summary["pending_at_launch"] == 3


def test_invalid_only_id_resumed(sandbox: Path) -> None:
    interrupted = sandbox
    _write_jsonl(
        interrupted / "biased_generation_attempts.jsonl",
        [_attempt(2, 1, valid=False)],
    )
    state = reconcile_state(
        interrupted / "biased_membership.jsonl",
        [interrupted / "biased_generation_attempts.jsonl"],
        max_attempts=5,
    )
    assert 2 in state["invalid_only_id_list"]
    assert 2 in state["unresolved_id_list"]


def test_never_attempted_id_resumed(sandbox: Path) -> None:
    interrupted = sandbox
    state = reconcile_state(
        interrupted / "biased_membership.jsonl",
        [],
        max_attempts=5,
    )
    assert state["never_attempted_id_list"] == [1, 2, 3, 4]
    assert state["accepted_valid_ids"] == 0
    assert state["unresolved_ids"] == 4


def test_duplicate_valid_id_detected(sandbox: Path) -> None:
    interrupted = sandbox
    _write_jsonl(
        interrupted / "biased_generation_attempts.jsonl",
        [_attempt(1, 1, valid=True), _attempt(1, 2, valid=True)],
    )
    state = reconcile_state(
        interrupted / "biased_membership.jsonl",
        [interrupted / "biased_generation_attempts.jsonl"],
        max_attempts=5,
    )
    assert state["duplicate_valid_ids"] == 1
    assert state["duplicate_valid_id_list"] == [1]


def test_resume_after_interrupted_write(sandbox: Path) -> None:
    interrupted = sandbox
    ledger = interrupted / "biased_generation_attempts.jsonl"
    with ledger.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(_attempt(1, 1, valid=True), sort_keys=True) + "\n")
        handle.write('{"pool_row_id": 2, "attempt": 1, "parse_valid": false\n')

    rows, malformed = __import__(
        "analysis.ccpc_bench_prereg.execution_4_resume", fromlist=["read_jsonl"]
    ).read_jsonl(ledger)
    assert len(rows) == 1
    assert malformed

    resume_dir = interrupted.parent / "resume_1"
    prepare_resume_dir(resume_dir, interrupted)
    completed, terminal = completed_and_terminal_ids([ledger], max_attempts=5)
    assert completed == {1}
    assert 2 not in completed


def test_exact_population_reconciliation(sandbox: Path) -> None:
    interrupted = sandbox
    _write_jsonl(
        interrupted / "biased_generation_attempts.jsonl",
        [
            _attempt(1, 1, valid=True),
            _attempt(2, 5, valid=False),
            _attempt(3, 1, valid=False),
        ],
    )
    state = reconcile_state(
        interrupted / "biased_membership.jsonl",
        [interrupted / "biased_generation_attempts.jsonl"],
        max_attempts=5,
    )
    assert state["accepted_valid_ids"] == 1
    assert state["exhausted_retry_budget_ids"] == 1
    assert state["invalid_only_not_exhausted_ids"] == 1
    assert state["never_attempted_ids"] == 1
    assert state["accepted_valid_ids"] + state["unresolved_ids"] == 4


def test_select_canary_ids_only_from_unresolved() -> None:
    unresolved = [10, 20, 30, 40, 50]
    selected = select_canary_ids(unresolved, 3)
    assert selected == [10, 20, 30]
