"""Qualification identity checks; these tests do not establish live parity."""

import json
from pathlib import Path

import pytest

from scripts.parity.qualify_matrix import case_identity, immutable_json


def test_all_required_cases_have_unique_source_identities():
    cases = json.loads(Path("scripts/parity/pending.json").read_text())[
        "expected_cases"
    ]
    identities = [case_identity(case) for case in cases]
    assert len(identities) == 31
    assert len({item[1] for item in identities}) == 31
    assert case_identity("bloom:age:ambiguous") == (
        "bloom",
        "hirundo-io/bloom-age-ambiguous-bias-free-text",
        {"bias_type": "age", "kind": "ambiguous"},
    )
    assert case_identity("ccpc")[1] == "hirundo-io/ccpc-500"


def test_unknown_case_rejected():
    with pytest.raises(ValueError, match="Not a required matrix case"):
        case_identity("unqover:gender:unbias")


def test_existing_identity_cannot_change(tmp_path):
    path = tmp_path / "identity.json"
    immutable_json(path, {"samples": 2})
    immutable_json(path, {"samples": 2})
    with pytest.raises(ValueError, match="Identity changed"):
        immutable_json(path, {"samples": 500})
    assert json.loads(path.read_text()) == {"samples": 2}


def test_separate_scoring_retains_task_metrics(tmp_path, monkeypatch):
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.log import read_eval_log
    from inspect_ai.scorer import Score, accuracy, scorer
    from inspect_ai.solver import generate

    from scripts.parity import qualify_native

    @scorer(metrics=[])
    def fixed_score():
        async def score(state, target):
            return Score(value=1)

        return score

    def task_factory(**kwargs):
        return Task(
            dataset=[Sample(id="one", input="Q")],
            solver=generate(),
            scorer=fixed_score(),
            metrics=[accuracy()],
        )

    monkeypatch.setattr(qualify_native._registry, "halueval", task_factory)
    task_args = tmp_path / "task.json"
    task_args.write_text("{}")
    for phase in ["answer", "judge"]:
        monkeypatch.setattr(
            "sys.argv",
            [
                "qualify_native",
                phase,
                "--family",
                "halueval",
                "--root",
                str(tmp_path / "run"),
                "--task-args",
                str(task_args),
                "--answer",
                "mockllm/answer",
                "--judge",
                "mockllm/judge",
            ],
        )
        qualify_native.main()
    log = read_eval_log(str(tmp_path / "run/scored.eval"))
    assert log.results is not None
    assert log.results.scores[0].metrics["accuracy"].value == 1


def test_matrix_missing_results_are_not_zero(tmp_path):
    from scripts.parity.matrix_report import summarize

    report = summarize(tmp_path)
    assert report["complete"] is False
    assert len(report["unfinished"]) == 31
    assert all(
        row["metrics"] is None and row["samples"] is None for row in report["cases"]
    )
