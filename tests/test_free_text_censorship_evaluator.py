from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from datasets import Dataset

from llm_behavior_eval import DatasetConfig, EvaluationConfig
from llm_behavior_eval.evaluation_utils import censorship_utils
from llm_behavior_eval.evaluation_utils.censorship_utils import (
    CCPC_DATASET_CONFIG,
    CCPC_DATASET_REPOSITORY,
    CCPC_DATASET_SPLIT,
    CensorshipStatus,
    load_censorship_benchmark,
    parse_censorship_judgment,
)
from llm_behavior_eval.evaluation_utils.enums import DatasetType
from llm_behavior_eval.evaluation_utils.free_text_censorship_evaluator import (
    FreeTextCensorshipEvaluator,
    _CensorshipGenerationRecord,
)


def _benchmark(size: int = 3) -> Dataset:
    return Dataset.from_dict(
        {
            "question": [f"question-{index}" for index in range(size)],
            "source_group_id": [f"group-{index}" for index in range(size)],
        }
    )


def test_loads_ccpc_with_the_supplied_dataset_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_load_dataset(path: str, **kwargs: object) -> Dataset:
        captured.update(path=path, **kwargs)
        return _benchmark()

    monkeypatch.setattr(censorship_utils, "load_dataset", fake_load_dataset)

    actual = load_censorship_benchmark("token", dataset_revision="dataset-sha")

    assert actual["source_group_id"] == ["group-0", "group-1", "group-2"]
    assert captured == {
        "path": CCPC_DATASET_REPOSITORY,
        "name": CCPC_DATASET_CONFIG,
        "split": CCPC_DATASET_SPLIT,
        "revision": "dataset-sha",
        "token": "token",
        "trust_remote_code": False,
    }


@pytest.mark.parametrize(
    "dataset",
    [
        Dataset.from_dict({"question": ["question"]}),
        Dataset.from_dict({"question": ["question"], "source_group_id": [""]}),
        Dataset.from_dict(
            {
                "question": ["one", "two"],
                "source_group_id": ["duplicate", "duplicate"],
            }
        ),
    ],
)
def test_load_censorship_benchmark_rejects_invalid_intrinsic_schema(
    monkeypatch: pytest.MonkeyPatch, dataset: Dataset
) -> None:
    monkeypatch.setattr(
        censorship_utils, "load_dataset", lambda *_args, **_kwargs: dataset
    )

    with pytest.raises(ValueError):
        load_censorship_benchmark()


@pytest.mark.parametrize(
    ("text", "expected"),
    [("True", True), (" false\n", False), ("true.", None), ("", None)],
)
def test_parse_censorship_judgment(text: str, expected: bool | None) -> None:
    assert parse_censorship_judgment(text) is expected


@pytest.mark.parametrize(
    ("model_finish", "judge_finish", "verdict", "expected"),
    [
        ("stop", "stop", True, CensorshipStatus.JUDGED_TRUE),
        ("stop", "stop", False, CensorshipStatus.JUDGED_FALSE),
        ("length", "stop", True, CensorshipStatus.MODEL_INCOMPLETE),
        (None, "stop", True, CensorshipStatus.MODEL_UNKNOWN),
        ("stop", "length", True, CensorshipStatus.JUDGE_INCOMPLETE),
        ("stop", None, True, CensorshipStatus.JUDGE_UNKNOWN),
        ("stop", "stop", None, CensorshipStatus.JUDGE_UNPARSEABLE),
    ],
)
def test_status_is_exhaustive(
    model_finish: str | None,
    judge_finish: str | None,
    verdict: bool | None,
    expected: CensorshipStatus,
) -> None:
    assert (
        FreeTextCensorshipEvaluator._status(model_finish, judge_finish, verdict)
        == expected
    )


def _evaluator(tmp_path: Path) -> FreeTextCensorshipEvaluator:
    evaluator = FreeTextCensorshipEvaluator.__new__(FreeTextCensorshipEvaluator)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="fake/model", results_dir=tmp_path
    )
    evaluator.dataset_config = DatasetConfig(
        file_path="chinese_censorship", dataset_type=DatasetType.BIAS
    )
    evaluator.benchmark_group_ids = ["group-0", "group-1"]
    evaluator.get_output_dir = lambda: tmp_path
    evaluator.get_model_slug = lambda: "model"
    evaluator.get_dataset_slug = lambda: "chinese_censorship"
    evaluator._append_summary_row = cast("Any", lambda *_args: None)
    return evaluator


def test_save_results_requires_a_complete_determinate_cohort(tmp_path: Path) -> None:
    evaluator = _evaluator(tmp_path)
    evaluator._save_results(
        [
            {
                "source_group_id": "group-0",
                "question": "one",
                "llm_answer": "one",
                "finish_reason": "stop",
                "judge_finish_reason": "stop",
                "judge_raw": "True",
                "judge_verdict": True,
                "status": CensorshipStatus.JUDGED_TRUE.value,
            },
            {
                "source_group_id": "group-1",
                "question": "two",
                "llm_answer": "two",
                "finish_reason": "length",
                "judge_finish_reason": None,
                "judge_raw": "",
                "judge_verdict": None,
                "status": CensorshipStatus.MODEL_INCOMPLETE.value,
            },
        ]
    )

    metrics = pd.read_csv(tmp_path / "metrics.csv").iloc[0]
    assert bool(metrics["Metric Valid"]) is False
    assert metrics["Expected Samples"] == 2
    assert pd.isna(metrics["Censorship rate (%) ⬇️"])


def test_grading_preserves_answers_and_skips_noncompleted_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evaluator = _evaluator(tmp_path)
    graded: list[tuple[list[str], list[str]]] = []

    def fake_grade_batch(
        _judge: object, questions: list[str], answers: list[str]
    ) -> tuple[list[bool | None], list[str], list[str | None]]:
        graded.append((questions, answers))
        return [True], ["True"], ["stop"]

    monkeypatch.setattr(evaluator, "_grade_batch", fake_grade_batch)
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(evaluator, "_save_results", captured.extend)
    generation = _CensorshipGenerationRecord(
        questions=["one", "two"],
        row_ids=["group-0", "group-1"],
        answers=["answer-one", "answer-two"],
        finish_reasons=["stop", "length"],
    )

    evaluator._grade_impl([generation], judge_engine=cast("Any", object()))

    assert graded == [(["one"], ["answer-one"])]
    assert captured[0]["status"] == CensorshipStatus.JUDGED_TRUE.value
    assert captured[1]["status"] == CensorshipStatus.MODEL_INCOMPLETE.value
