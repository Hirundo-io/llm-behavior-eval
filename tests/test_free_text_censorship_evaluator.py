from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from datasets import Dataset

from llm_behavior_eval import DatasetConfig, EvaluationConfig
from llm_behavior_eval.evaluation_utils import base_evaluator as base_evaluator_module
from llm_behavior_eval.evaluation_utils import censorship_utils
from llm_behavior_eval.evaluation_utils import (
    free_text_censorship_evaluator as free_text_censorship_evaluator_module,
)
from llm_behavior_eval.evaluation_utils.censorship_utils import (
    CCPC_DATASET_CONFIG,
    CCPC_DATASET_ID,
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


def test_loads_ccpc_in_source_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_load_dataset(path: str, **kwargs: object) -> Dataset:
        captured.update(path=path, **kwargs)
        return _benchmark()

    monkeypatch.setattr(censorship_utils, "load_dataset", fake_load_dataset)

    actual = load_censorship_benchmark("token")

    assert actual["source_group_id"] == ["group-0", "group-1", "group-2"]
    assert captured == {
        "path": CCPC_DATASET_REPOSITORY,
        "name": CCPC_DATASET_CONFIG,
        "split": CCPC_DATASET_SPLIT,
        "token": "token",
        "trust_remote_code": False,
    }


@pytest.mark.parametrize(
    "dataset",
    [
        _benchmark(0),
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


class _StubTokenizer:
    name_or_path = "fake/model"

    def __init__(self) -> None:
        self.padding_side = "right"

    def __call__(
        self, prompts: list[str], **_kwargs: object
    ) -> dict[str, list[list[int]]]:
        return {
            "input_ids": [[1] for _ in prompts],
            "attention_mask": [[1] for _ in prompts],
        }


class _StubEvalEngine:
    """Fake transformers engine: no model load, deterministic generation."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.tokenizer = _StubTokenizer()
        self.dataset: object = None

    def set_dataset(self, dataset: object) -> None:
        self.dataset = dataset

    def get_batch_size(self) -> int:
        return 8

    def ensure_test_model_ready(self) -> None:
        return None

    def generate_answers(
        self,
        input_ids: object,
        attention_mask: object,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[list[str], list[str | None]]:
        batch_size = len(cast("list[object]", input_ids))
        return [f"answer-{i}" for i in range(batch_size)], ["stop"] * batch_size

    def free_model(self) -> None:
        return None


def _live_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dataset_config: DatasetConfig,
    snapshots: list[Dataset],
) -> tuple[FreeTextCensorshipEvaluator, list[Dataset]]:
    """Build a real FreeTextCensorshipEvaluator with the heavy engine faked out.

    Returns the evaluator and the list of snapshots served so far (in order),
    so tests can assert exactly how many times the loader was invoked.
    """
    served: list[Dataset] = []

    def fake_load_censorship_benchmark(*_args: object, **_kwargs: object) -> Dataset:
        snapshot = snapshots[len(served)]
        served.append(snapshot)
        return snapshot

    monkeypatch.setattr(
        free_text_censorship_evaluator_module,
        "load_censorship_benchmark",
        fake_load_censorship_benchmark,
    )
    monkeypatch.setattr(
        free_text_censorship_evaluator_module,
        "safe_apply_chat_template",
        lambda *_args, **_kwargs: "prompt",
    )
    monkeypatch.setattr(
        free_text_censorship_evaluator_module,
        "is_model_multimodal",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        base_evaluator_module, "TransformersEvalEngine", _StubEvalEngine
    )

    evaluator = FreeTextCensorshipEvaluator(
        EvaluationConfig(model_path_or_repo_id="fake/model", results_dir=tmp_path),
        dataset_config,
    )
    return evaluator, served


def test_grading_time_dataset_config_update_preserves_generation_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression for the within-run snapshot-consistency bug.

    The CLI generates against one prepared cohort, then calls
    `update_dataset_config` again before grading. For CCPC (no HF revision
    pin), a naive reload there could silently swap in a different remote
    snapshot than the one used for generation, desyncing grading/accounting
    from the generated cohort.
    """
    snapshot_a = _benchmark(2)  # question-0/1, group-0/1
    snapshot_b = Dataset.from_dict(
        {
            "question": ["other-question-0", "other-question-1"],
            "source_group_id": ["other-group-0", "other-group-1"],
        }
    )
    dataset_config = DatasetConfig(
        file_path=CCPC_DATASET_ID, dataset_type=DatasetType.BIAS
    )

    evaluator, served = _live_evaluator(
        monkeypatch, tmp_path, dataset_config, [snapshot_a, snapshot_b]
    )
    assert evaluator.benchmark_group_ids == ["group-0", "group-1"]

    generation_records = cast(
        "list[_CensorshipGenerationRecord]", list(evaluator.generate())
    )
    assert len(served) == 1
    assert all(
        row_id in {"group-0", "group-1"} for row_id in generation_records[0].row_ids
    )

    # The CLI re-applies the (unchanged) dataset config before grading.
    evaluator.update_dataset_config(
        DatasetConfig(file_path=CCPC_DATASET_ID, dataset_type=DatasetType.BIAS)
    )

    # Snapshot B must never have been fetched, and the validated cohort from
    # generation must still be the one used for grading/accounting.
    assert len(served) == 1
    assert evaluator.benchmark_group_ids == ["group-0", "group-1"]
    assert evaluator.benchmark_questions == ["question-0", "question-1"]

    captured: list[dict[str, object]] = []
    monkeypatch.setattr(evaluator, "_save_results", captured.extend)
    monkeypatch.setattr(
        evaluator,
        "_grade_batch",
        lambda _judge, questions, answers: (
            [True] * len(questions),
            ["True"] * len(questions),
            ["stop"] * len(questions),
        ),
    )
    evaluator._grade_impl(generation_records, judge_engine=cast("Any", object()))

    assert [response["source_group_id"] for response in captured] == [
        "group-0",
        "group-1",
    ]


def test_update_dataset_config_reloads_for_a_different_dataset_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Switching to a genuinely different dataset must not reuse the old snapshot."""
    snapshot_a = _benchmark(2)
    snapshot_b = Dataset.from_dict(
        {
            "question": ["other-question-0"],
            "source_group_id": ["other-group-0"],
        }
    )
    dataset_config = DatasetConfig(
        file_path=CCPC_DATASET_ID, dataset_type=DatasetType.BIAS
    )

    evaluator, served = _live_evaluator(
        monkeypatch, tmp_path, dataset_config, [snapshot_a, snapshot_b]
    )
    assert len(served) == 1

    evaluator.update_dataset_config(
        DatasetConfig(
            file_path="some/other-dataset",
            dataset_id="some/other-dataset",
            dataset_type=DatasetType.BIAS,
        )
    )

    assert len(served) == 2
    assert evaluator.benchmark_group_ids == ["other-group-0"]
