from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
import torch
from datasets import Dataset
from transformers.data.data_collator import default_data_collator

from llm_behavior_eval import DatasetConfig, EvaluationConfig, evaluate
from llm_behavior_eval.evaluation_utils import censorship_utils
from llm_behavior_eval.evaluation_utils.censorship_utils import (
    CCPC_DATASET_CONFIG,
    CCPC_DATASET_ID,
    CCPC_DATASET_REPOSITORY,
    CCPC_DATASET_REVISION,
    CCPC_DATASET_SPLIT,
    CCPC_EXPECTED_ROWS,
    CCPC_JUDGE_MODEL,
    CCPC_JUDGE_REPETITION_PENALTY,
    CCPC_MODEL_REPETITION_PENALTY,
    CensorshipStatus,
    load_censorship_benchmark,
    parse_censorship_judgment,
)
from llm_behavior_eval.evaluation_utils.enums import DatasetType
from llm_behavior_eval.evaluation_utils.evaluate_factory import EvaluateFactory
from llm_behavior_eval.evaluation_utils.free_text_censorship_evaluator import (
    CensorshipResponse,
    FreeTextCensorshipEvaluator,
    _CensorshipGenerationRecord,
)


def _benchmark(size: int = CCPC_EXPECTED_ROWS, **columns: list[Any]) -> Dataset:
    data: dict[str, list[Any]] = {
        "question": [f"question-{index}" for index in range(size)],
        "source_group_id": [f"group-{index}" for index in range(size)],
    }
    data.update(columns)
    return Dataset.from_dict(data)


def _patch_load_dataset(
    monkeypatch: pytest.MonkeyPatch, dataset: Dataset
) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_load_dataset(path: str, **kwargs: Any) -> Dataset:
        captured.update(path=path, **kwargs)
        return dataset

    monkeypatch.setattr(censorship_utils, "load_dataset", fake_load_dataset)
    return captured


def test_loads_exact_pinned_dataset_in_published_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _benchmark()
    captured = _patch_load_dataset(monkeypatch, expected)

    actual = load_censorship_benchmark("token")

    assert actual["source_group_id"] == expected["source_group_id"]
    assert captured == {
        "path": CCPC_DATASET_REPOSITORY,
        "name": CCPC_DATASET_CONFIG,
        "split": CCPC_DATASET_SPLIT,
        "revision": CCPC_DATASET_REVISION,
        "token": "token",
        "trust_remote_code": False,
    }


@pytest.mark.parametrize(
    ("dataset", "message"),
    [
        (_benchmark(extra=["x"] * CCPC_EXPECTED_ROWS), "columns must be exactly"),
        (_benchmark(CCPC_EXPECTED_ROWS - 1), "exactly 288 rows"),
        (
            _benchmark(
                source_group_id=["duplicate"] * CCPC_EXPECTED_ROWS,
            ),
            "must be unique",
        ),
        (
            _benchmark(question=[" "] * CCPC_EXPECTED_ROWS),
            "field 'question' must be a non-empty string",
        ),
        (
            _benchmark(source_group_id=[""] * CCPC_EXPECTED_ROWS),
            "field 'source_group_id' must be a non-empty string",
        ),
        (
            _benchmark(question=list(range(CCPC_EXPECTED_ROWS))),
            "field 'question' must be a non-empty string",
        ),
        (
            _benchmark(source_group_id=list(range(CCPC_EXPECTED_ROWS))),
            "field 'source_group_id' must be a non-empty string",
        ),
    ],
)
def test_rejects_invalid_dataset_contract(
    monkeypatch: pytest.MonkeyPatch, dataset: Dataset, message: str
) -> None:
    _patch_load_dataset(monkeypatch, dataset)
    with pytest.raises(ValueError, match=message):
        load_censorship_benchmark()


class _Tokenizer:
    name_or_path = "fake/model"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompts: list[str], **kwargs: Any) -> dict[str, list[list[int]]]:
        del kwargs
        self.prompts = prompts
        return {
            "input_ids": [[index] for index in range(len(prompts))],
            "attention_mask": [[1] for _ in prompts],
        }


class _DatasetEngine:
    is_judge = False

    def set_dataset(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def get_batch_size(self) -> int:
        return 31


def test_prepare_dataloader_never_shuffles_or_truncates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    benchmark = _benchmark()
    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.free_text_censorship_evaluator.load_censorship_benchmark",
        lambda token: benchmark,
    )
    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.free_text_censorship_evaluator.is_model_multimodal",
        lambda *args: False,
    )
    monkeypatch.setattr(
        "llm_behavior_eval.evaluation_utils.free_text_censorship_evaluator.safe_apply_chat_template",
        lambda tokenizer, messages, **kwargs: messages[0]["content"],
    )
    evaluator = FreeTextCensorshipEvaluator.__new__(FreeTextCensorshipEvaluator)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        max_samples=1,
        max_answer_tokens=8,
    )
    evaluator.dataset_config = DatasetConfig(
        file_path="chinese_censorship", dataset_type=DatasetType.BIAS
    )
    evaluator.tokenizer = cast("Any", _Tokenizer())
    evaluator.trust_remote_code = False
    evaluator.eval_engine = cast("Any", _DatasetEngine())
    evaluator.data_collator = default_data_collator

    evaluator.prepare_dataloader()

    indices = [
        int(index)
        for batch in evaluator.eval_loader
        for index in batch["ccpc_row_index"].tolist()
    ]
    assert indices == list(range(CCPC_EXPECTED_ROWS))
    assert evaluator.benchmark_questions == benchmark["question"]
    assert evaluator.num_samples == CCPC_EXPECTED_ROWS


@pytest.mark.parametrize(
    ("text", "expected"),
    [("True", True), (" false \n", False), ("True because...", None), ("", None)],
)
def test_parse_censorship_judgment(text: str, expected: bool | None) -> None:
    assert parse_censorship_judgment(text) is expected


class _RecordingEngine:
    is_judge = False

    def __init__(self) -> None:
        self.penalties: list[float] = []

    def generate_answers(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sampling_config: Any,
        repetition_penalty: float = 1.0,
    ) -> tuple[list[str], list[str | None]]:
        del attention_mask, sampling_config
        self.penalties.append(repetition_penalty)
        return ["answer"] * len(input_ids), ["stop"] * len(input_ids)


def test_model_and_judge_use_role_specific_repetition_penalties(tmp_path: Any) -> None:
    evaluator = FreeTextCensorshipEvaluator.__new__(FreeTextCensorshipEvaluator)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        judge_path_or_repo_id=CCPC_JUDGE_MODEL,
        sample_judge=False,
    )
    evaluator.dataset_config = DatasetConfig(
        file_path="chinese_censorship", dataset_type=DatasetType.BIAS
    )
    engine = _RecordingEngine()
    evaluator.eval_engine = cast("Any", engine)
    evaluator.judge_tokenizer = cast("Any", _Tokenizer())

    evaluator._generate_model_answers(torch.tensor([[1]]), torch.tensor([[1]]))
    evaluator._process_judge_prompts_batch(cast("Any", engine), ["judge prompt"])

    assert engine.penalties == [
        CCPC_MODEL_REPETITION_PENALTY,
        CCPC_JUDGE_REPETITION_PENALTY,
    ]
    provenance = evaluator._current_run_config()["ccpc_benchmark"]
    assert provenance["model_repetition_penalty"] == CCPC_MODEL_REPETITION_PENALTY
    assert provenance["judge_repetition_penalty"] == CCPC_JUDGE_REPETITION_PENALTY


def _response(index: int, status: CensorshipStatus) -> CensorshipResponse:
    verdict = status == CensorshipStatus.JUDGED_TRUE
    return {
        "source_group_id": f"group-{index}",
        "question": f"question-{index}",
        "llm_answer": "answer",
        "finish_reason": "stop",
        "judge_finish_reason": "stop",
        "judge_raw": str(verdict),
        "judge_verdict": verdict
        if status in {CensorshipStatus.JUDGED_TRUE, CensorshipStatus.JUDGED_FALSE}
        else None,
        "status": status.value,
    }


@pytest.mark.parametrize("invalid_status", [None, *list(CensorshipStatus)[2:]])
def test_metric_valid_only_for_an_all_determinate_cohort(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    invalid_status: CensorshipStatus | None,
) -> None:
    evaluator = FreeTextCensorshipEvaluator.__new__(FreeTextCensorshipEvaluator)
    evaluator.benchmark_group_ids = [
        f"group-{index}" for index in range(CCPC_EXPECTED_ROWS)
    ]
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="fake/model", results_dir=tmp_path
    )
    monkeypatch.setattr(evaluator, "get_output_dir", lambda: tmp_path)
    monkeypatch.setattr(evaluator, "get_model_slug", lambda: "model")
    monkeypatch.setattr(evaluator, "get_dataset_slug", lambda: "chinese_censorship")
    monkeypatch.setattr(evaluator, "_append_summary_row", lambda *args: None)
    responses = [
        _response(
            index,
            invalid_status
            if index == 0 and invalid_status is not None
            else CensorshipStatus.JUDGED_TRUE
            if index == 0
            else CensorshipStatus.JUDGED_FALSE,
        )
        for index in range(CCPC_EXPECTED_ROWS)
    ]

    evaluator._save_results(responses)

    metrics = pd.read_csv(tmp_path / "metrics.csv").iloc[0]
    assert bool(metrics["Metric Valid"]) is (invalid_status is None)
    if invalid_status is None:
        assert metrics["Censorship rate (%) ⬇️"] == pytest.approx(100 / 288)
    else:
        assert pd.isna(metrics["Censorship rate (%) ⬇️"])


def test_grading_preserves_raw_evidence_and_all_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = FreeTextCensorshipEvaluator.__new__(FreeTextCensorshipEvaluator)
    generation = _CensorshipGenerationRecord(
        questions=[f"q{i}" for i in range(7)],
        source_group_ids=[f"g{i}" for i in range(7)],
        answers=[f"raw-{i}" for i in range(7)],
        finish_reasons=["stop", "stop", "length", None, "stop", "stop", "stop"],
    )
    monkeypatch.setattr(evaluator, "_format_answers", lambda answers: answers)
    monkeypatch.setattr(
        evaluator,
        "_grade_batch",
        lambda *args: (
            [True, False, True, False, None],
            ["True", "False", "truncated", "unknown", "malformed"],
            ["stop", "stop", "length", None, "stop"],
        ),
    )
    captured: list[CensorshipResponse] = []
    monkeypatch.setattr(evaluator, "_save_results", captured.extend)

    evaluator._grade_impl([generation], cast("Any", object()))

    assert [item["status"] for item in captured] == [
        status.value for status in CensorshipStatus
    ]
    assert captured[0]["llm_answer"] == "raw-0"
    assert captured[6]["judge_raw"] == "malformed"


@pytest.mark.parametrize(
    "behavior", ["chinese_censorship", "CHINESE_CENSORSHIP", " chinese_censorship "]
)
def test_cli_accepts_censorship_behavior(behavior: str) -> None:
    assert evaluate._behavior_presets(behavior) == [CCPC_DATASET_ID]


def test_cli_and_factory_route_to_the_dedicated_evaluator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    captured: list[DatasetConfig] = []
    monkeypatch.setattr(
        evaluate.EvaluateFactory,
        "create_evaluator",
        lambda config, dataset: captured.append(dataset)
        or SimpleNamespace(
            started_mlflow_run=False,
            update_dataset_config=lambda dataset: None,
            generate=lambda: [],
            free_test_model=lambda: None,
            dataset_mlflow_run=nullcontext,
            get_grading_context=nullcontext,
            grade=lambda *args: None,
            cleanup=lambda error=False: None,
        ),
    )
    evaluate.main("fake/model", "chinese_censorship", output_dir=tmp_path)
    assert captured[-1].dataset_id == CCPC_DATASET_ID
    assert EvaluateFactory.get_evaluator_family(CCPC_DATASET_ID) == "censorship"
