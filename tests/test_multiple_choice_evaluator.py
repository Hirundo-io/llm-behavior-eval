from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, cast

import pandas as pd
import pytest

pytest.importorskip("torch")
import torch

from llm_behavior_eval.evaluation_utils.multiple_choice_evaluator import (
    MultipleChoiceEvaluator,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from llm_behavior_eval.evaluation_utils.base_evaluator import _GenerationRecord
    from llm_behavior_eval.evaluation_utils.eval_engine import EvalEngine


class _DummyCounts(TypedDict):
    valid: int
    invalid: int


class _DummyMetadata(TypedDict):
    gold_label: int


class _DummyMultipleChoiceEvaluator(
    MultipleChoiceEvaluator[_DummyMetadata, _DummyCounts]
):
    def __init__(self, output_dir) -> None:
        self._output_dir = output_dir
        self.eval_loader = [
            {
                "test_input_ids": torch.tensor([[1, 2], [3, 4]]),
                "test_attention_mask": torch.tensor([[1, 1], [1, 1]]),
                "gold_labels": torch.tensor([0, 1]),
            }
        ]
        self.mlflow_config = None
        self._generate_calls: list[bool | None] = []

    def ensure_test_model_ready(self) -> None:
        return None

    def cleanup(self, error: bool = False) -> None:
        return None

    def generate_answers(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        do_sample: bool | None = None,
    ) -> list[str]:
        self._generate_calls.append(do_sample)
        return ["A", "not a choice"]

    def get_output_dir(self):
        self._output_dir.mkdir(parents=True, exist_ok=True)
        return self._output_dir

    def extract_batch_metadata(self, batch: dict[str, Any]) -> list[_DummyMetadata]:
        gold_labels_obj = batch["gold_labels"]
        if not hasattr(gold_labels_obj, "tolist"):
            raise TypeError(
                f"Expected tensor-like gold labels, got {type(gold_labels_obj)}"
            )
        # Safe cast: guarded by runtime `hasattr(..., "tolist")` check above.
        gold_labels_any = cast("Any", gold_labels_obj)
        # Safe cast: test fixture sets `gold_labels` as an integer tensor.
        gold_labels = cast("list[int]", gold_labels_any.tolist())
        return [{"gold_label": label} for label in gold_labels]

    def create_metrics_accumulator(self) -> _DummyCounts:
        return _DummyCounts(valid=0, invalid=0)

    def record_prediction(
        self,
        *,
        metrics_accumulator: _DummyCounts,
        predicted_label: int | None,
        sample_metadata: _DummyMetadata,
    ) -> None:
        if predicted_label is None:
            metrics_accumulator["invalid"] += 1
        else:
            metrics_accumulator["valid"] += 1

    def build_metrics_payload(
        self,
        metrics_accumulator: _DummyCounts,
    ) -> dict[str, float | str]:
        total = metrics_accumulator["valid"] + metrics_accumulator["invalid"]
        invalid_rate = metrics_accumulator["invalid"] / total if total else 0.0
        return {
            "dataset_type": "bias",
            "valid": float(metrics_accumulator["valid"]),
            "invalid": float(metrics_accumulator["invalid"]),
            "invalid_answer_rate": invalid_rate,
        }


def test_parse_multiple_choice_answer_is_strict() -> None:
    assert _DummyMultipleChoiceEvaluator.parse_multiple_choice_answer("A") == 0
    assert _DummyMultipleChoiceEvaluator.parse_multiple_choice_answer("  b") == 1
    assert _DummyMultipleChoiceEvaluator.parse_multiple_choice_answer("\nC") == 2
    assert _DummyMultipleChoiceEvaluator.parse_multiple_choice_answer("answer: A") == 0
    assert (
        _DummyMultipleChoiceEvaluator.parse_multiple_choice_answer("the answer is A")
        is None
    )


def test_generate_and_grade_shared_flow(tmp_path) -> None:
    evaluator = _DummyMultipleChoiceEvaluator(output_dir=tmp_path)

    generations = evaluator.generate()
    assert evaluator._generate_calls == [False]

    evaluator.grade(generations)

    responses_path = tmp_path / "responses.json"
    metrics_path = tmp_path / "metrics.csv"
    assert responses_path.exists()
    assert metrics_path.exists()

    metrics = pd.read_csv(metrics_path)
    assert metrics.loc[0, "valid"] == 1.0
    assert metrics.loc[0, "invalid"] == 1.0
    assert metrics.loc[0, "invalid_answer_rate"] == 0.5


def test_evaluate_runs_generate_free_grade_in_order(tmp_path) -> None:
    evaluator = _DummyMultipleChoiceEvaluator(output_dir=tmp_path)
    events: list[str] = []

    def _generate() -> list[_GenerationRecord]:
        events.append("generate")
        return []

    def _free_test_model() -> None:
        events.append("free_test_model")

    def _grade(
        generations: Sequence[_GenerationRecord],
        judge_engine: EvalEngine | None = None,
    ) -> None:
        _ = generations
        _ = judge_engine
        events.append("grade")

    evaluator.generate = _generate
    evaluator.free_test_model = _free_test_model
    evaluator.grade = _grade

    evaluator.evaluate()

    assert events == ["generate", "free_test_model", "grade"]
