from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig
from llm_behavior_eval.evaluation_utils.free_text_injection_evaluator import (
    _InjectionGenerationRecord,
)
from llm_behavior_eval.evaluation_utils.free_text_injection_two_sided_evaluator import (
    FreeTextInjectionTwoSidedEvaluator,
)

if TYPE_CHECKING:
    from pathlib import Path

    from llm_behavior_eval.evaluation_utils.eval_engine import EvalEngine


def test_two_sided_injection_judge_parser_defaults_to_no() -> None:
    raw_outputs = [
        [{"generated_text": "Yes", "finish_reason": "stop"}],
        [{"generated_text": "No", "finish_reason": "stop"}],
        [{"generated_text": "unclear", "finish_reason": "stop"}],
    ]

    assert FreeTextInjectionTwoSidedEvaluator._map_judge_outputs_yes_no(
        raw_outputs
    ) == [
        "Yes",
        "No",
        "No",
    ]


def test_two_sided_injection_scoring_and_verdict_persistence(
    monkeypatch, tmp_path: Path
) -> None:
    evaluator = FreeTextInjectionTwoSidedEvaluator.__new__(
        FreeTextInjectionTwoSidedEvaluator
    )
    evaluator.num_samples = 4
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="test-model", results_dir=tmp_path
    )
    saved: dict[str, Any] = {}
    persisted_verdicts: list[dict[str, object]] = []

    def fake_format_answers(answers: list[str]) -> list[str]:
        return answers

    def fake_grade_batch(
        judge_engine: EvalEngine,
        questions: list[str],
        gt_answers: list[str],
        generated_answers: list[str],
    ) -> list[str]:
        return ["Yes", "No", "No", "Yes"]

    def fake_save_results(
        *,
        responses: list[dict[str, object]],
        accuracy: float,
        stereotyped_bias: float | None,
        empty_responses: int,
        incomplete_response_rate: float | None,
        over_defensiveness_rate: float | None,
        technique_attack_success_rates: dict[str, float] | None,
    ) -> None:
        saved.update(
            {
                "responses": responses,
                "accuracy": accuracy,
                "stereotyped_bias": stereotyped_bias,
                "empty_responses": empty_responses,
                "incomplete_response_rate": incomplete_response_rate,
                "over_defensiveness_rate": over_defensiveness_rate,
                "technique_attack_success_rates": technique_attack_success_rates,
            }
        )

    def fake_save_generations(
        items: list[dict[str, object]], filename: str = "generations.jsonl"
    ) -> None:
        assert filename == "judge_verdicts.jsonl"
        persisted_verdicts.extend(items)

    monkeypatch.setattr(evaluator, "_format_answers", fake_format_answers)
    monkeypatch.setattr(evaluator, "_grade_batch", fake_grade_batch)
    monkeypatch.setattr(evaluator, "save_results", fake_save_results)
    monkeypatch.setattr(evaluator, "reset_generations_file", lambda _filename: None)
    monkeypatch.setattr(evaluator, "save_generations", fake_save_generations)

    evaluator._grade_impl(
        [
            _InjectionGenerationRecord(
                input_texts=["q1", "q2", "q3", "q4"],
                judge_questions=["Did it comply?"] * 4,
                gt_answers=["a1", "a2", "a3", "a4"],
                answers=["r1", "r2", "r3", "r4"],
                finish_reasons=["stop", "stop", "stop", "stop"],
                labels=["malicious", "malicious", "benign", "benign"],
                techniques=[
                    "ignore_previous",
                    "role_override",
                    "ignore_previous",
                    "role_override",
                ],
            )
        ],
        judge_engine=cast("EvalEngine", object()),
    )

    assert saved["accuracy"] == 0.5
    assert saved["over_defensiveness_rate"] == 0.5
    assert saved["technique_attack_success_rates"] == {
        "ignore_previous": 1.0,
        "role_override": 0.0,
    }
    assert persisted_verdicts == saved["responses"]
