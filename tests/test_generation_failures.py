import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from llm_behavior_eval import DatasetConfig, EvaluationConfig
from llm_behavior_eval.evaluation_utils.enums import DatasetType
from llm_behavior_eval.evaluation_utils.free_text_bias_evaluator import (
    Agreement,
    FreeTextBiasEvaluator,
    _BiasGenerationRecord,
)
from llm_behavior_eval.evaluation_utils.free_text_hallu_evaluator import (
    FreeTextHaluEvaluator,
    _HalluGenerationRecord,
)
from llm_behavior_eval.evaluation_utils.free_text_injection_evaluator import (
    FreeTextPromptInjectionEvaluator,
    _InjectionGenerationRecord,
)

if TYPE_CHECKING:
    from llm_behavior_eval.evaluation_utils.eval_engine import EvalEngine


def _evaluator(
    evaluator_class: type[Any],
    tmp_path: Path,
    dataset_id: str,
    num_samples: int,
) -> Any:
    evaluator = object.__new__(evaluator_class)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="fake/model", results_dir=tmp_path
    )
    evaluator.dataset_config = DatasetConfig(
        file_path=dataset_id, dataset_type=DatasetType.BIAS
    )
    evaluator.num_samples = num_samples
    evaluator.mlflow_config = None
    return evaluator


def _metrics(evaluator: object) -> dict[str, str]:
    output_path = cast("FreeTextHaluEvaluator", evaluator).get_output_dir()
    with (output_path / "metrics.csv").open(newline="", encoding="utf-8") as file:
        return next(csv.DictReader(file))


def _responses(evaluator: object) -> list[dict]:
    output_path = cast("FreeTextHaluEvaluator", evaluator).get_output_dir()
    return json.loads((output_path / "responses.json").read_text(encoding="utf-8"))


def test_bias_generation_failures_do_not_consume_judge_results(
    monkeypatch, tmp_path: Path
) -> None:
    evaluator = _evaluator(
        FreeTextBiasEvaluator, tmp_path, "hirundo-io/bbq-gender-bias-free-text", 2
    )
    evaluator.has_stereotype = False
    judged_answers: list[list[str]] = []

    def fake_match(_judge, answers, *_args):
        judged_answers.append(answers)
        return (
            [(Agreement.YES, Agreement.NO)],
            ["no"],
            [[{"generated_text": "gt_agree: yes"}]],
            [[{"generated_text": "candidate_uncertain: no"}]],
        )

    monkeypatch.setattr(evaluator, "_match_llm_answers", fake_match)
    evaluator._grade_impl(
        [
            _BiasGenerationRecord(
                questions=["judged", "failed"],
                answers=["answer", ""],
                correct_answers=["answer", "answer"],
                stereotyped_answers=None,
                finish_reasons=["stop", "harmony_parse_error"],
            )
        ],
        cast("EvalEngine", object()),
    )

    assert judged_answers == [["answer"]]
    responses = _responses(evaluator)
    assert responses[1]["grading_status"] == "generation_failure"
    assert responses[1]["finish_reason"] == "harmony_parse_error"
    metrics = _metrics(evaluator)
    assert metrics["Generation failures"] == "1"
    assert metrics["Generation failure rate (%) ⬇️"] == "50.000"


def test_bias_parse_error_only_persists_without_a_score(
    monkeypatch, tmp_path: Path
) -> None:
    evaluator = _evaluator(
        FreeTextBiasEvaluator, tmp_path, "hirundo-io/bbq-gender-bias-free-text", 1
    )
    evaluator.has_stereotype = False
    monkeypatch.setattr(
        evaluator,
        "_match_llm_answers",
        lambda *_args: (_ for _ in ()).throw(AssertionError("judge must not run")),
    )

    evaluator._grade_impl(
        [
            _BiasGenerationRecord(
                questions=["failed"],
                answers=[""],
                correct_answers=["answer"],
                stereotyped_answers=None,
                finish_reasons=["harmony_parse_error"],
            )
        ],
        cast("EvalEngine", object()),
    )

    metrics = _metrics(evaluator)
    assert "Error (%) ⬇️" not in metrics
    assert metrics["Generation failures"] == "1"
    assert _responses(evaluator)[0]["grading_status"] == "generation_failure"


def test_hallucination_generation_failure_is_excluded_from_accuracy(
    monkeypatch, tmp_path: Path
) -> None:
    evaluator = _evaluator(FreeTextHaluEvaluator, tmp_path, "hirundo-io/halueval", 2)
    judged_questions: list[list[str]] = []

    def fake_grade(_judge, questions, *_args):
        judged_questions.append(questions)
        return ["CORRECT"]

    monkeypatch.setattr(evaluator, "_grade_batch", fake_grade)
    evaluator._grade_impl(
        [
            _HalluGenerationRecord(
                input_texts=["judged", "failed"],
                gt_answers=["gold", "gold"],
                answers=["answer", ""],
                finish_reasons=["stop", "harmony_parse_error"],
            )
        ],
        cast("EvalEngine", object()),
    )

    assert judged_questions == [["judged"]]
    responses = _responses(evaluator)
    assert responses[1]["grade"] is None
    assert responses[1]["grading_status"] == "generation_failure"
    metrics = _metrics(evaluator)
    assert metrics["Accuracy (%) ⬆️"] == "100.000"
    assert metrics["Generation failures"] == "1"


def test_injection_generation_failure_is_excluded_from_attack_success(
    monkeypatch, tmp_path: Path
) -> None:
    evaluator = _evaluator(
        FreeTextPromptInjectionEvaluator,
        tmp_path,
        "hirundo-io/prompt-injection-purple-llama",
        2,
    )
    judged_questions: list[list[str]] = []

    def fake_grade(_judge, questions, *_args):
        judged_questions.append(questions)
        return ["No"]

    monkeypatch.setattr(evaluator, "_grade_batch", fake_grade)
    evaluator._grade_impl(
        [
            _InjectionGenerationRecord(
                input_texts=["prompt", "failed"],
                judge_questions=["judge prompt", "failed judge prompt"],
                gt_answers=["", ""],
                answers=["safe", ""],
                finish_reasons=["stop", "harmony_parse_error"],
            )
        ],
        cast("EvalEngine", object()),
    )

    assert judged_questions == [["judge prompt"]]
    responses = _responses(evaluator)
    assert responses[1]["judge"] is None
    assert responses[1]["grading_status"] == "generation_failure"
    metrics = _metrics(evaluator)
    assert metrics["Attack success rate (%) ⬇️"] == "0.000"
    assert metrics["Generation failures"] == "1"
