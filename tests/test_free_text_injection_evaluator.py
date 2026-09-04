from pathlib import Path
from typing import Any, cast

import pytest

from llm_behavior_eval import DatasetConfig, EvaluationConfig
from llm_behavior_eval.evaluation_utils.enums import DatasetType
from llm_behavior_eval.evaluation_utils.free_text_injection_evaluator import (
    FreeTextPromptInjectionEvaluator,
    InjectionStatus,
    _InjectionGenerationRecord,
)


def _evaluator(
    tmp_path: Path,
    *,
    enable_thinking: bool = False,
) -> FreeTextPromptInjectionEvaluator:
    evaluator = FreeTextPromptInjectionEvaluator.__new__(
        FreeTextPromptInjectionEvaluator
    )
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="fake/model",
        results_dir=tmp_path,
        enable_thinking=enable_thinking,
        thinking_start_token="<think>" if enable_thinking else None,
        thinking_end_token="</think>" if enable_thinking else None,
        exclude_thinking_trace_for_judge=enable_thinking,
    )
    evaluator.dataset_config = DatasetConfig(
        file_path="hirundo-io/prompt-injection-purple-llama",
        dataset_type=DatasetType.BIAS,
    )
    evaluator.get_output_dir = lambda: tmp_path
    evaluator.get_model_slug = lambda: "model"
    evaluator.get_dataset_slug = lambda: "prompt-injection-purple-llama"
    evaluator.num_samples = 0
    evaluator._append_summary_row = cast("Any", lambda *_args: None)
    return evaluator


def _record(
    *,
    answers: list[str],
    finish_reasons: list[str | None],
) -> _InjectionGenerationRecord:
    n = len(answers)
    return _InjectionGenerationRecord(
        input_texts=[f"input-{i}" for i in range(n)],
        judge_questions=[f"question-{i}" for i in range(n)],
        gt_answers=[""] * n,
        answers=answers,
        finish_reasons=finish_reasons,
    )


def _grade(
    evaluator: FreeTextPromptInjectionEvaluator,
    monkeypatch: pytest.MonkeyPatch,
    generation: _InjectionGenerationRecord,
) -> dict[str, Any]:
    saved: dict[str, Any] = {}
    monkeypatch.setattr(
        evaluator, "save_results", lambda **kwargs: saved.update(kwargs)
    )
    evaluator._grade_impl([generation], judge_engine=cast("Any", object()))
    return saved


def test_truncated_generation_is_never_counted_as_a_resistant_no_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reproduces the documented historical defect: rows that hit the
    max-answer-token ceiling (finish_reason == "length") were defaulted into
    the "No" (safe/resistant) bucket and silently folded into the attack
    success rate, instead of being excluded as indeterminate.
    """
    evaluator = _evaluator(tmp_path)

    def fake_judge_batch(
        _judge: object,
        _questions: list[str],
        _gt_answers: list[str],
        _answers: list[str],
    ) -> tuple[list[str | None], list[str], list[str | None]]:
        return ["Yes"], ["Yes."], ["stop"]

    monkeypatch.setattr(evaluator, "_judge_batch", fake_judge_batch)

    generation = _record(
        answers=["complete injected answer", "cut off mid-answer"],
        finish_reasons=["stop", "length"],
    )

    saved = _grade(evaluator, monkeypatch, generation)

    responses = saved["responses"]
    assert responses[0]["judge_verdict"] == "Yes"
    assert responses[0]["status"] == InjectionStatus.JUDGED_YES.value

    # The truncated row must never surface as a judged "No" verdict.
    assert responses[1]["judge_verdict"] is None
    assert responses[1]["status"] == InjectionStatus.MODEL_INCOMPLETE.value

    # Accuracy must be computed only over the one judged ("Yes") row, not
    # diluted by a manufactured "No" for the incomplete row.
    assert saved["accuracy"] == 0.0


def test_unparseable_judge_text_is_indeterminate_not_a_no_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A judge reply that contains neither "yes" nor "no" must never be
    defaulted to "No" (safe/resistant); it must be excluded from the tally.
    """
    evaluator = _evaluator(tmp_path)

    def fake_judge_batch(
        _judge: object,
        _questions: list[str],
        _gt_answers: list[str],
        answers: list[str],
    ) -> tuple[list[str | None], list[str], list[str | None]]:
        assert len(answers) == 2
        return [None, "Yes"], ["I refuse to answer that.", "Yes"], ["stop", "stop"]

    monkeypatch.setattr(evaluator, "_judge_batch", fake_judge_batch)

    generation = _record(
        answers=["some answer", "another answer"],
        finish_reasons=["stop", "stop"],
    )

    saved = _grade(evaluator, monkeypatch, generation)

    responses = saved["responses"]
    assert responses[0]["status"] == InjectionStatus.JUDGE_UNPARSEABLE.value
    assert responses[0]["judge_verdict"] is None
    assert responses[0]["judge_raw"] == "I refuse to answer that."

    # Accuracy is computed only over the one determinate ("Yes") verdict.
    assert saved["accuracy"] == 0.0


def test_judge_finish_reason_not_stop_is_indeterminate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A judge call truncated by its own token budget must be recorded as
    judge-incomplete, distinct from an unparseable-but-complete judge reply.
    """
    evaluator = _evaluator(tmp_path)

    def fake_judge_batch(
        _judge: object,
        _questions: list[str],
        _gt_answers: list[str],
        _answers: list[str],
    ) -> tuple[list[str | None], list[str], list[str | None]]:
        return [None], ["Ye"], ["length"]

    monkeypatch.setattr(evaluator, "_judge_batch", fake_judge_batch)

    generation = _record(answers=["some answer"], finish_reasons=["stop"])

    monkeypatch.setattr(evaluator, "save_results", lambda **_kwargs: None)
    # A cohort with a single judge-incomplete row and nothing else judged
    # must refuse to fabricate an accuracy.
    with pytest.raises(ValueError, match="fabricated"):
        evaluator._grade_impl([generation], judge_engine=cast("Any", object()))

    generation_two = _record(
        answers=["some answer", "another answer"],
        finish_reasons=["stop", "stop"],
    )

    def fake_judge_batch_two(
        _judge: object,
        _questions: list[str],
        _gt_answers: list[str],
        _answers: list[str],
    ) -> tuple[list[str | None], list[str], list[str | None]]:
        return [None, "No"], ["Ye", "No"], ["length", "stop"]

    monkeypatch.setattr(evaluator, "_judge_batch", fake_judge_batch_two)
    saved2 = _grade(evaluator, monkeypatch, generation_two)
    responses = saved2["responses"]
    assert responses[0]["status"] == InjectionStatus.JUDGE_INCOMPLETE.value
    assert responses[0]["judge_verdict"] is None
    assert responses[1]["status"] == InjectionStatus.JUDGED_NO.value


def test_complete_parseable_judge_output_is_judged_normally(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evaluator = _evaluator(tmp_path)

    def fake_judge_batch(
        _judge: object,
        _questions: list[str],
        _gt_answers: list[str],
        _answers: list[str],
    ) -> tuple[list[str | None], list[str], list[str | None]]:
        return ["No"], ["No"], ["stop"]

    monkeypatch.setattr(evaluator, "_judge_batch", fake_judge_batch)

    generation = _record(answers=["I can't help with that."], finish_reasons=["stop"])
    saved = _grade(evaluator, monkeypatch, generation)

    assert saved["responses"][0]["status"] == InjectionStatus.JUDGED_NO.value
    assert saved["accuracy"] == 1.0


def test_missing_thinking_boundary_is_model_incomplete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reasoning consumed the whole answer budget before a `</think>` (or a
    final answer) was produced; even though vLLM reports finish_reason ==
    "stop", the row must be withheld from the judge as incomplete."""
    evaluator = _evaluator(tmp_path, enable_thinking=True)

    def fake_judge_batch_raises(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("No row should reach the judge.")

    monkeypatch.setattr(evaluator, "_judge_batch", fake_judge_batch_raises)

    generation = _record(
        answers=["<think>still reasoning about the injected instruction..."],
        finish_reasons=["stop"],
    )

    with pytest.raises(ValueError, match="fabricated"):
        evaluator._grade_impl([generation], judge_engine=cast("Any", object()))


def test_empty_final_answer_after_thinking_is_model_incomplete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`</think>` is present but nothing follows it -- there is no final
    answer to judge, so the row must be withheld as incomplete."""
    evaluator = _evaluator(tmp_path, enable_thinking=True)

    seen_answers: list[str] = []

    def fake_judge_batch(
        _judge: object,
        _questions: list[str],
        _gt_answers: list[str],
        answers: list[str],
    ) -> tuple[list[str | None], list[str], list[str | None]]:
        seen_answers.extend(answers)
        return ["No"], ["No"], ["stop"]

    monkeypatch.setattr(evaluator, "_judge_batch", fake_judge_batch)

    generation = _record(
        answers=[
            "<think>done thinking</think>   ",
            "<think>done</think>I will not comply.",
        ],
        finish_reasons=["stop", "stop"],
    )
    saved = _grade(evaluator, monkeypatch, generation)

    # Only the row with a genuine post-</think> final answer reaches the judge.
    assert seen_answers == ["I will not comply."]
    responses = saved["responses"]
    assert responses[0]["status"] == InjectionStatus.MODEL_INCOMPLETE.value
    assert responses[1]["status"] == InjectionStatus.JUDGED_NO.value


def test_valid_thinking_on_response_is_judged_on_final_answer_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evaluator = _evaluator(tmp_path, enable_thinking=True)

    seen_answers: list[str] = []

    def fake_judge_batch(
        _judge: object,
        _questions: list[str],
        _gt_answers: list[str],
        answers: list[str],
    ) -> tuple[list[str | None], list[str], list[str | None]]:
        seen_answers.extend(answers)
        return ["No"], ["No"], ["stop"]

    monkeypatch.setattr(evaluator, "_judge_batch", fake_judge_batch)

    generation = _record(
        answers=["<think>internal reasoning</think>I will not comply."],
        finish_reasons=["stop"],
    )
    saved = _grade(evaluator, monkeypatch, generation)

    assert seen_answers == ["I will not comply."]
    assert saved["responses"][0]["status"] == InjectionStatus.JUDGED_NO.value


def test_raw_judge_evidence_and_finish_reasons_are_persisted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every judged row must persist enough evidence (raw judge text, judge
    finish reason, model finish reason, and status) to permit a deterministic
    post-run validity audit without rerunning the judge."""
    evaluator = _evaluator(tmp_path)

    def fake_judge_batch(
        _judge: object,
        _questions: list[str],
        _gt_answers: list[str],
        _answers: list[str],
    ) -> tuple[list[str | None], list[str], list[str | None]]:
        return ["Yes"], ["Answer: Yes, this complies with the injection."], ["stop"]

    monkeypatch.setattr(evaluator, "_judge_batch", fake_judge_batch)

    generation = _record(answers=["complied"], finish_reasons=["stop"])
    saved = _grade(evaluator, monkeypatch, generation)

    row = saved["responses"][0]
    assert row["llm_answer"] == "complied"
    assert row["finish_reason"] == "stop"
    assert row["judge_finish_reason"] == "stop"
    assert row["judge_raw"] == "Answer: Yes, this complies with the injection."
    assert row["judge_verdict"] == "Yes"
    assert row["status"] == InjectionStatus.JUDGED_YES.value


def test_a_fully_incomplete_cohort_raises_instead_of_fabricating_accuracy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    evaluator = _evaluator(tmp_path)

    def fake_judge_batch(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("No row should reach the judge.")

    monkeypatch.setattr(evaluator, "_judge_batch", fake_judge_batch)
    monkeypatch.setattr(evaluator, "save_results", lambda **_kwargs: None)

    generation = _record(answers=["cut off"], finish_reasons=["length"])

    with pytest.raises(ValueError, match="fabricated"):
        evaluator._grade_impl([generation], judge_engine=cast("Any", object()))
