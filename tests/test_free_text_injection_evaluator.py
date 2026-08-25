import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from llm_behavior_eval.evaluation_utils.free_text_bias_evaluator import (
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
from llm_behavior_eval.evaluation_utils.free_text_refusal_evaluator import (
    FreeTextRefusalEvaluator,
    _RefusalGenerationRecord,
)


def test_generation_record_rejects_short_generated_answers() -> None:
    evaluator = object.__new__(FreeTextPromptInjectionEvaluator)
    generation = _InjectionGenerationRecord(
        input_texts=["prompt one", "prompt two"],
        judge_questions=["question one", "question two"],
        gt_answers=["No", "No"],
        answers=["answer one"],
        finish_reasons=["stop"],
    )

    with pytest.raises(
        ValueError,
        match=r"fresh batch 3.*answers=1",
    ):
        evaluator._validate_generation_record(generation, context="fresh batch 3")


def test_generation_record_rejects_short_finish_reasons() -> None:
    evaluator = object.__new__(FreeTextPromptInjectionEvaluator)
    generation = _InjectionGenerationRecord(
        input_texts=["prompt"],
        judge_questions=["question"],
        gt_answers=["No"],
        answers=["answer"],
        finish_reasons=[],
    )

    with pytest.raises(ValueError, match=r"grading.*finish_reasons=0"):
        evaluator._validate_generation_record(generation, context="grading")


def test_legacy_cached_generation_uses_input_texts_as_judge_questions() -> None:
    evaluator = object.__new__(FreeTextPromptInjectionEvaluator)
    generation = FreeTextPromptInjectionEvaluator._generation_from_dict(
        {
            "input_texts": ["legacy question"],
            "gt_answers": ["No"],
            "answers": ["answer"],
            "finish_reasons": ["stop"],
        }
    )

    evaluator._validate_generation_record(generation, context="cached batch 0")
    assert generation.judge_questions == ["legacy question"]


@pytest.mark.parametrize(
    ("evaluator_type", "generation", "short_field"),
    [
        (
            FreeTextHaluEvaluator,
            _HalluGenerationRecord(
                input_texts=["one", "two"],
                gt_answers=["A", "B"],
                answers=["answer"],
                finish_reasons=["stop"],
            ),
            "answers=1",
        ),
        (
            FreeTextRefusalEvaluator,
            _RefusalGenerationRecord(
                input_texts=["one", "two"],
                expected_labels=["safe", "unsafe"],
                answers=["answer"],
                finish_reasons=["stop"],
            ),
            "answers=1",
        ),
        (
            FreeTextBiasEvaluator,
            _BiasGenerationRecord(
                questions=["one", "two"],
                correct_answers=["A", "B"],
                stereotyped_answers=["B", "A"],
                answers=["answer"],
                finish_reasons=["stop"],
            ),
            "answers=1",
        ),
    ],
)
def test_all_free_text_evaluators_reject_misaligned_generation_records(
    evaluator_type: type,
    generation: object,
    short_field: str,
) -> None:
    evaluator = object.__new__(evaluator_type)

    with pytest.raises(ValueError, match=short_field):
        evaluator._validate_generation_record(generation, context="grading")
