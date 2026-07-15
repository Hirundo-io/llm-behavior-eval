import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from llm_behavior_eval.evaluation_utils.free_text_injection_evaluator import (
    FreeTextPromptInjectionEvaluator,
    _InjectionGenerationRecord,
)


def test_generation_record_rejects_short_generated_answers() -> None:
    generation = _InjectionGenerationRecord(
        input_texts=["prompt one", "prompt two"],
        judge_questions=["question one", "question two"],
        gt_answers=["No", "No"],
        answers=["answer one"],
        finish_reasons=["stop"],
    )

    with pytest.raises(
        ValueError,
        match=(
            r"fresh batch 3.*input_texts=2.*judge_questions=2.*gt_answers=2"
            r".*answers=1.*finish_reasons=1"
        ),
    ):
        FreeTextPromptInjectionEvaluator._validate_generation_record(
            generation, context="fresh batch 3"
        )


def test_generation_record_rejects_short_finish_reasons() -> None:
    generation = _InjectionGenerationRecord(
        input_texts=["prompt"],
        judge_questions=["question"],
        gt_answers=["No"],
        answers=["answer"],
        finish_reasons=[],
    )

    with pytest.raises(ValueError, match=r"grading.*finish_reasons=0"):
        FreeTextPromptInjectionEvaluator._validate_generation_record(
            generation, context="grading"
        )


def test_legacy_cached_generation_uses_input_texts_as_judge_questions() -> None:
    generation = FreeTextPromptInjectionEvaluator._generation_from_dict(
        {
            "input_texts": ["legacy question"],
            "gt_answers": ["No"],
            "answers": ["answer"],
            "finish_reasons": ["stop"],
        }
    )

    FreeTextPromptInjectionEvaluator._validate_generation_record(
        generation, context="cached batch 0"
    )
    assert generation.judge_questions == ["legacy question"]
