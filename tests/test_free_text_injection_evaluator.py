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
        match=(
            r"fresh batch 3.*input_texts=2.*judge_questions=2.*gt_answers=2"
            r".*answers=1.*finish_reasons=1"
        ),
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


def test_incomplete_cached_generation_keeps_only_valid_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = object.__new__(FreeTextPromptInjectionEvaluator)
    valid = {
        "input_texts": ["prompt one"],
        "judge_questions": ["question one"],
        "gt_answers": ["No"],
        "answers": ["answer one"],
        "finish_reasons": ["stop"],
    }
    incomplete = {
        "input_texts": ["prompt two", "prompt three"],
        "judge_questions": ["question two", "question three"],
        "gt_answers": ["No", "No"],
        "answers": ["answer two"],
        "finish_reasons": ["stop"],
    }
    reset_filenames: list[str] = []
    saved: list[tuple[list[dict], str]] = []
    monkeypatch.setattr(
        evaluator,
        "load_completed_generation_dicts",
        lambda _filename: [valid, incomplete],
    )
    monkeypatch.setattr(
        evaluator,
        "reset_generations_file",
        lambda filename: reset_filenames.append(filename),
    )
    monkeypatch.setattr(
        evaluator,
        "save_generations",
        lambda items, filename: saved.append((items, filename)),
    )

    result = evaluator.load_aligned_generation_dicts(
        ("input_texts", "gt_answers", "answers", "finish_reasons"),
        optional_fields=("judge_questions",),
    )

    assert result == [valid]
    assert reset_filenames == ["generations.jsonl"]
    assert saved == [([valid], "generations.jsonl")]


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
