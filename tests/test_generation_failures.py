from typing import TYPE_CHECKING, Any, cast

import pytest

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

if TYPE_CHECKING:
    from llm_behavior_eval.evaluation_utils.eval_engine import EvalEngine


@pytest.mark.parametrize(
    ("evaluator_class", "generation"),
    [
        (
            FreeTextBiasEvaluator,
            _BiasGenerationRecord(
                questions=["question"],
                answers=["answer"],
                correct_answers=["correct"],
                stereotyped_answers=None,
                finish_reasons=["harmony_parse_error"],
            ),
        ),
        (
            FreeTextHaluEvaluator,
            _HalluGenerationRecord(
                input_texts=["question"],
                gt_answers=["correct"],
                answers=["answer"],
                finish_reasons=["harmony_parse_error"],
            ),
        ),
        (
            FreeTextPromptInjectionEvaluator,
            _InjectionGenerationRecord(
                input_texts=["question"],
                judge_questions=["judge question"],
                gt_answers=["correct"],
                answers=["answer"],
                finish_reasons=["harmony_parse_error"],
            ),
        ),
    ],
)
def test_unsupported_generation_finish_reason_fails_closed(
    evaluator_class: type[Any], generation: Any
) -> None:
    evaluator = object.__new__(evaluator_class)

    with pytest.raises(ValueError, match="unsupported finish reason"):
        evaluator._grade_impl([generation], cast("EvalEngine", object()))
