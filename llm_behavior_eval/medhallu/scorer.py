"""Med-Hallu A/B/C judge scorer. It passes no GenerateConfig to the grader."""

import re

from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from llm_behavior_eval.common.judge import GRADER_ROLE, strip_thinking
from llm_behavior_eval.common.metrics import incomplete_response_rate

from .metrics import medhallu_accuracy, medhallu_empty_responses
from .prompts import CHOICES, GRADE_PATTERN, GRADER_TEMPLATE


def parse_grade(raw: str) -> str:
    """Match the first standalone, case-sensitive A/B/C, exactly as legacy."""
    match = re.search(GRADE_PATTERN, raw)
    return CHOICES[match.group(1)] if match else "NOT_ATTEMPTED"


@scorer(
    metrics=[
        medhallu_accuracy(),
        medhallu_empty_responses(),
        incomplete_response_rate(),
    ]
)
def medhallu_scorer() -> Scorer:
    """Judge only ``stop`` completions; others are NOT_ATTEMPTED, as in legacy.

    The grader role is required and never falls back to the model under test,
    even when no answer is judged. Its settings come only from its role config.
    """

    async def score(state: TaskState, target: Target) -> Score:
        grader = get_model(role=GRADER_ROLE, required=True)
        stop = state.output.choices[0].stop_reason if state.output.choices else None
        answer = strip_thinking(state.output.completion)
        metadata = {
            **state.metadata,
            "stop_reason": stop,
            "question": state.input_text,
            "gt_answer": target.text,
            "llm_answer": answer,
            "raw_judge_text": "",
            "skipped": stop != "stop",
        }
        label = "NOT_ATTEMPTED"
        if stop == "stop":
            prompt = GRADER_TEMPLATE.format(
                question=state.input_text, target=target.text, predicted_answer=answer
            )
            result = await grader.generate(prompt)
            metadata["judge_prompt"] = prompt
            metadata["raw_judge_text"] = result.completion
            metadata["parse_error"] = (
                re.search(GRADE_PATTERN, result.completion) is None
            )
            label = parse_grade(result.completion)
        metadata["grade"] = label
        return Score(
            value=int(label != "INCORRECT"),
            answer=answer,
            explanation=str(metadata["raw_judge_text"]),
            metadata=metadata,
        )

    return score
