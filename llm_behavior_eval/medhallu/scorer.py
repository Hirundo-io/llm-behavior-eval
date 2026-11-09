"""Med-Hallu A/B/C judge scorer. It passes no GenerateConfig to the grader.

The grade depends on the answer's stop reason, so a model under test on
Inspect's ``hf/`` provider, which reports none, fails with
``StopReasonUnavailableError`` instead of grading every answer NOT_ATTEMPTED.
"""

import re

from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from llm_behavior_eval.common.judge import (
    GRADER_ROLE,
    THINKING_END,
    THINKING_START,
    require_stop_reason,
    strip_thinking,
)
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
def medhallu_scorer(
    *,
    exclude_thinking_trace_for_judge: bool = True,
    thinking_start_token: str | None = THINKING_START,
    thinking_end_token: str | None = THINKING_END,
) -> Scorer:
    """Judge only ``stop`` completions; others are NOT_ATTEMPTED, as in legacy.

    The grader role is required and never falls back to the model under test,
    even when no answer is judged. Its settings come only from its role config.

    Args:
        exclude_thinking_trace_for_judge: Strip the answer's inline thinking
            trace before the grader sees it (the task argument of the same name).
        thinking_start_token: Marker that opens the answer's thinking trace.
        thinking_end_token: Marker that closes the answer's thinking trace.

    Raises:
        StopReasonUnavailableError: From the scorer, when the model under test
            runs on Inspect's ``hf/`` provider.
    """

    async def score(state: TaskState, target: Target) -> Score:
        grader = get_model(role=GRADER_ROLE, required=True)
        require_stop_reason(state)
        stop = state.output.choices[0].stop_reason if state.output.choices else None
        answer = strip_thinking(
            state.output.completion,
            exclude_thinking_trace_for_judge=exclude_thinking_trace_for_judge,
            thinking_start_token=thinking_start_token,
            thinking_end_token=thinking_end_token,
        )
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
