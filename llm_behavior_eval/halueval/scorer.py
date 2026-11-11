"""HaluEval judge: legacy SimpleQA-style A/B/C grading by the ``grader`` role.

Only answers that finished with ``stop`` are judged. Every other answer
(``max_tokens``, ``model_length``, unknown or missing stop reason) is graded
NOT_ATTEMPTED without a judge call, and an unmatched judge reply is NOT_ATTEMPTED
too, as in the legacy ``FreeTextHaluEvaluator``. Because the grade depends on the
stop reason, a model under test on Inspect's ``hf/`` provider, which reports
none, fails with ``StopReasonUnavailableError`` instead of grading every answer
NOT_ATTEMPTED. The scorer passes no GenerateConfig: the grader uses only its
role's config.
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

from .metrics import halueval_accuracy, halueval_empty_responses
from .prompts import CHOICES, GRADE_PATTERN, GRADER_TEMPLATE

NOT_ATTEMPTED = "NOT_ATTEMPTED"


def parse_grade(raw: str) -> str:
    """Map the first standalone, case-sensitive A/B/C; otherwise NOT_ATTEMPTED."""
    match = re.search(GRADE_PATTERN, raw)
    return CHOICES[match.group(1)] if match else NOT_ATTEMPTED


@scorer(
    metrics=[
        halueval_accuracy(),
        halueval_empty_responses(),
        incomplete_response_rate(),
    ]
)
def halueval_scorer(
    *,
    exclude_thinking_trace_for_judge: bool = True,
    thinking_start_token: str | None = THINKING_START,
    thinking_end_token: str | None = THINKING_END,
) -> Scorer:
    """Score 1 unless the grade is INCORRECT; the grade is in ``metadata["grade"]``.

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
        # Required even when no answer is judged, so a missing grader fails fast.
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
        grade = NOT_ATTEMPTED
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
            grade = parse_grade(result.completion)
        metadata["grade"] = grade
        return Score(
            value=int(grade != "INCORRECT"),
            answer=answer,
            explanation=str(metadata["raw_judge_text"]),
            metadata=metadata,
        )

    return score
