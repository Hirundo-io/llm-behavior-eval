"""HaluEval judge: legacy SimpleQA-style A/B/C grading by the ``grader`` role.

Only answers that finished with ``stop`` are judged. Every other answer
(``max_tokens``, unknown or missing stop reason) is graded NOT_ATTEMPTED without
a judge call, and an unmatched judge reply is NOT_ATTEMPTED too, as in the legacy
``FreeTextHaluEvaluator``. The scorer passes no GenerateConfig: the grader uses
only its role's config.
"""

import re

from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from llm_behavior_eval.common.judge import GRADER_ROLE, strip_thinking
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
def halueval_scorer() -> Scorer:
    """Score 1 unless the grade is INCORRECT; the grade is in ``metadata["grade"]``."""

    async def score(state: TaskState, target: Target) -> Score:
        # Required even when no answer is judged, so a missing grader fails fast.
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
