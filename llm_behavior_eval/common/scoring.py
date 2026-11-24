"""Scoring pieces every judge-based benchmark here needs.

Nothing here sets inference options: the grader is whatever model is bound to
the ``grader`` role, with that role's own settings.
"""

import math

from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    Model,
    ModelOutput,
    StopReason,
    get_model,
)
from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, metric, scorer
from inspect_ai.solver import TaskState
from inspect_evals_utils import message_content_text
from inspect_evals_utils.scorers import SCORING_FAILED

GRADER_ROLE = "grader"
INCOMPLETE_STOP_REASONS = frozenset({"max_tokens", "model_length"})


def stop_reason(state: TaskState) -> StopReason | None:
    """Why the answer under test stopped; ``None`` when it has no choices."""
    return state.output.stop_reason if state.output.choices else None


def judged_answer(state: TaskState, *, include_reasoning: bool = False) -> str:
    """The answer text the judge sees.

    ``ModelOutput.completion`` leaves out reasoning that Inspect's provider
    parsed out of the reply. With ``include_reasoning`` (a benchmark's
    ``judge_sees_answer_reasoning`` task argument) that reasoning is put back
    by ``inspect_evals_utils.message_content_text`` as ``<reasoning>`` blocks.
    """
    if include_reasoning and state.output.choices:
        return message_content_text(state.output.message)
    return state.output.completion


def grader() -> Model:
    """The model bound to the ``grader`` role; it never falls back."""
    return get_model(role=GRADER_ROLE, required=True)


async def ask_grader(system: str, user: str) -> ModelOutput:
    """Send a system and a user message to the grader, with no settings."""
    return await grader().generate(
        [ChatMessageSystem(content=system), ChatMessageUser(content=user)]
    )


@metric(scores="unreduced")
def incomplete_response_rate() -> Metric:
    """Share of answers stopped by ``max_tokens`` or ``model_length``.

    Unreduced, so every (sample, epoch) counts; NaN when no answer has a known
    stop reason.
    """

    def compute(scores: list[SampleScore]) -> float:
        if not scores:
            return math.nan
        return sum(s.score.as_float() for s in scores) / len(scores)

    return compute


@scorer(metrics=[incomplete_response_rate()])
def completion_scorer() -> Scorer:
    """Whether each answer stopped at a length limit, apart from grading.

    Being its own scorer, its denominator does not change with grading
    outcomes. 1 for ``max_tokens`` or ``model_length``, 0 for another known
    stop reason, unscored when the provider reports none (Inspect's ``hf/``).
    """

    async def score(state: TaskState, target: Target) -> Score:
        stop = stop_reason(state)
        metadata = {"stop_reason": stop}
        if stop is None or stop == "unknown":
            return Score.unscored(
                reason=SCORING_FAILED,
                explanation="The provider reported no stop reason",
                metadata=metadata,
            )
        return Score(value=int(stop in INCOMPLETE_STOP_REASONS), metadata=metadata)

    return score
