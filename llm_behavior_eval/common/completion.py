"""Completion diagnostics: did the answer under test finish or was it cut off?

``completion_scorer`` is its own scorer, listed in every benchmark task beside
the grading scorer (``scorer=[<benchmark>_scorer(...), completion_scorer()]``).
Inspect drops unscored observations per scorer, so keeping completion apart
means an answer whose grading was skipped or failed still counts in
``incomplete_response_rate``: its denominator is every (sample, epoch)
observation with a known stop reason.
"""

from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from .metrics import (
    INCOMPLETE_STOP_REASONS,
    UNKNOWN_STOP_REASON,
    incomplete_response_rate,
)

UNKNOWN_STOP_SCORE_REASON = "scoring_failed"
"""``Score.reason`` of an observation whose stop reason is unknown."""


@scorer(metrics=[incomplete_response_rate()])
def completion_scorer() -> Scorer:
    """Score whether the answer under test stopped at a length limit.

    The value is 1.0 when the answer's ``stop_reason`` is in
    ``INCOMPLETE_STOP_REASONS`` (``max_tokens``, ``model_length``) and 0.0
    for any other known stop reason. An answer whose stop reason is
    ``unknown`` (every answer on Inspect's ``hf/`` provider), or that has no
    choices, cannot be classified and is ``Score.unscored``, so
    ``incomplete_response_rate`` leaves it out and reports NaN rather than 0
    when no stop reason is known. It does not call ``require_stop_reason``:
    a diagnostic that cannot be computed is NaN, not an error.

    Score metadata records ``stop_reason``. The scorer needs no grader.

    Returns:
        A scorer whose only metric is ``incomplete_response_rate``.
    """

    async def score(state: TaskState, target: Target) -> Score:
        stop = state.output.choices[0].stop_reason if state.output.choices else None
        metadata = {"stop_reason": stop}
        if stop is None or stop == UNKNOWN_STOP_REASON:
            return Score.unscored(
                reason=UNKNOWN_STOP_SCORE_REASON,
                explanation="The provider reported no stop reason for this answer",
                metadata=metadata,
            )
        incomplete = stop in INCOMPLETE_STOP_REASONS
        return Score(
            value=1.0 if incomplete else 0.0,
            explanation=f"Answer stopped with {stop!r}",
            metadata=metadata,
        )

    return score
