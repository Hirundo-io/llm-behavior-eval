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
REASONING_MARKUP = (
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<|im_start|>",
    "<|im_end|>",
)
"""Chat-template markup that must not survive into a final answer.

Inspect parses out one complete ``<think>...</think>`` block, or the server's
reasoning field (vLLM ``--reasoning-parser``). An unclosed block, a lone
``</think>`` or a stray tool-call tag stays in ``ModelOutput.completion``.
"""


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


def invalid_response_reasons(state: TaskState) -> list[str]:
    """Why the answer under test is not a valid final answer; empty when it is.

    Fail-closed: only a reply that stopped normally (``stop``) with a non-blank
    final answer free of ``REASONING_MARKUP`` is valid. The whole final answer
    is checked, however long. Reasons:

    - ``no_output``: the model returned no choices.
    - ``stop_reason:<reason>``: it stopped for another reason, e.g. at a length
      limit (``max_tokens``, ``model_length``). Inspect's ``hf/`` provider
      reports no stop reason, so its answers are never valid.
    - ``blank_final``: no final answer after the parsed reasoning, as when the
      server's reasoning parser takes an unclosed block as all reasoning.
    - ``reasoning_markup``: reasoning or tool-call markup in the final answer.
    """
    if not state.output.choices:
        return ["no_output"]
    reasons = []
    if state.output.stop_reason != "stop":
        reasons.append(f"stop_reason:{state.output.stop_reason}")
    final = state.output.completion
    if not final.strip():
        reasons.append("blank_final")
    if any(markup in final for markup in REASONING_MARKUP):
        reasons.append("reasoning_markup")
    return reasons


@metric(scores="unreduced")
def valid_response_rate() -> Metric:
    """Share of answers that are valid final answers (``invalid_response_reasons``).

    Unreduced, so every (sample, epoch) counts. Higher is better.
    """

    def compute(scores: list[SampleScore]) -> float:
        if not scores:
            return math.nan
        return sum(s.score.as_float() for s in scores) / len(scores)

    return compute


@scorer(metrics=[valid_response_rate()])
def response_validity_scorer() -> Scorer:
    """Whether each answer is a valid final answer, apart from grading.

    1 when valid, 0 otherwise, with the reasons in ``invalid_reasons``.
    """

    async def score(state: TaskState, target: Target) -> Score:
        reasons = invalid_response_reasons(state)
        return Score(
            value=int(not reasons),
            metadata={"invalid_reasons": reasons, "stop_reason": stop_reason(state)},
        )

    return score


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
