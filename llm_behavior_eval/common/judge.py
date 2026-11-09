"""Grader call, thinking-trace stripping and answer parsing for judge-based scorers.

Nothing here sets inference options for either model. The grader is whatever
model is bound to the ``grader`` role, with whatever GenerateConfig the user gave
that role; unset options fall through to the serving stack's defaults.
"""

import math
import re
from collections.abc import Callable

from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from .metrics import field_rate, incomplete_response_rate

GRADER_ROLE = "grader"
THINKING_START = "<think>"
THINKING_END = "</think>"


def strip_thinking(
    answer: str,
    *,
    exclude_thinking_trace_for_judge: bool = True,
    thinking_start_token: str | None = THINKING_START,
    thinking_end_token: str | None = THINKING_END,
) -> str:
    """Drop an inline reasoning trace: keep only the text after the last end marker.

    Applies only when both markers are configured and either one appears, so an
    unclosed start marker keeps the whole answer. Structured ``ContentReasoning``
    is already excluded from ``ModelOutput.completion`` by Inspect.

    Args:
        answer: Model output text.
        exclude_thinking_trace_for_judge: ``False`` returns ``answer`` unchanged.
        thinking_start_token: Marker that opens the trace; ``None`` disables
            stripping.
        thinking_end_token: Marker that closes the trace; ``None`` disables
            stripping.

    Returns:
        The stripped text after the last end marker, or ``answer`` unchanged.
    """
    if (
        exclude_thinking_trace_for_judge
        and thinking_start_token
        and thinking_end_token
        and (thinking_start_token in answer or thinking_end_token in answer)
    ):
        return answer.rsplit(thinking_end_token, 1)[-1].strip()
    return answer


@scorer(metrics=[field_rate(), incomplete_response_rate()])
def judge_scorer(
    prompt: str,
    pattern: str,
    *,
    parse: Callable[[str], float] = float,
    only_stop: bool = False,
    exclude_thinking_trace_for_judge: bool = True,
    thinking_start_token: str | None = THINKING_START,
    thinking_end_token: str | None = THINKING_END,
) -> Scorer:
    """Ask the ``grader`` role to judge an answer; regex group 1 feeds ``parse``.

    The grader role is required: it never falls back to the model under test.
    Generation settings come only from the role's own GenerateConfig; this
    scorer passes none. The verdict regex is applied to the grader's final
    answer: an inline ``<think>...</think>`` trace in the grader's reply is
    removed first (structured reasoning never reaches ``completion``), and the
    full reply stays in ``raw_judge_text``.

    Invalid (unmatched, unparsable or non-finite) and skipped scores have value 0
    with ``parse_error``/``skipped`` metadata, and are excluded by ``field_rate``.
    The answer's stop reason is kept in score metadata. Benchmark scorers may
    wrap this scorer and supply their own metrics.

    Args:
        prompt: Grader prompt, formatted with ``{question}``, ``{answer}``,
            ``{target}`` and ``{metadata}``.
        pattern: Regex applied to the grader's final answer; group 1 is parsed.
        parse: Converts group 1 to a score; ``ValueError`` marks a parse error.
        only_stop: Skip answers that did not finish with ``stop``, without
            calling the grader.
        exclude_thinking_trace_for_judge: Strip the answer's inline thinking
            trace before the grader sees it.
        thinking_start_token: Marker that opens the answer's thinking trace.
        thinking_end_token: Marker that closes the answer's thinking trace.

    Returns:
        A scorer recording ``stop_reason`` and ``raw_judge_text`` metadata.
    """
    regex = re.compile(pattern)

    async def score(state: TaskState, target: Target) -> Score:
        stop = state.output.choices[0].stop_reason if state.output.choices else None
        metadata = {**state.metadata, "stop_reason": stop, "raw_judge_text": ""}
        if only_stop and stop != "stop":
            metadata["skipped"] = True
            return Score(
                value=0,
                answer=state.output.completion,
                explanation="Answer did not finish with stop",
                metadata=metadata,
            )
        answer = strip_thinking(
            state.output.completion,
            exclude_thinking_trace_for_judge=exclude_thinking_trace_for_judge,
            thinking_start_token=thinking_start_token,
            thinking_end_token=thinking_end_token,
        )
        rendered = prompt.format(
            question=state.input_text,
            answer=answer,
            target=target.text,
            metadata=state.metadata,
        )
        grader = get_model(role=GRADER_ROLE, required=True)
        result = await grader.generate(rendered)
        raw = result.completion
        metadata["raw_judge_text"] = raw
        match = regex.search(strip_thinking(raw))
        try:
            if match is None:
                raise ValueError("Judge response did not match pattern")
            parsed = match.group(1)
            value = parse(parsed)
            if not math.isfinite(value):
                raise ValueError(f"Judge score is not finite: {parsed!r}")
        except (ValueError, IndexError) as exc:
            metadata["parse_error"] = str(exc)
            return Score(value=0, answer=raw, explanation=str(exc), metadata=metadata)
        return Score(value=value, answer=parsed, explanation=raw, metadata=metadata)

    return score
