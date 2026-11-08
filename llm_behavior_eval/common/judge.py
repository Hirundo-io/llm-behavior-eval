"""Native Inspect grader role with explicit parsing and legacy answer formatting."""

import re
from collections.abc import Callable

from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from llm_behavior_eval.common.preprocessing import judge_question, judge_target

from .metrics import field_rate, incomplete_response_rate

THINKING_START = "<think>"
THINKING_END = "</think>"


def strip_thinking(
    answer: str,
    *,
    exclude_thinking_trace_for_judge: bool = True,
    thinking_start_token: str | None = THINKING_START,
    thinking_end_token: str | None = THINKING_END,
) -> str:
    """Match FreeTextSharedEvaluator._format_answers, including unmatched markers."""
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
    config: GenerateConfig | None = None,
    parse: Callable[[str], float] = float,
    only_stop: bool = False,
    exclude_thinking_trace_for_judge: bool = True,
    thinking_start_token: str | None = THINKING_START,
    thinking_end_token: str | None = THINKING_END,
) -> Scorer:
    """Format {question}/{answer}/{target}/{metadata}; regex group 1 feeds parse.

    Requires an independent grader role. Invalid/skipped scores have value 0 but
    are excluded by field_rate; raw judge text and status remain in metadata.
    Benchmark scorers may wrap this scorer and supply their own metrics.
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
            question=judge_question(state),
            answer=answer,
            target=judge_target(state, target.text),
            metadata=state.metadata,
        )
        grader = get_model(role="grader", required=True)
        result = await grader.generate(
            rendered, config=config or GenerateConfig(temperature=0, max_tokens=32)
        )
        raw = result.completion
        metadata["raw_judge_text"] = raw
        match = regex.search(raw)
        try:
            if match is None:
                raise ValueError("Judge response did not match pattern")
            parsed = match.group(1)
            value = parse(parsed)
        except (ValueError, IndexError) as exc:
            metadata["parse_error"] = str(exc)
            return Score(value=0, answer=raw, explanation=str(exc), metadata=metadata)
        return Score(value=value, answer=parsed, explanation=raw, metadata=metadata)

    return score
