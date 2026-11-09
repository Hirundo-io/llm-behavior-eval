"""Benchmark-owned scorer built on the shared grader mechanics."""

from inspect_ai.scorer import Scorer

from llm_behavior_eval.common.judge import judge_scorer

from .prompts import JUDGE_PATTERN, JUDGE_PROMPT


def scorer(
    *,
    exclude_thinking_trace_for_judge: bool,
    thinking_start_token: str | None,
    thinking_end_token: str | None,
) -> Scorer:
    return judge_scorer(
        JUDGE_PROMPT,
        JUDGE_PATTERN,
        only_stop=True,
        exclude_thinking_trace_for_judge=exclude_thinking_trace_for_judge,
        thinking_start_token=thinking_start_token,
        thinking_end_token=thinking_end_token,
    )
