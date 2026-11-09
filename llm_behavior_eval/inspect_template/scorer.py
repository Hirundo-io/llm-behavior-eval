"""Benchmark-owned scorer built on the shared grader mechanics."""

from inspect_ai.scorer import Scorer

from llm_behavior_eval.common.judge import judge_scorer

from .prompts import JUDGE_PATTERN, JUDGE_PROMPT


def scorer() -> Scorer:
    return judge_scorer(JUDGE_PROMPT, JUDGE_PATTERN, only_stop=True)
