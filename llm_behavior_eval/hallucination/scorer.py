"""Shared HaluEval/Med-Hallu judge and legacy metrics.

accuracy: percentage, higher is better, 100 * (1 - INCORRECT / all samples).
empty_responses: NOT_ATTEMPTED count. incomplete_response_rate: fraction of all
samples ending at max_tokens (legacy length). Skips/parse failures stay in totals.
"""

import re

from inspect_ai.model import get_model
from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, metric, scorer
from inspect_ai.solver import TaskState

from llm_behavior_eval.common.judge import strip_thinking
from llm_behavior_eval.common.metrics import incomplete_response_rate
from llm_behavior_eval.common.preprocessing import (
    judge_config,
    judge_question,
    judge_target,
)
from llm_behavior_eval.hallucination.prompts import GRADER_TEMPLATE
from llm_behavior_eval.hallucination.solver import configured_generation

CHOICES = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}


def parse_grade(raw: str) -> str:
    """Match the first standalone, case-sensitive A/B/C, exactly as legacy."""
    match = re.search(r"\b([ABC])\b", raw)
    return CHOICES[match.group(1)] if match else "NOT_ATTEMPTED"


@metric
def accuracy() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        return (
            100
            * (
                1
                - sum(
                    (s.score.metadata or {}).get("grade") == "INCORRECT" for s in scores
                )
                / len(scores)
            )
            if scores
            else float("nan")
        )

    return compute


@metric
def empty_responses() -> Metric:
    def compute(scores: list[SampleScore]) -> float:
        return sum(
            (s.score.metadata or {}).get("grade") == "NOT_ATTEMPTED" for s in scores
        )

    return compute


@scorer(metrics=[accuracy(), empty_responses(), incomplete_response_rate()])
def hallucination_scorer(*, thinking_disabled: bool = False) -> Scorer:
    """Require explicit grader; endpoint thinking must be disabled independently.

    thinking_disabled attests to external thinking-off configuration for providers
    without a supported native control. It does not set a provider option.
    """

    async def score(state: TaskState, target: Target) -> Score:
        grader = get_model(role="grader", required=True)
        config = configured_generation(
            grader, judge=True, thinking_disabled=thinking_disabled
        )
        config = judge_config(state, config)
        stop = state.output.choices[0].stop_reason if state.output.choices else None
        answer = strip_thinking(state.output.completion)
        metadata = {
            **state.metadata,
            "stop_reason": stop,
            "question": judge_question(state),
            "gt_answer": judge_target(state, target.text),
            "llm_answer": answer,
            "raw_judge_text": "",
            "skipped": stop != "stop",
        }
        label = "NOT_ATTEMPTED"
        if stop == "stop":
            prompt = GRADER_TEMPLATE.format(
                question=judge_question(state),
                target=judge_target(state, target.text),
                predicted_answer=answer,
            )
            result = await grader.generate(prompt, config=config)
            metadata["judge_prompt"] = prompt
            metadata["raw_judge_text"] = result.completion
            metadata["parse_error"] = (
                re.search(r"\b([ABC])\b", result.completion) is None
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
