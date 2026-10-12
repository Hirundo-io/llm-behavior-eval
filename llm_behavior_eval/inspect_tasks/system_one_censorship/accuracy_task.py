"""Accuracy, Brier and calibration of System One decision models (Inspect task).

Complements the bias audits: the same model, served the same way, answering
public typed-decision benchmarks with gold labels. Plain LLMs are read with the
same Yes-minus-No logprob readout the bias tasks use.

Example::

    inspect eval llm_behavior_eval/system_one_accuracy \\
        --model systemone/kev-4b --model-base-url http://127.0.0.1:8831 \\
        -T suites=jevbench,human
"""

from __future__ import annotations

import json
from typing import Any

from inspect_ai import Task, task
from inspect_ai.model import ChatMessageUser, ModelOutput, get_model
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    grouped,
    mean,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver

from .accuracy import (
    accuracy_dataset,
    distribution_from_answer,
    distribution_from_logits,
    expected_calibration_error,
    score_distribution,
)
from .protocol import yes_minus_no
from .provider import SystemOneAPI, encode_request, unwrap_response
from .scenarios import scenario_candidate_prompts
from .solver import _first_token_logprobs, logprob_config

ACCURACY_TASK_VERSION = 1
QUESTION_ID = "q"


@solver
def accuracy_readout(disable_thinking: bool = True) -> Solver:
    """Ask the model the item's one question and keep its full distribution."""
    config = logprob_config(disable_thinking)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()
        question = state.metadata["question"]
        if isinstance(model.api, SystemOneAPI):
            request = {
                "state": state.metadata["state"],
                "questions": {QUESTION_ID: question},
            }
            output = await model.generate([encode_request(request)])
            response = unwrap_response(json.loads(output.completion))
            dist = distribution_from_answer(question, response["answers"][QUESTION_ID])
            readout = "systemone"
        else:
            logits = []
            for prompt in scenario_candidate_prompts(state.metadata["state"], question):
                output = await model.generate(
                    [ChatMessageUser(content=prompt)], config=config
                )
                logits.append(yes_minus_no(_first_token_logprobs(output))[0])
            dist = distribution_from_logits(question, logits)
            readout = "logprobs"
        state.metadata["distribution"] = dist
        state.metadata["readout"] = readout
        state.output = ModelOutput.from_content(
            model=str(model), content=json.dumps(dist)
        )
        return state

    return solve


@metric
def ece() -> Metric:
    """Expected calibration error of the top-label confidence (10 bins)."""

    def compute(scores: list[SampleScore]) -> float:
        pairs = [
            (s.score.metadata["confidence"], s.score.metadata["correct"])  # type: ignore[index]
            for s in scores
        ]
        return expected_calibration_error(pairs)

    return compute


@scorer(
    metrics={
        "correct": [mean(), stderr(), grouped(mean(), "source")],
        "brier": [mean(), grouped(mean(), "source")],
        "confidence": [ece()],
    }
)
def accuracy_score() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        dist: dict[str, Any] = state.metadata["distribution"]
        values = score_distribution(dist, target.text)
        return Score(
            value=values,
            answer=max(dist, key=dist.__getitem__),
            metadata={
                "source": state.metadata["source"],
                "task": state.metadata["task"],
                "distribution": dist,
                "readout": state.metadata["readout"],
                "correct": values["correct"],
                "confidence": values["confidence"],
            },
        )

    return score


@task
def system_one_accuracy(
    suites: str | list[str] = "jevbench,human",
    per_task: int = 50,
    max_options: int = 16,
    disable_thinking: bool = True,
) -> Task:
    """Accuracy of a decision model on public typed-decision benchmarks.

    Args:
        suites: Comma-separated suites: ``jevbench`` (231 authored cases) and/or
            ``human`` (human-labelled public datasets).
        per_task: Items sampled per task in the ``human`` suite.
        max_options: Skip ``human`` questions with more options than this.
        disable_thinking: Pass ``enable_thinking=False`` in the logprob readout.
    """
    if isinstance(suites, str):
        suites = [s.strip() for s in suites.split(",") if s.strip()]
    return Task(
        dataset=accuracy_dataset(suites, per_task=per_task, max_options=max_options),
        solver=accuracy_readout(disable_thinking=disable_thinking),
        scorer=accuracy_score(),
        version=ACCURACY_TASK_VERSION,
    )
