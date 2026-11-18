"""CCPC-500-v2 task composed from Inspect dataset, solver and scorer APIs."""

from inspect_ai import Task, task
from inspect_ai.solver import generate, prompt_template

from llm_behavior_eval.ccpc_v2.dataset import (
    DATASET_CONFIG,
    DATASET_FINGERPRINT,
    DATASET_REPOSITORY,
    DATASET_REVISION,
    DATASET_SPLIT,
    EXPECTED_ROWS,
    dataset,
)
from llm_behavior_eval.ccpc_v2.prompts import RUBRIC_VERSION
from llm_behavior_eval.ccpc_v2.scorer import ccpc_v2_scorer
from llm_behavior_eval.common.scoring import completion_scorer, response_validity_scorer

TASK_VERSION = 1
"""Bump when a change can alter results (prompts, parsing, data, metrics).

CCPC-500-v2 results are not comparable with CCPC-500 (v1) results.
"""
REFERENCE_JUDGE = "google/gemma-4-26B-A4B-it"


@task
def ccpc_v2(
    dataset_path: str | None = None,
    judge_sees_answer_reasoning: bool = False,
    require_valid_response: bool = False,
) -> Task:
    """CCPC-500-v2 censorship rate, judged by the required ``grader`` role.

    Args:
        dataset_path: Read this local source instead of the pinned private Hub
            dataset; it must hold the same rows as the frozen release.
        judge_sees_answer_reasoning: Give the judge the answer's parsed
            reasoning too, as the legacy engine judged the whole answer.
        require_valid_response: Judge only valid final answers and add the
            ``valid_response_rate`` diagnostic (thinking-on runs, see
            ``thinking_on.yaml``). Any invalid answer leaves
            ``censorship_rate`` undefined. Off by default, which keeps the
            legacy judge-everything contract.

    Returns:
        A task with no model or grader inference defaults.
    """
    return Task(
        dataset=dataset(dataset_path),
        solver=[prompt_template("{prompt}\n"), generate()],
        scorer=[
            ccpc_v2_scorer(
                judge_sees_answer_reasoning=judge_sees_answer_reasoning,
                require_valid_response=require_valid_response,
            ),
            completion_scorer(),
            *([response_validity_scorer()] if require_valid_response else []),
        ],
        version=TASK_VERSION,
        metadata={
            "dataset_repository": DATASET_REPOSITORY,
            "dataset_config": DATASET_CONFIG,
            "dataset_split": DATASET_SPLIT,
            "dataset_revision": DATASET_REVISION,
            "dataset_rows": EXPECTED_ROWS,
            "dataset_fingerprint": DATASET_FINGERPRINT,
            "rubric_version": RUBRIC_VERSION,
            "reference_judge": REFERENCE_JUDGE,
        },
    )
