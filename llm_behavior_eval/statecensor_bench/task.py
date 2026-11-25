"""StateCensor-Bench task composed from Inspect dataset, solver and scorer APIs."""

from inspect_ai import Task, task
from inspect_ai.solver import generate, prompt_template

from llm_behavior_eval.common.scoring import completion_scorer, response_validity_scorer
from llm_behavior_eval.statecensor_bench.dataset import (
    DATASET_FILE,
    DATASET_REPOSITORY,
    DATASET_REVISION,
    DATASET_SHA256,
    EXPECTED_ROWS,
    dataset,
)
from llm_behavior_eval.statecensor_bench.prompts import RUBRIC_VERSION
from llm_behavior_eval.statecensor_bench.scorer import statecensor_bench_scorer

TASK_VERSION = 1
"""Bump when a change can alter results (prompts, parsing, data, metrics).

StateCensor-Bench results are not comparable with CCPC-500 (v1) results.
"""
REFERENCE_JUDGE = "google/gemma-4-26B-A4B-it"


@task
def statecensor_bench(
    dataset_path: str | None = None,
    judge_sees_answer_reasoning: bool = False,
    require_valid_response: bool = False,
) -> Task:
    """StateCensor-Bench censorship rate, judged by the required ``grader`` role.

    Args:
        dataset_path: Read this local copy of the pinned release file
            (``data/test-00000-of-00001.jsonl``) instead of downloading it
            from the Hub; it must have the pinned SHA-256.
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
            statecensor_bench_scorer(
                judge_sees_answer_reasoning=judge_sees_answer_reasoning,
                require_valid_response=require_valid_response,
            ),
            completion_scorer(),
            *([response_validity_scorer()] if require_valid_response else []),
        ],
        version=TASK_VERSION,
        metadata={
            "dataset_repository": DATASET_REPOSITORY,
            "dataset_revision": DATASET_REVISION,
            "dataset_file": DATASET_FILE,
            "dataset_sha256": DATASET_SHA256,
            "dataset_rows": EXPECTED_ROWS,
            "rubric_version": RUBRIC_VERSION,
            "reference_judge": REFERENCE_JUDGE,
        },
    )
