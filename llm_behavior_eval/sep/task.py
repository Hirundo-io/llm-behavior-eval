"""SEP task: instruction-data separation (Zverev et al., ICLR 2025)."""

from inspect_ai import Task, task

from llm_behavior_eval.common.scoring import completion_scorer
from llm_behavior_eval.sep.dataset import (
    DATASET_COMMIT,
    DATASET_FILE,
    DATASET_LICENSE,
    DATASET_REPOSITORY,
    DATASET_SHA256,
    EXPECTED_ROWS,
    dataset,
)
from llm_behavior_eval.sep.scorer import sep_scorer
from llm_behavior_eval.sep.solver import sep_solver

TASK_VERSION = 1
"""Bump when a change can alter results (prompts, matching, data, metrics)."""


@task
def sep(dataset_path: str | None = None, system_role: bool = True) -> Task:
    """SEP separation and utility scores, scored by the witness rule (no grader).

    Each item is generated twice, with the probe in the instruction (system)
    argument and in the data (user) argument.

    Args:
        dataset_path: Read this local copy of the authors'
            ``datasets/SEP_dataset.json`` instead of the pinned download; it
            must have the pinned SHA-256.
        system_role: Instruction argument as the system message, data
            argument as the user message (the paper's default). ``False``
            sends one user message ``System prompt: ... User prompt: ...``, as
            the paper did for models without a system role (Gemma, Starling).

    Returns:
        A task with no inference settings.
    """
    return Task(
        dataset=dataset(dataset_path),
        solver=sep_solver(system_role=system_role),
        scorer=[sep_scorer(), completion_scorer()],
        version=TASK_VERSION,
        metadata={
            "dataset_repository": DATASET_REPOSITORY,
            "dataset_commit": DATASET_COMMIT,
            "dataset_file": DATASET_FILE,
            "dataset_sha256": DATASET_SHA256,
            "dataset_rows": EXPECTED_ROWS,
            "dataset_license": DATASET_LICENSE,
            "prompt_template": "naive (template 0)",
        },
    )
