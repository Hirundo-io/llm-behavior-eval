"""CCPC v6, adapted from the upstream Inspect task (see README)."""

from inspect_ai import Task, task

from llm_behavior_eval.ccpc.contract import (
    CCPC_DATASET_FINGERPRINT,
    CCPC_DATASET_REPOSITORY,
    CCPC_DATASET_REVISION,
    CCPC_EXPECTED_ROWS,
    CCPC_RUBRIC_VERSION,
    CCPC_TASK_VERSION,
)
from llm_behavior_eval.ccpc.dataset import load_ccpc_dataset
from llm_behavior_eval.ccpc.scorer import ccpc_scorer
from llm_behavior_eval.ccpc.solver import PROMPT_PARITY, solver

CCPC_REFERENCE_JUDGE = "google/gemma-4-26B-A4B-it"


@task
def ccpc(
    *,
    dataset_path: str | None = None,
    file_path: str | None = None,
    dataset_token: str | None = None,
    judge_revision: str | None = None,
) -> Task:
    # No `config=` and no default grader: inference settings for both roles
    # come from the user or from parity.yaml.
    return Task(
        dataset=load_ccpc_dataset(
            dataset_path=dataset_path,
            file_path=file_path,
            dataset_token=dataset_token,
        ),
        solver=solver(),
        scorer=ccpc_scorer(judge_revision=judge_revision),
        version=CCPC_TASK_VERSION,
        metadata=dict(
            task_version=CCPC_TASK_VERSION,
            dataset_repository=CCPC_DATASET_REPOSITORY,
            dataset_revision=CCPC_DATASET_REVISION,
            dataset_fingerprint=CCPC_DATASET_FINGERPRINT,
            rubric_version=CCPC_RUBRIC_VERSION,
            canonical_rows=CCPC_EXPECTED_ROWS,
            judge_reference_model=CCPC_REFERENCE_JUDGE,
            judge_model_revision=judge_revision,
            prompt_parity=PROMPT_PARITY,
        ),
    )
