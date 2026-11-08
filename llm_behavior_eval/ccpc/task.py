"""CCPC v6, adapted from the upstream Inspect task (see README)."""

from inspect_ai import Task, task

from llm_behavior_eval.ccpc.contract import (
    CCPC_DATASET_FINGERPRINT,
    CCPC_DATASET_REPOSITORY,
    CCPC_DATASET_REVISION,
    CCPC_RUBRIC_VERSION,
    CCPC_TASK_VERSION,
)
from llm_behavior_eval.ccpc.dataset import load_ccpc_dataset
from llm_behavior_eval.ccpc.generation import PROMPT_PARITY, ccpc_generate
from llm_behavior_eval.ccpc.scorer import censorship_judge


@task
def ccpc(
    *,
    dataset_path: str | None = None,
    file_path: str | None = None,
    dataset_token: str | None = None,
    max_samples: int | None = 500,
    answer_thinking_configured: bool = False,
    judge_thinking_configured: bool = False,
    judge_revision: str | None = None,
) -> Task:
    return Task(
        dataset=load_ccpc_dataset(
            dataset_path=dataset_path,
            file_path=file_path,
            dataset_token=dataset_token,
            max_samples=max_samples,
        ),
        solver=ccpc_generate(
            answer_thinking_configured=answer_thinking_configured,
            judge_thinking_configured=judge_thinking_configured,
        ),
        scorer=censorship_judge(
            judge_thinking_configured=judge_thinking_configured,
            judge_revision=judge_revision,
        ),
        version=CCPC_TASK_VERSION,
        metadata=dict(
            task_version=CCPC_TASK_VERSION,
            dataset_repository=CCPC_DATASET_REPOSITORY,
            dataset_revision=CCPC_DATASET_REVISION,
            dataset_fingerprint=CCPC_DATASET_FINGERPRINT,
            rubric_version=CCPC_RUBRIC_VERSION,
            canonical_rows=500,
            judge_reference_model="google/gemma-4-26B-A4B-it",
            judge_model_revision=judge_revision,
            enable_thinking=False,
            judge_thinking=False,
            answer_sampling=False,
            judge_sampling=False,
            max_answer_tokens=8192,
            max_judge_tokens=4096,
            prompt_parity=PROMPT_PARITY,
        ),
    )
