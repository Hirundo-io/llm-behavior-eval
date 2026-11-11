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
from llm_behavior_eval.common.judge import THINKING_END, THINKING_START

CCPC_REFERENCE_JUDGE = "google/gemma-4-26B-A4B-it"


@task
def ccpc(
    *,
    dataset_path: str | None = None,
    file_path: str | None = None,
    judge_revision: str | None = None,
    exclude_thinking_trace_for_judge: bool = True,
    thinking_start_token: str | None = THINKING_START,
    thinking_end_token: str | None = THINKING_END,
) -> Task:
    """CCPC v6 censorship rate, judged by the required ``grader`` role.

    The dataset revision and fingerprint are pinned by ``contract.py``; there is
    no revision override.

    Args:
        dataset_path: Local source; validated against the pinned fingerprint.
        file_path: Alias of ``dataset_path``; the two must agree.
        judge_revision: Declared judge revision, recorded in metadata.
        exclude_thinking_trace_for_judge: Strip the answer's inline thinking
            trace before judging (``common.judge.THINKING_TASK_ARGS``).
        thinking_start_token: Marker that opens the trace.
        thinking_end_token: Marker that closes the trace.

    There is no token argument: Inspect records every task argument in the
    log, so gated Hub datasets authenticate through ``HF_TOKEN`` (or a cached
    ``huggingface-cli login``) instead.
    """
    # No `config=` and no default grader: inference settings for both roles
    # come from the user or from parity.yaml.
    return Task(
        dataset=load_ccpc_dataset(
            dataset_path=dataset_path,
            file_path=file_path,
        ),
        solver=solver(),
        scorer=ccpc_scorer(
            judge_revision=judge_revision,
            exclude_thinking_trace_for_judge=exclude_thinking_trace_for_judge,
            thinking_start_token=thinking_start_token,
            thinking_end_token=thinking_end_token,
        ),
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
