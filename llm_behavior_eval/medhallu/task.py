"""Native Inspect Med-Hallu. No default grader and no inference settings."""

from inspect_ai import Task, task

from llm_behavior_eval.common.judge import THINKING_END, THINKING_START
from llm_behavior_eval.medhallu.dataset import dataset
from llm_behavior_eval.medhallu.scorer import medhallu_scorer
from llm_behavior_eval.medhallu.solver import medhallu_solver


@task
def medhallu(
    dataset_path: str | None = None,
    revision: str | None = None,
    exclude_thinking_trace_for_judge: bool = True,
    thinking_start_token: str | None = THINKING_START,
    thinking_end_token: str | None = THINKING_END,
) -> Task:
    """Med-Hallu medical hallucination benchmark; requires ``--model-role grader=...``.

    Args:
        dataset_path: HF path or ``save_to_disk`` snapshot to read instead of
            the Hub.
        revision: Hub dataset revision (branch, tag or commit) to pin.
        exclude_thinking_trace_for_judge: Strip the answer's inline thinking
            trace before the grader sees it (``common.judge.THINKING_TASK_ARGS``).
        thinking_start_token: Marker that opens the trace.
        thinking_end_token: Marker that closes the trace.

    There is no token argument: Inspect records every task argument in the
    log, so gated Hub datasets authenticate through ``HF_TOKEN`` (or a cached
    ``huggingface-cli login``) instead.
    """
    # No `config=`: inference settings come from the user (or parity.yaml).
    return Task(
        dataset=dataset(dataset_path=dataset_path, revision=revision),
        solver=medhallu_solver(),
        scorer=medhallu_scorer(
            exclude_thinking_trace_for_judge=exclude_thinking_trace_for_judge,
            thinking_start_token=thinking_start_token,
            thinking_end_token=thinking_end_token,
        ),
    )
