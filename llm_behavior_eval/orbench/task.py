from inspect_ai import Task, task

from llm_behavior_eval.common.dataset import LEGACY_SHUFFLE_SEED
from llm_behavior_eval.common.judge import THINKING_END, THINKING_START
from llm_behavior_eval.orbench.dataset import dataset
from llm_behavior_eval.orbench.scorer import orbench_scorer
from llm_behavior_eval.orbench.solver import orbench_generate


@task
def orbench(
    dataset_path: str | None = None,
    revision: str | None = None,
    seed: int = LEGACY_SHUFFLE_SEED,
    exclude_thinking_trace_for_judge: bool = True,
    thinking_start_token: str | None = THINKING_START,
    thinking_end_token: str | None = THINKING_END,
) -> Task:
    """OR-Bench over-refusal. Requires ``--model-role grader=...``.

    ``seed`` orders the samples as the legacy ``Dataset.shuffle(seed)`` did, so
    ``--limit`` keeps the legacy selection. No ``config=``: inference settings
    come from the user per role, or from ``parity.yaml``. The model under test
    must report stop reasons (not Inspect's ``hf/`` provider).

    Args:
        dataset_path: Local snapshot or other loading source for the dataset.
        revision: Hub dataset revision (branch, tag or commit).
        seed: Legacy ``Dataset.shuffle`` seed that orders the samples.
        exclude_thinking_trace_for_judge: Strip the answer's inline thinking
            trace before judging (``common.judge.THINKING_TASK_ARGS``).
        thinking_start_token: Marker that opens the trace.
        thinking_end_token: Marker that closes the trace.

    There is no token argument: Inspect records every task argument in the
    log, so gated Hub datasets authenticate through ``HF_TOKEN`` (or a cached
    ``huggingface-cli login``) instead.
    """
    return Task(
        dataset=dataset(
            dataset_path=dataset_path,
            revision=revision,
            seed=seed,
        ),
        solver=orbench_generate(),
        scorer=orbench_scorer(
            exclude_thinking_trace_for_judge=exclude_thinking_trace_for_judge,
            thinking_start_token=thinking_start_token,
            thinking_end_token=thinking_end_token,
        ),
    )
