from inspect_ai import Task, task

from llm_behavior_eval.common.judge import THINKING_END, THINKING_START
from llm_behavior_eval.orbench.dataset import LEGACY_SEED, dataset
from llm_behavior_eval.orbench.scorer import orbench_scorer
from llm_behavior_eval.orbench.solver import orbench_generate


@task
def orbench(
    dataset_path: str | None = None,
    dataset_token: str | None = None,
    seed: int = LEGACY_SEED,
    exclude_thinking_trace_for_judge: bool = True,
    thinking_start_token: str | None = THINKING_START,
    thinking_end_token: str | None = THINKING_END,
) -> Task:
    """OR-Bench over-refusal. Requires ``--model-role grader=...``.

    ``seed`` orders the samples as the legacy ``Dataset.shuffle(seed)`` did, so
    ``--limit`` keeps the legacy selection. No ``config=``: inference settings
    come from the user per role, or from ``parity.yaml``.
    """
    return Task(
        dataset=dataset(
            dataset_path=dataset_path, dataset_token=dataset_token, seed=seed
        ),
        solver=orbench_generate(),
        scorer=orbench_scorer(
            exclude_thinking_trace_for_judge=exclude_thinking_trace_for_judge,
            thinking_start_token=thinking_start_token,
            thinking_end_token=thinking_end_token,
        ),
    )
