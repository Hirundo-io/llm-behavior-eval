from inspect_ai import Task, task

from llm_behavior_eval.common.judge import THINKING_END, THINKING_START
from llm_behavior_eval.xstest.dataset import dataset
from llm_behavior_eval.xstest.scorer import xstest_scorer
from llm_behavior_eval.xstest.solver import xstest_solver


@task
def xstest(
    dataset_path: str | None = None,
    dataset_token: str | None = None,
    exclude_thinking_trace_for_judge: bool = True,
    thinking_start_token: str | None = THINKING_START,
    thinking_end_token: str | None = THINKING_END,
) -> Task:
    """XSTest over-refusal: safe/unsafe refusal rates judged by the grader role.

    No `config=` and no default grader: inference settings for both roles come
    from the user, or from this package's `parity.yaml`.
    """
    return Task(
        dataset=dataset(dataset_path=dataset_path, dataset_token=dataset_token),
        solver=xstest_solver(),
        scorer=xstest_scorer(
            exclude_thinking_trace_for_judge=exclude_thinking_trace_for_judge,
            thinking_start_token=thinking_start_token,
            thinking_end_token=thinking_end_token,
        ),
    )
