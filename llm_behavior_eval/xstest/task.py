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
    legacy_tokenizer: str | None = None,
    legacy_tokenizer_revision: str | None = None,
) -> Task:
    """XSTest over-refusal: safe/unsafe refusal rates judged by the grader role.

    No `config=` and no default grader: inference settings for both roles come
    from the user, or from this package's `parity.yaml`.

    ``legacy_tokenizer`` (off by default) opts in to the legacy engine's text
    preprocessing for parity runs: it names the answer model's local tokenizer
    snapshot (or a cached repository with an immutable
    ``legacy_tokenizer_revision``), and the judge then sees the legacy decoded
    prompt as the question. The model's input is unchanged.
    """
    return Task(
        dataset=dataset(dataset_path=dataset_path, dataset_token=dataset_token),
        solver=xstest_solver(legacy_tokenizer, legacy_tokenizer_revision),
        scorer=xstest_scorer(
            exclude_thinking_trace_for_judge=exclude_thinking_trace_for_judge,
            thinking_start_token=thinking_start_token,
            thinking_end_token=thinking_end_token,
        ),
    )
