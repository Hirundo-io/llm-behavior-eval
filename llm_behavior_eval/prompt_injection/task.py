from inspect_ai import Task, task

from llm_behavior_eval.common.dataset import LEGACY_SHUFFLE_SEED
from llm_behavior_eval.common.judge import THINKING_END, THINKING_START
from llm_behavior_eval.prompt_injection.dataset import dataset
from llm_behavior_eval.prompt_injection.scorer import prompt_injection_scorer
from llm_behavior_eval.prompt_injection.solver import solver


@task
def prompt_injection(
    dataset_path: str | None = None,
    revision: str | None = None,
    seed: int = LEGACY_SHUFFLE_SEED,
    response_tokenizer: str | None = None,
    exclude_thinking_trace_for_judge: bool = True,
    thinking_start_token: str | None = THINKING_START,
    thinking_end_token: str | None = THINKING_END,
) -> Task:
    """Purple Llama prompt injection, judged Yes/No by the ``grader`` role.

    No ``config=`` and no default grader: inference settings for both roles come
    from the user, or from ``parity.yaml`` for the legacy settings.

    Args:
        dataset_path: Local ``save_to_disk`` snapshot or alternative HF source.
        revision: Hub dataset revision (branch, tag or commit) to pin.
        seed: Dataset shuffle seed (legacy 42).
        response_tokenizer: Local tokenizer for per-sample ``response_tokens``.
        exclude_thinking_trace_for_judge: Strip the answer's inline thinking
            trace before the grader sees it (``common.judge.THINKING_TASK_ARGS``).
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
        solver=solver(),
        scorer=prompt_injection_scorer(
            response_tokenizer,
            exclude_thinking_trace_for_judge=exclude_thinking_trace_for_judge,
            thinking_start_token=thinking_start_token,
            thinking_end_token=thinking_end_token,
        ),
    )
