from inspect_ai import Task, task

from llm_behavior_eval.bloom.dataset import dataset
from llm_behavior_eval.bloom.metrics import bloom_metrics
from llm_behavior_eval.bloom.scorer import bloom_scorer
from llm_behavior_eval.bloom.solver import solver
from llm_behavior_eval.common.dataset import LEGACY_SHUFFLE_SEED
from llm_behavior_eval.common.judge import THINKING_END, THINKING_START


@task
def bloom(
    bias_type: str = "gender",
    kind: str = "bias",
    dataset_path: str | None = None,
    revision: str | None = None,
    seed: int = LEGACY_SHUFFLE_SEED,
    exclude_thinking_trace_for_judge: bool = True,
    thinking_start_token: str | None = THINKING_START,
    thinking_end_token: str | None = THINKING_END,
) -> Task:
    """One BLOOM split: ``bias_type`` age|gender|race, ``kind`` bias|unbias|ambiguous.

    Requires an explicit ``grader`` role. No ``config=``: inference settings come
    from the user per role, or from ``bloom/parity.yaml`` for legacy parity.

    Args:
        bias_type: ``age``, ``gender`` or ``race``.
        kind: ``bias``, ``unbias`` or ``ambiguous``.
        dataset_path: Local source or ``save_to_disk`` snapshot to read instead
            of the HF Hub; the logical dataset id is unchanged.
        revision: HF Hub dataset revision (branch, tag or commit).
        seed: Sample order only: the legacy shuffle seed.
        exclude_thinking_trace_for_judge: Strip the answer's inline thinking
            trace before the grader sees it (``common.judge.THINKING_TASK_ARGS``).
        thinking_start_token: Marker that opens the trace.
        thinking_end_token: Marker that closes the trace.

    There is no token argument: Inspect records every task argument in the
    log, so gated Hub datasets authenticate through ``HF_TOKEN`` (or a cached
    ``huggingface-cli login``) instead.
    """
    data = dataset(
        bias_type,
        kind,
        dataset_path=dataset_path,
        revision=revision,
        seed=seed,
    )
    has_stereotype = (
        bool((data[0].metadata or {})["has_stereotype"]) if len(data) else False
    )
    return Task(
        dataset=data,
        solver=solver(),
        scorer=bloom_scorer(
            exclude_thinking_trace_for_judge=exclude_thinking_trace_for_judge,
            thinking_start_token=thinking_start_token,
            thinking_end_token=thinking_end_token,
        ),
        metrics=[
            bloom_metrics("bias" if kind == "ambiguous" else kind, has_stereotype)
        ],
    )
