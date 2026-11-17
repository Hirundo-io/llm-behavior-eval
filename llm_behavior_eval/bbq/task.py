"""Native BBQ; one parameterized task selects one of the twelve splits."""

from inspect_ai import Task, task

from llm_behavior_eval.bbq.dataset import dataset
from llm_behavior_eval.bbq.metrics import bbq_metrics
from llm_behavior_eval.bbq.scorer import bbq_scorer
from llm_behavior_eval.bbq.solver import bbq_solver
from llm_behavior_eval.common.dataset import LEGACY_SHUFFLE_SEED
from llm_behavior_eval.common.judge import THINKING_END, THINKING_START


@task
def bbq(
    bias_type: str = "gender",
    kind: str = "bias",
    dataset_path: str | None = None,
    revision: str | None = None,
    seed: int = LEGACY_SHUFFLE_SEED,
    exclude_thinking_trace_for_judge: bool = True,
    thinking_start_token: str | None = THINKING_START,
    thinking_end_token: str | None = THINKING_END,
) -> Task:
    """BBQ split ``hirundo-io/bbq-{bias_type}-{kind}-free-text``.

    Requires ``--model-role grader=...``; no grader is bound or implied. No
    inference settings are set here (no ``config=``): pass them per role, or
    pass ``parity.yaml`` with ``--run-config`` for the legacy settings.

    Args:
        bias_type: Bias category of the split.
        kind: ``bias`` (ambiguous contexts) or ``unbias`` (disambiguated).
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
        solver=bbq_solver(),
        scorer=bbq_scorer(
            exclude_thinking_trace_for_judge=exclude_thinking_trace_for_judge,
            thinking_start_token=thinking_start_token,
            thinking_end_token=thinking_end_token,
        ),
        metrics=[bbq_metrics(kind, has_stereotype)],
    )
