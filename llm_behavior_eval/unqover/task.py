from inspect_ai import Task, task

from llm_behavior_eval.common.dataset import LEGACY_SHUFFLE_SEED
from llm_behavior_eval.common.judge import THINKING_END, THINKING_START
from llm_behavior_eval.unqover.dataset import KIND, dataset
from llm_behavior_eval.unqover.metrics import unqover_metrics
from llm_behavior_eval.unqover.scorer import unqover_scorer
from llm_behavior_eval.unqover.solver import unqover_solver


@task
def unqover(
    bias_type: str = "gender",
    kind: str = KIND,
    dataset_path: str | None = None,
    revision: str | None = None,
    seed: int = LEGACY_SHUFFLE_SEED,
    exclude_thinking_trace_for_judge: bool = True,
    thinking_start_token: str | None = THINKING_START,
    thinking_end_token: str | None = THINKING_END,
) -> Task:
    """UNQOVER stereotype probing on one ambiguous (bias) split.

    Primary metric ``error`` = 1 - accuracy (lower is better). Requires
    ``--model-role grader=...``. Sets no inference settings for either role;
    pass ``--run-config llm_behavior_eval/unqover/parity.yaml`` for the legacy
    engine's settings and 500-sample cap.

    Args:
        bias_type: ``religion``, ``gender``, ``race`` or ``nationality``.
        kind: Only ``bias`` exists.
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
    # No `config=` and no grader binding: inference settings and the judge
    # always come from the user.
    return Task(
        dataset=data,
        solver=unqover_solver(),
        scorer=unqover_scorer(
            exclude_thinking_trace_for_judge=exclude_thinking_trace_for_judge,
            thinking_start_token=thinking_start_token,
            thinking_end_token=thinking_end_token,
        ),
        metrics=[unqover_metrics(has_stereotype)],
    )
