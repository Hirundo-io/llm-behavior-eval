"""Native BBQ; one parameterized task selects one of the twelve splits."""

from inspect_ai import Task, task

from llm_behavior_eval.bbq.dataset import LEGACY_SHUFFLE_SEED, dataset
from llm_behavior_eval.bbq.metrics import bbq_metrics
from llm_behavior_eval.bbq.scorer import bbq_scorer
from llm_behavior_eval.bbq.solver import bbq_solver


@task
def bbq(
    bias_type: str = "gender",
    kind: str = "bias",
    dataset_path: str | None = None,
    dataset_token: str | None = None,
    seed: int = LEGACY_SHUFFLE_SEED,
) -> Task:
    """BBQ split ``hirundo-io/bbq-{bias_type}-{kind}-free-text``.

    Requires ``--model-role grader=...``; no grader is bound or implied. No
    inference settings are set here (no ``config=``): pass them per role, or
    pass ``parity.yaml`` with ``--run-config`` for the legacy settings.
    ``seed`` only orders the samples (the legacy shuffle).
    """
    data = dataset(
        bias_type,
        kind,
        dataset_path=dataset_path,
        dataset_token=dataset_token,
        seed=seed,
    )
    has_stereotype = (
        bool((data[0].metadata or {})["has_stereotype"]) if len(data) else False
    )
    return Task(
        dataset=data,
        solver=bbq_solver(),
        scorer=bbq_scorer(),
        metrics=[bbq_metrics(kind, has_stereotype)],
    )
