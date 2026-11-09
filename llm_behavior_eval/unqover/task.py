from inspect_ai import Task, task

from llm_behavior_eval.unqover.dataset import KIND, LEGACY_SEED, dataset
from llm_behavior_eval.unqover.metrics import unqover_metrics
from llm_behavior_eval.unqover.scorer import unqover_scorer
from llm_behavior_eval.unqover.solver import unqover_solver


@task
def unqover(
    bias_type: str = "gender",
    kind: str = KIND,
    dataset_path: str | None = None,
    dataset_token: str | None = None,
    seed: int = LEGACY_SEED,
) -> Task:
    """UNQOVER stereotype probing on one ambiguous (bias) split.

    Primary metric ``error`` = 1 - accuracy (lower is better). Requires
    ``--model-role grader=...``. Sets no inference settings for either role;
    pass ``--run-config llm_behavior_eval/unqover/parity.yaml`` for the legacy
    engine's settings and 500-sample cap.
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
    # No `config=` and no grader binding: inference settings and the judge
    # always come from the user.
    return Task(
        dataset=data,
        solver=unqover_solver(),
        scorer=unqover_scorer(),
        metrics=[unqover_metrics(has_stereotype)],
    )
