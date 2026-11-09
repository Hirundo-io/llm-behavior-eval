from inspect_ai import Task, task

from llm_behavior_eval.bloom.dataset import LEGACY_SHUFFLE_SEED, dataset
from llm_behavior_eval.bloom.metrics import bloom_metrics
from llm_behavior_eval.bloom.scorer import bloom_scorer
from llm_behavior_eval.bloom.solver import solver


@task
def bloom(
    bias_type: str = "gender",
    kind: str = "bias",
    dataset_path: str | None = None,
    dataset_token: str | None = None,
    seed: int = LEGACY_SHUFFLE_SEED,
) -> Task:
    """One BLOOM split: ``bias_type`` age|gender|race, ``kind`` bias|unbias|ambiguous.

    Requires an explicit ``grader`` role. No ``config=``: inference settings come
    from the user per role, or from ``bloom/parity.yaml`` for legacy parity.
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
        solver=solver(),
        scorer=bloom_scorer(),
        metrics=[
            bloom_metrics("bias" if kind == "ambiguous" else kind, has_stereotype)
        ],
    )
