"""Native Inspect Med-Hallu. No default grader and no inference settings."""

from inspect_ai import Task, task

from llm_behavior_eval.medhallu.dataset import dataset
from llm_behavior_eval.medhallu.scorer import medhallu_scorer
from llm_behavior_eval.medhallu.solver import medhallu_solver


@task
def medhallu(dataset_path: str | None = None, dataset_token: str | None = None) -> Task:
    # No `config=`: inference settings come from the user (or parity.yaml).
    return Task(
        dataset=dataset(dataset_path=dataset_path, dataset_token=dataset_token),
        solver=medhallu_solver(),
        scorer=medhallu_scorer(),
    )
