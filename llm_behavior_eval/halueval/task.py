"""Native Inspect HaluEval. Binds no grader and sets no inference settings."""

from inspect_ai import Task, task

from llm_behavior_eval.halueval.dataset import dataset
from llm_behavior_eval.halueval.scorer import halueval_scorer
from llm_behavior_eval.halueval.solver import halueval_solver


@task
def halueval(dataset_path: str | None = None, dataset_token: str | None = None) -> Task:
    """HaluEval free-text hallucination benchmark; requires ``--model-role grader=...``."""
    return Task(
        dataset=dataset(dataset_path=dataset_path, dataset_token=dataset_token),
        solver=halueval_solver(),
        scorer=halueval_scorer(),
    )
