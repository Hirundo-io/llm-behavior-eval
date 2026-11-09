"""Native Inspect Med-Hallu. No default grader and no inference settings."""

from inspect_ai import Task, task

from llm_behavior_eval.medhallu.dataset import dataset
from llm_behavior_eval.medhallu.scorer import medhallu_scorer
from llm_behavior_eval.medhallu.solver import medhallu_solver


@task
def medhallu(
    dataset_path: str | None = None,
    dataset_token: str | None = None,
    legacy_tokenizer: str | None = None,
    legacy_tokenizer_revision: str | None = None,
) -> Task:
    """Med-Hallu; requires ``--model-role grader=...``.

    No ``config=``: inference settings come from the user (or parity.yaml).

    ``legacy_tokenizer`` (off by default) opts in to the legacy engine's text
    preprocessing for parity runs: it names the answer model's local tokenizer
    snapshot (or a cached repository with an immutable
    ``legacy_tokenizer_revision``), and the judge then sees the legacy decoded
    prompt and target. The model's input is unchanged.
    """
    return Task(
        dataset=dataset(dataset_path=dataset_path, dataset_token=dataset_token),
        solver=medhallu_solver(legacy_tokenizer, legacy_tokenizer_revision),
        scorer=medhallu_scorer(),
    )
