"""Native Inspect HaluEval. Binds no grader and sets no inference settings."""

from inspect_ai import Task, task

from llm_behavior_eval.halueval.dataset import dataset
from llm_behavior_eval.halueval.scorer import halueval_scorer
from llm_behavior_eval.halueval.solver import halueval_solver


@task
def halueval(
    dataset_path: str | None = None,
    dataset_token: str | None = None,
    legacy_tokenizer: str | None = None,
    legacy_tokenizer_revision: str | None = None,
) -> Task:
    """HaluEval free-text hallucination benchmark; requires ``--model-role grader=...``.

    ``legacy_tokenizer`` (off by default) opts in to the legacy engine's text
    preprocessing for parity runs: it names the answer model's local tokenizer
    snapshot (or a cached repository with an immutable
    ``legacy_tokenizer_revision``), and the judge then sees the legacy decoded
    prompt and target. The model's input is unchanged.
    """
    return Task(
        dataset=dataset(dataset_path=dataset_path, dataset_token=dataset_token),
        solver=halueval_solver(legacy_tokenizer, legacy_tokenizer_revision),
        scorer=halueval_scorer(),
    )
