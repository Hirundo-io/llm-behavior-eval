from inspect_ai import Task, task

from llm_behavior_eval.prompt_injection.dataset import LEGACY_SHUFFLE_SEED, dataset
from llm_behavior_eval.prompt_injection.scorer import prompt_injection_scorer
from llm_behavior_eval.prompt_injection.solver import solver


@task
def prompt_injection(
    dataset_path: str | None = None,
    dataset_token: str | None = None,
    seed: int = LEGACY_SHUFFLE_SEED,
    response_tokenizer: str | None = None,
    legacy_tokenizer: str | None = None,
    legacy_tokenizer_revision: str | None = None,
) -> Task:
    """Purple Llama prompt injection, judged Yes/No by the ``grader`` role.

    No ``config=`` and no default grader: inference settings for both roles come
    from the user, or from ``parity.yaml`` for the legacy settings. ``seed`` is
    the dataset shuffle seed (legacy 42); ``response_tokenizer`` optionally
    names a local tokenizer for per-sample ``response_tokens``.

    ``legacy_tokenizer`` (off by default) opts in to the legacy engine's text
    preprocessing for parity runs: it names the answer model's local tokenizer
    snapshot (or a cached repository with an immutable
    ``legacy_tokenizer_revision``), and the judge then sees the per-row judge
    question after the legacy 256-token round trip. The model's input is
    unchanged.
    """
    return Task(
        dataset=dataset(
            dataset_path=dataset_path, dataset_token=dataset_token, seed=seed
        ),
        solver=solver(legacy_tokenizer, legacy_tokenizer_revision),
        scorer=prompt_injection_scorer(response_tokenizer),
    )
