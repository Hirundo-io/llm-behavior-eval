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
    legacy_tokenizer: str | None = None,
    legacy_tokenizer_revision: str | None = None,
) -> Task:
    """BBQ split ``hirundo-io/bbq-{bias_type}-{kind}-free-text``.

    Requires ``--model-role grader=...``; no grader is bound or implied. No
    inference settings are set here (no ``config=``): pass them per role, or
    pass ``parity.yaml`` with ``--run-config`` for the legacy settings.
    ``seed`` only orders the samples (the legacy shuffle).

    ``legacy_tokenizer`` (off by default) opts in to the legacy engine's text
    preprocessing for parity runs: it names the answer model's local tokenizer
    snapshot (or a cached repository with an immutable
    ``legacy_tokenizer_revision``), and the judge then sees the legacy decoded
    prompt, target and stereotype. The model's input is unchanged.
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
        solver=bbq_solver(legacy_tokenizer, legacy_tokenizer_revision),
        scorer=bbq_scorer(),
        metrics=[bbq_metrics(kind, has_stereotype)],
    )
