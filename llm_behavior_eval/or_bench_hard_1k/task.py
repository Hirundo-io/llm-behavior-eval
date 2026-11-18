"""OR-Bench Hard-1K with its Hugging Face dataset pinned to a commit.

The upstream task (``or_bench/or_bench_hard_1k`` in ``inspect-evals-orbench``) loads
``bench-llm/or-bench`` with no revision and takes no argument to set one. This task is
the upstream one with only that changed: it reuses the upstream sample conversion,
solver and scorer, and loads the dataset at ``DATASET_REVISION``.
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import hf_dataset
from inspect_ai.solver import generate

DATASET_CONFIG = "or-bench-hard-1k"
DATASET_REVISION = "<SHA>"


@task
def or_bench_hard_1k(limit: int | None = None, judge: str | None = None) -> Task:
    """OR-Bench Hard-1K over-refusal on a pinned dataset commit.

    Args:
        limit: Maximum number of samples to load.
        judge: Judge model; unset, OR-Bench judges with the ``grader`` role or else the
            model under test.

    Returns:
        The upstream task, with the dataset at ``DATASET_REVISION``.
    """
    # Imported here so the registry loads without the optional `or-bench` extra.
    from or_bench.or_bench import DATASET_PATH, or_bench_refusal, record_to_sample

    return Task(
        dataset=hf_dataset(
            DATASET_PATH,
            split="train",
            name=DATASET_CONFIG,
            revision=DATASET_REVISION,
            sample_fields=record_to_sample,
            limit=limit,
        ),
        solver=generate(),
        scorer=or_bench_refusal(judge=judge),
    )
