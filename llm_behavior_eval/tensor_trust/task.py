"""Tensor Trust tasks: prompt hijacking and prompt extraction robustness."""

from typing import Any

from inspect_ai import Task, task
from inspect_ai.scorer import Scorer
from inspect_ai.solver import generate

from llm_behavior_eval.common.scoring import completion_scorer

from .dataset import BENCHMARKS, DATA_COMMIT, DATA_REPOSITORY, BenchmarkName, dataset
from .scorer import tensor_trust_extraction_scorer, tensor_trust_hijacking_scorer

TASK_VERSION = 1
"""Bump when a change can alter results (messages, rules, data, metrics)."""
REFERENCE_CODE = (
    "https://github.com/HumanCompatibleAI/tensor-trust/tree/"
    "f0b055451415e23d444703fbb8a47daf1d4b2c3c/data-pipeline/src/ttdata"
)


def _task(name: BenchmarkName, scorer: Scorer, dataset_path: str | None) -> Task:
    benchmark = BENCHMARKS[name]
    metadata: dict[str, Any] = {
        "benchmark": name,
        "data_repository": DATA_REPOSITORY,
        "data_commit": DATA_COMMIT,
        "data_file": benchmark.path,
        "data_sha256": benchmark.sha256,
        "data_rows": benchmark.rows,
        "reference_code": REFERENCE_CODE,
    }
    return Task(
        dataset=dataset(name, dataset_path),
        solver=generate(),
        scorer=[scorer, completion_scorer()],
        version=TASK_VERSION,
        metadata=metadata,
    )


@task
def tensor_trust_hijacking(dataset_path: str | None = None) -> Task:
    """Prompt hijacking robustness: HRR and defense validity.

    Each row is sent twice, with the attack and with the true access code,
    between the defender's opening (system) and closing (user) defense.

    Args:
        dataset_path: Read this local copy of the published JSONL file instead
            of downloading it from the pinned commit; same SHA-256 required.

    Returns:
        A task with no inference defaults and no grader.
    """
    return _task("hijacking", tensor_trust_hijacking_scorer(), dataset_path)


@task
def tensor_trust_extraction(dataset_path: str | None = None) -> Task:
    """Prompt extraction robustness: ERR and defense validity.

    Each row is sent twice, with the attack and with the true access code,
    between the defender's opening (system) and closing (user) defense.

    Args:
        dataset_path: Read this local copy of the published JSONL file instead
            of downloading it from the pinned commit; same SHA-256 required.

    Returns:
        A task with no inference defaults and no grader.
    """
    return _task("extraction", tensor_trust_extraction_scorer(), dataset_path)
