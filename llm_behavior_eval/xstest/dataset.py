from inspect_ai.dataset import MemoryDataset

from llm_behavior_eval.evaluation_utils.refusal_utils import XSTEST_DATASET
from llm_behavior_eval.refusal.dataset import refusal_dataset


def dataset(
    max_samples: int | None = 500,
    *,
    dataset_path: str | None = None,
    dataset_token: str | None = None,
) -> MemoryDataset:
    return refusal_dataset(
        XSTEST_DATASET,
        dataset_path=dataset_path,
        dataset_token=dataset_token,
        max_samples=max_samples,
        seed=42,
    )
