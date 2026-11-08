"""Shared refusal source validation; IDs are source row positions before capping."""

from typing import cast

from inspect_ai.dataset import MemoryDataset, Sample

from llm_behavior_eval.common.dataset import load_hf_dataset, to_inspect_dataset
from llm_behavior_eval.evaluation_utils.refusal_utils import normalize_refusal_label


def refusal_dataset(
    dataset_id: str,
    *,
    dataset_path: str | None = None,
    dataset_token: str | None = None,
    max_samples: int | None = 500,
    seed: int | None = None,
) -> MemoryDataset:
    source = load_hf_dataset(
        dataset_id, dataset_path=dataset_path, dataset_token=dataset_token
    )
    records = []
    for index, row in enumerate(source):
        if "question" not in row and "prompt" not in row:
            raise ValueError("Refusal dataset requires question or prompt")
        records.append(
            dict(
                id=f"{dataset_id}:{index}",
                question=str(row.get("question", row.get("prompt"))),
                label=normalize_refusal_label(row["label"]),
            )
        )
    # Optional to preserve existing callers; OR-Bench requests the legacy seed.
    if seed is not None and records:
        from datasets import Dataset

        order = cast(
            "Dataset",
            Dataset.from_dict({"index": list(range(len(records)))}).shuffle(seed=seed),
        )
        records = [records[index] for index in cast("list[int]", list(order["index"]))]
    return to_inspect_dataset(
        records,
        lambda row: Sample(
            id=row["id"],
            input=row["question"],
            metadata={"label": row["label"], "dataset_id": dataset_id},
        ),
        dataset_id=dataset_id,
        max_samples=max_samples,
    )
