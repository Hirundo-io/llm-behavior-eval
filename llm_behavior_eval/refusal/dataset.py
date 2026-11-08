"""Shared refusal source validation; IDs are source row positions before capping."""

from inspect_ai.dataset import MemoryDataset, Sample

from llm_behavior_eval.common.dataset import load_hf_dataset, to_inspect_dataset
from llm_behavior_eval.evaluation_utils.refusal_utils import normalize_refusal_label


def refusal_dataset(
    dataset_id: str,
    *,
    dataset_path: str | None = None,
    dataset_token: str | None = None,
    max_samples: int | None = 500,
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
