"""HaluEval identity and ordered, offline-capable dataset conversion."""

from typing import cast

from datasets import Dataset
from inspect_ai.dataset import MemoryDataset, Sample

from llm_behavior_eval.common.dataset import load_hf_dataset, to_inspect_dataset

DATASET_ID = "hirundo-io/halueval"
PARITY_GAPS = [
    "Question is raw text, not decoded chat-templated input",
    "Input is not legacy-tokenizer truncated to 1024 tokens",
    "Target is not legacy-tokenizer truncated to 256 tokens and decoded",
]


def dataset(
    max_samples: int | None = 500,
    *,
    dataset_path: str | None = None,
    dataset_token: str | None = None,
) -> MemoryDataset:
    """Validate the full source before capping; IDs are stable source row positions."""
    source = load_hf_dataset(
        DATASET_ID, dataset_path=dataset_path, dataset_token=dataset_token
    )
    records = []
    for index, row in enumerate(source):
        if any(
            not isinstance(row.get(key), str) or not row[key]
            for key in ("question", "answer")
        ):
            raise ValueError(
                f"HaluEval row {index} requires nonempty question and answer strings"
            )
        records.append({**row, "source_index": index})
    if records:
        shuffled = cast("Dataset", Dataset.from_list(records).shuffle(seed=42))
        records = list(shuffled)
    return to_inspect_dataset(
        records,
        lambda row: Sample(
            id=f"{DATASET_ID}:{row['source_index']}",
            input=row["question"],
            target=row["answer"],
            metadata={
                "dataset_id": DATASET_ID,
                "source_index": row["source_index"],
                "parity_gaps": PARITY_GAPS,
            },
        ),
        dataset_id=DATASET_ID,
        max_samples=max_samples,
    )
