"""HaluEval source validation and legacy-ordered dataset conversion."""

from typing import Any, cast

from inspect_ai.dataset import MemoryDataset, Sample

from llm_behavior_eval.common.dataset import load_hf_dataset, to_inspect_dataset

DATASET_ID = "hirundo-io/halueval"
# Legacy DatasetConfig.seed: the legacy engine shuffled the full split with this
# seed and then kept the first max_samples rows.
LEGACY_SHUFFLE_SEED = 42
PARITY_GAPS = [
    "Question is raw text, not decoded chat-templated input",
    "Input is not legacy-tokenizer truncated to 1024 tokens",
    "Target is not legacy-tokenizer truncated to 256 tokens and decoded",
]


def legacy_order(num_rows: int) -> list[int]:
    """Source row positions in legacy order: ``Dataset.shuffle(seed=42)``.

    Inspect's ``--limit N`` (or ``eval_config.limit`` in ``parity.yaml``) then
    keeps the first N, the same rows the legacy ``max_samples`` cap selected.
    """
    from datasets import Dataset

    if num_rows == 0:
        return []
    rows = Dataset.from_dict({"index": list(range(num_rows))})
    return cast("list[int]", list(rows.shuffle(seed=LEGACY_SHUFFLE_SEED)["index"]))


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        id=f"{DATASET_ID}:{record['source_index']}",
        input=record["question"],
        target=record["answer"],
        metadata={
            "dataset_id": DATASET_ID,
            "source_index": record["source_index"],
            "parity_gaps": PARITY_GAPS,
        },
    )


def dataset(
    *, dataset_path: str | None = None, dataset_token: str | None = None
) -> MemoryDataset:
    """Validate every source row, then convert all rows in legacy order.

    Sample IDs are ``hirundo-io/halueval:<zero-based source row>``, stable under
    any limit.
    """
    source = load_hf_dataset(
        DATASET_ID, dataset_path=dataset_path, dataset_token=dataset_token
    )
    rows = [dict(source[i]) for i in range(len(source))]
    for index, row in enumerate(rows):
        if any(
            not isinstance(row.get(key), str) or not row[key]
            for key in ("question", "answer")
        ):
            raise ValueError(
                f"HaluEval row {index} requires nonempty question and answer strings"
            )
    records = [
        {**rows[index], "source_index": index} for index in legacy_order(len(rows))
    ]
    return to_inspect_dataset(records, record_to_sample, dataset_id=DATASET_ID)
