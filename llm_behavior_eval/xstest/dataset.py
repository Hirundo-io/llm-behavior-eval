"""XSTest source validation, legacy sample order and Inspect conversion.

Sample IDs are source row positions (``hirundo-io/XSTest:<row>``). The whole
source is validated, then reordered with the legacy seed-42 shuffle, so a
sample limit (``--limit N`` or the parity config's ``eval_config.limit``)
selects the same rows the legacy engine selected with ``max_samples=N``.
"""

from typing import Any, cast

from inspect_ai.dataset import MemoryDataset, Sample

from llm_behavior_eval.common.dataset import load_hf_dataset, to_inspect_dataset

DATASET_ID = "hirundo-io/XSTest"
SAFE_LABEL = "safe"
UNSAFE_LABEL = "unsafe"
# Legacy DatasetConfig.seed default; the legacy engine shuffled before capping.
LEGACY_SHUFFLE_SEED = 42


def normalize_label(raw_label: Any) -> str:
    """Map a source label to ``safe`` or ``unsafe``; anything else is an error."""
    label = str(raw_label).strip().lower()
    if label not in {SAFE_LABEL, UNSAFE_LABEL}:
        raise ValueError(
            "XSTest label must be "
            f"{SAFE_LABEL!r} or {UNSAFE_LABEL!r}, got {raw_label!r}"
        )
    return label


def legacy_order(size: int, seed: int = LEGACY_SHUFFLE_SEED) -> list[int]:
    """Row order of the legacy ``datasets.Dataset.shuffle(seed=seed)``."""
    if size == 0:
        return []
    from datasets import Dataset

    shuffled = cast(
        "Dataset", Dataset.from_dict({"row": list(range(size))}).shuffle(seed=seed)
    )
    return cast("list[int]", list(shuffled["row"]))


def validate_records(source: Any) -> list[dict[str, Any]]:
    """Validate every source row; ``question`` is preferred over ``prompt``."""
    records = []
    for index, row in enumerate(source):
        if "question" not in row and "prompt" not in row:
            raise ValueError("XSTest dataset requires a 'question' or 'prompt' column")
        if "label" not in row:
            raise ValueError("XSTest dataset requires a 'label' column")
        records.append(
            {
                "id": f"{DATASET_ID}:{index}",
                "question": str(row.get("question", row.get("prompt"))),
                "label": normalize_label(row["label"]),
            }
        )
    return records


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        id=record["id"],
        input=record["question"],
        metadata={"label": record["label"], "dataset_id": DATASET_ID},
    )


def dataset(
    *, dataset_path: str | None = None, dataset_token: str | None = None
) -> MemoryDataset:
    records = validate_records(
        load_hf_dataset(
            DATASET_ID, dataset_path=dataset_path, dataset_token=dataset_token
        )
    )
    records = [records[index] for index in legacy_order(len(records))]
    return to_inspect_dataset(records, record_to_sample, dataset_id=DATASET_ID)
