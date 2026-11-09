"""OR-Bench source validation, legacy label mapping and seeded sample order.

The legacy engine normalized the whole source, shuffled it once with
``datasets.Dataset.shuffle(seed=42)`` and then kept the first ``max_samples``
rows. Conversion here keeps every row in that same shuffled order, so Inspect's
``limit`` (``eval_config.limit`` in ``parity.yaml``) selects the legacy samples.
Sample IDs are the original source row positions, so they survive shuffling.
"""

from typing import Any, cast

from inspect_ai.dataset import MemoryDataset, Sample

from llm_behavior_eval.common.dataset import load_hf_dataset, to_inspect_dataset

DATASET_ID = "hirundo-io/or-bench"
SAFE_LABEL = "safe"
UNSAFE_LABEL = "unsafe"
LEGACY_SEED = 42


def normalize_label(raw_label: Any) -> str:
    """Map a source label to ``safe``/``unsafe``, ignoring case and whitespace."""
    label = str(raw_label).strip().lower()
    if label not in {SAFE_LABEL, UNSAFE_LABEL}:
        raise ValueError(
            "OR-Bench label must be "
            f"{SAFE_LABEL!r} or {UNSAFE_LABEL!r}, got {raw_label!r}"
        )
    return label


def source_records(source: Any) -> list[dict[str, Any]]:
    """Validate every source row; ``question`` takes precedence over ``prompt``."""
    records = []
    for index, row in enumerate(source):
        if "question" not in row and "prompt" not in row:
            raise ValueError("OR-Bench dataset requires a question or prompt column")
        if "label" not in row:
            raise ValueError("OR-Bench dataset requires a label column")
        records.append(
            {
                "id": f"{DATASET_ID}:{index}",
                "question": str(
                    row["question"] if "question" in row else row["prompt"]
                ),
                "label": normalize_label(row["label"]),
            }
        )
    return records


def legacy_order(count: int, seed: int) -> list[int]:
    """Row order produced by ``datasets.Dataset.shuffle(seed=seed)``."""
    from datasets import Dataset

    if count == 0:
        return []
    shuffled = Dataset.from_dict({"index": list(range(count))}).shuffle(seed=seed)
    return cast("list[int]", list(shuffled["index"]))


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        id=record["id"],
        input=record["question"],
        metadata={"label": record["label"], "dataset_id": DATASET_ID},
    )


def dataset(
    *,
    dataset_path: str | None = None,
    dataset_token: str | None = None,
    seed: int = LEGACY_SEED,
) -> MemoryDataset:
    """Every OR-Bench row, validated, in legacy shuffled order for ``seed``."""
    source = load_hf_dataset(
        DATASET_ID, dataset_path=dataset_path, dataset_token=dataset_token
    )
    records = source_records(source)
    ordered = [records[index] for index in legacy_order(len(records), seed)]
    return to_inspect_dataset(ordered, record_to_sample, dataset_id=DATASET_ID)
