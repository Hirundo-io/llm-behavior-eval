"""Validated CCPC source order and stable IDs; cap only after validation."""

import hashlib
import json
from pathlib import Path

from datasets import Dataset, load_dataset
from inspect_ai.dataset import MemoryDataset, Sample

from llm_behavior_eval.ccpc.contract import (
    CCPC_DATASET_CONFIG,
    CCPC_DATASET_FINGERPRINT,
    CCPC_DATASET_ID,
    CCPC_DATASET_REPOSITORY,
    CCPC_DATASET_REVISION,
    CCPC_DATASET_SPLIT,
    CCPC_EXPECTED_ROWS,
)
from llm_behavior_eval.common.dataset import load_hf_dataset, to_inspect_dataset


def censorship_dataset_fingerprint(dataset: Dataset) -> str:
    """Fingerprint every benchmark ID and question in published row order."""
    rows = [
        {
            "question": dataset[index]["question"],
            "benchmark_id": dataset[index]["benchmark_id"],
        }
        for index in range(len(dataset))
    ]
    serialized = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_ccpc_dataset(
    *,
    dataset_path: str | None = None,
    file_path: str | None = None,
    dataset_token: str | None = None,
    max_samples: int | None = 500,
) -> MemoryDataset:
    if dataset_path and file_path and dataset_path != file_path:
        raise ValueError("dataset_path and file_path disagree")
    source = dataset_path or file_path
    if source in (None, CCPC_DATASET_ID, CCPC_DATASET_REPOSITORY):
        dataset = load_dataset(
            CCPC_DATASET_REPOSITORY,
            name=CCPC_DATASET_CONFIG,
            split=CCPC_DATASET_SPLIT,
            revision=CCPC_DATASET_REVISION,
            token=dataset_token,
        )
    else:
        path = Path(source)
        if (path / "dataset_dict.json").exists():
            from datasets import load_from_disk

            dataset = load_from_disk(source)[CCPC_DATASET_SPLIT]
        else:
            dataset = load_hf_dataset(
                CCPC_DATASET_ID, dataset_path=source, dataset_token=dataset_token
            )
    expected_columns = ("question", "benchmark_id")
    if not isinstance(dataset, Dataset):
        raise ValueError("CCPC-Bench must load as one Dataset split.")
    if set(dataset.column_names) != set(expected_columns):
        raise ValueError(
            f"CCPC-Bench columns must be exactly {list(expected_columns)}; "
            f"found {dataset.column_names}."
        )
    if len(dataset) != CCPC_EXPECTED_ROWS:
        raise ValueError(
            f"CCPC-Bench must contain exactly {CCPC_EXPECTED_ROWS} rows; "
            f"found {len(dataset)}."
        )

    benchmark_ids: list[str] = []
    for index in range(len(dataset)):
        row = dataset[index]
        for column in expected_columns:
            value = row[column]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"CCPC-Bench row {index} field {column!r} must be a "
                    "non-empty string."
                )
        benchmark_ids.append(row["benchmark_id"])
    if len(set(benchmark_ids)) != CCPC_EXPECTED_ROWS:
        raise ValueError("CCPC-Bench benchmark_id values must be unique.")
    fingerprint = censorship_dataset_fingerprint(dataset)
    if fingerprint != CCPC_DATASET_FINGERPRINT:
        raise ValueError(
            "CCPC-Bench rows do not match the pinned published order and contents; "
            f"expected fingerprint {CCPC_DATASET_FINGERPRINT}, found {fingerprint}."
        )
    return to_inspect_dataset(
        dataset, record_to_sample, dataset_id=CCPC_DATASET_ID, max_samples=max_samples
    )


def record_to_sample(record: dict) -> Sample:
    # Upstream record_to_sample, with legacy trailing newline applied by solver.
    return Sample(
        input=record["question"],
        id=record["benchmark_id"],
        metadata={"benchmark_id": record["benchmark_id"]},
    )
