"""Benchmark-owned dataset conversion."""

from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample

from llm_behavior_eval.common.dataset import to_inspect_dataset

DATASET_ID = "inspect_template"


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(id=record["id"], input=record["question"], target=record["answer"])


def dataset() -> MemoryDataset:
    # Real benchmarks call common.dataset.load_hf_dataset and validate the
    # complete source before conversion.
    records = [{"id": "addition", "question": "What is 1 + 1?", "answer": "2"}]
    return to_inspect_dataset(records, record_to_sample, dataset_id=DATASET_ID)
