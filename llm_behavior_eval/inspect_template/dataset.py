"""Benchmark-owned dataset conversion."""

from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample

from llm_behavior_eval.common.dataset import to_inspect_dataset

DATASET_ID = "inspect_template"


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(id=record["id"], input=record["question"], target=record["answer"])


def dataset() -> MemoryDataset:
    # Real one-record-to-one-sample Hub benchmarks call Inspect's native
    # hf_dataset(path, split=..., name=..., revision=...,
    # sample_fields=record_to_sample, shuffle=LEGACY_SHUFFLE_SEED,
    # token=resolve_dataset_token()). Snapshots, full-source validation and
    # expanding conversions use common.dataset.load_hf_dataset (explicit
    # split=), legacy_order and to_inspect_dataset.
    records = [{"id": "addition", "question": "What is 1 + 1?", "answer": "2"}]
    return to_inspect_dataset(records, record_to_sample, dataset_id=DATASET_ID)
