from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample

from llm_behavior_eval.common.dataset import to_inspect_dataset


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(id=record["id"], input=record["question"], target=record["answer"])


def dataset(max_samples: int | None = 500) -> MemoryDataset:
    return to_inspect_dataset(
        [{"id": "addition", "question": "What is 1 + 1?", "answer": "2"}],
        record_to_sample,
        dataset_id="inspect_template",
        max_samples=max_samples,
    )
