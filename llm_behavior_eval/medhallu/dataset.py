"""Med-Hallu identity and ordered, offline-capable dataset conversion."""

from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample

from llm_behavior_eval.common.dataset import load_hf_dataset, to_inspect_dataset

DATASET_ID = "hirundo-io/medhallu"
# Legacy BaseEvaluator.prepare_dataloader shuffles with DatasetConfig.seed (42)
# before capping at max_samples.
LEGACY_SHUFFLE_SEED = 42
PARITY_GAPS = [
    "Question is raw text, not decoded chat-templated input",
    "Input is not legacy-tokenizer truncated to 1024 tokens",
    "Target is not legacy-tokenizer truncated to 256 tokens and decoded",
]


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
    """Validate every row, then order samples exactly as the legacy shuffle did.

    Sample IDs are stable zero-based source row positions. The returned dataset
    holds the full split in legacy order, so Inspect's ``limit`` (for example
    ``eval_config.limit`` in ``parity.yaml``) selects the legacy samples.
    """
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
                f"Med-Hallu row {index} requires nonempty question and answer strings"
            )
        records.append({**row, "source_index": index})
    if records:
        from datasets import Dataset

        shuffled = Dataset.from_list(records).shuffle(seed=LEGACY_SHUFFLE_SEED)
        records = list(shuffled)
    return to_inspect_dataset(records, record_to_sample, dataset_id=DATASET_ID)
