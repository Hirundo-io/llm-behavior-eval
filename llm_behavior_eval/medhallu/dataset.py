"""Med-Hallu identity and ordered, offline-capable dataset conversion."""

from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_behavior_eval.common.dataset import (
    legacy_order,
    load_hf_dataset,
    to_inspect_dataset,
)

DATASET_ID = "hirundo-io/medhallu"
PARITY_GAPS = [
    "Question is raw text, not decoded chat-templated input",
    "Input is not legacy-tokenizer truncated to 1024 tokens",
    "Target is not legacy-tokenizer truncated to 256 tokens and decoded",
]


class MedHalluRow(BaseModel):
    """The source fields this benchmark reads; other columns are ignored."""

    model_config = ConfigDict(strict=True, extra="ignore")

    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


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
    *,
    dataset_path: str | None = None,
    revision: str | None = None,
) -> MemoryDataset:
    """Validate every row, then order samples exactly as the legacy shuffle did.

    Sample IDs are stable zero-based source row positions. The returned dataset
    holds the full split in legacy order (``common.dataset.legacy_order``: the
    legacy engine's seed-42 shuffle before its ``max_samples`` cap), so
    Inspect's ``limit`` (for example ``eval_config.limit`` in ``parity.yaml``)
    selects the legacy samples. ``revision`` pins the Hub dataset revision.
    """
    source = load_hf_dataset(
        DATASET_ID,
        dataset_path=dataset_path,
        revision=revision,
    )
    rows = [dict(source[i]) for i in range(len(source))]
    for index, row in enumerate(rows):
        try:
            MedHalluRow.model_validate(row)
        except ValidationError as exc:
            raise ValueError(
                f"Med-Hallu row {index} requires nonempty question and answer strings"
            ) from exc
    records = [
        {**rows[index], "source_index": index} for index in legacy_order(len(rows))
    ]
    return to_inspect_dataset(records, record_to_sample, dataset_id=DATASET_ID)
