"""UNQOVER split identity, full-source validation and legacy sample order."""

from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample
from pydantic import BaseModel, ConfigDict, ValidationError

from llm_behavior_eval.common.dataset import (
    LEGACY_SHUFFLE_SEED,
    legacy_order,
    load_hf_dataset,
    to_inspect_dataset,
)

BIAS_TYPES = ("religion", "gender", "race", "nationality")
KIND = "bias"
SOURCE_INDEX = "__unqover_source_index"


class UnqoverRow(BaseModel):
    """The text columns every row of a split must have."""

    model_config = ConfigDict(strict=True, extra="ignore")

    question: str
    answer: str


class StereotypedUnqoverRow(UnqoverRow):
    """A row of a split that has the ``stereotyped_answer`` column.

    When the column exists, every row needs a string value (the legacy engine
    read a null as an empty string; no published split has one).
    """

    stereotyped_answer: str


def dataset_id(bias_type: str) -> str:
    """Logical identity, matching the legacy ``unqover:bias:<type>`` preset."""
    return f"hirundo-io/unqover-{bias_type}-{KIND}-free-text"


def validate_split(bias_type: str, kind: str) -> None:
    """Reject unsupported selections before any data is loaded."""
    if kind != KIND:
        raise ValueError(
            f"UNQOVER supports only kind=bias; no {kind!r} split exists"
            " (only the ambiguous/bias split was published)"
        )
    if bias_type not in BIAS_TYPES:
        raise ValueError(f"Expected a UNQOVER bias_type in {BIAS_TYPES}")


def dataset(
    bias_type: str,
    kind: str = KIND,
    *,
    dataset_path: str | None = None,
    revision: str | None = None,
    seed: int = LEGACY_SHUFFLE_SEED,
) -> MemoryDataset:
    """Every row of one split, in the legacy engine's shuffled order.

    The legacy engine shuffled the split with ``seed`` (42) and then kept the
    first ``max_samples`` (500). Conversion reproduces the shuffle
    (``common.dataset.legacy_order``), so Inspect's ``--limit 500``
    (``eval_config.limit`` in ``parity.yaml``) selects the same samples. Sample
    IDs carry the original zero-based row index. ``revision`` pins the HF Hub
    dataset revision.
    """
    validate_split(bias_type, kind)
    identity = dataset_id(bias_type)
    records = load_hf_dataset(
        identity,
        dataset_path=dataset_path,
        revision=revision,
    )
    has_stereotype = "stereotyped_answer" in records.column_names
    row_model = StereotypedUnqoverRow if has_stereotype else UnqoverRow
    for index, row in enumerate(records):
        try:
            row_model.model_validate(row)
        except ValidationError as error:
            raise ValueError(
                f"{identity}: row {index}: required text columns: "
                f"{list(row_model.model_fields)}"
            ) from error
    indexed = records.add_column(SOURCE_INDEX, list(range(len(records)))).select(
        legacy_order(len(records), seed)
    )

    def record_to_sample(row: dict[str, Any]) -> Sample:
        return Sample(
            input=row["question"],
            target=row["answer"],
            id=f"{identity}:{row[SOURCE_INDEX]}",
            metadata={
                "dataset_id": identity,
                "bias_type": bias_type,
                "kind": kind,
                "has_stereotype": has_stereotype,
                "stereotyped_answer": row.get("stereotyped_answer"),
            },
        )

    return to_inspect_dataset(indexed, record_to_sample, dataset_id=identity)
