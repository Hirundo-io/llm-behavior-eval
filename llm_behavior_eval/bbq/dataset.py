"""BBQ split identity, full-source validation and legacy-ordered conversion."""

from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample
from pydantic import BaseModel, ConfigDict, ValidationError

from llm_behavior_eval.common.dataset import (
    LEGACY_SHUFFLE_SEED,
    legacy_order,
    load_hf_dataset,
    to_inspect_dataset,
)

BIAS_TYPES = ("gender", "race", "nationality", "physical", "age", "religion")
KINDS = ("bias", "unbias")
_SOURCE_INDEX = "__bbq_source_index"


class BbqRow(BaseModel):
    """The text columns every row of a split must have."""

    model_config = ConfigDict(strict=True, extra="ignore")

    question: str
    answer: str


class StereotypedBbqRow(BbqRow):
    """A row of a split that has the ``stereotyped_answer`` column.

    When the column exists, every row needs a string value (the legacy engine
    read a null as an empty string; no published split has one).
    """

    stereotyped_answer: str


def dataset_id(bias_type: str, kind: str) -> str:
    if bias_type not in BIAS_TYPES or kind not in KINDS:
        raise ValueError(
            f"Expected bias_type in {BIAS_TYPES} and kind in {KINDS}; "
            f"got {bias_type!r}, {kind!r}"
        )
    return f"hirundo-io/bbq-{bias_type}-{kind}-free-text"


def dataset(
    bias_type: str,
    kind: str,
    *,
    dataset_path: str | None = None,
    revision: str | None = None,
    seed: int = LEGACY_SHUFFLE_SEED,
) -> MemoryDataset:
    """Every row of one split, validated, in the legacy engine's shuffled order.

    The legacy engine shuffled with ``seed`` and then kept the first
    ``max_samples`` rows, so Inspect's ``--limit N`` (or ``eval_config.limit``
    in ``parity.yaml``) selects the same rows (``common.dataset.legacy_order``).
    Sample IDs keep the zero-based source row index. ``revision`` pins the HF
    Hub dataset revision.
    """
    identity = dataset_id(bias_type, kind)
    records = load_hf_dataset(
        identity,
        dataset_path=dataset_path,
        revision=revision,
    )
    has_stereotype = "stereotyped_answer" in records.column_names
    row_model = StereotypedBbqRow if has_stereotype else BbqRow
    for index, row in enumerate(records):
        try:
            row_model.model_validate(row)
        except ValidationError as error:
            raise ValueError(
                f"{identity}: row {index}: required text columns: "
                f"{list(row_model.model_fields)}"
            ) from error
    shuffled = records.add_column(_SOURCE_INDEX, list(range(len(records)))).select(
        legacy_order(len(records), seed)
    )

    def record_to_sample(row: dict[str, Any]) -> Sample:
        return Sample(
            input=row["question"],
            target=row["answer"],
            id=f"{identity}:{row[_SOURCE_INDEX]}",
            metadata={
                "dataset_id": identity,
                "bias_type": bias_type,
                "kind": kind,
                "has_stereotype": has_stereotype,
                "stereotyped_answer": row.get("stereotyped_answer"),
            },
        )

    return to_inspect_dataset(shuffled, record_to_sample, dataset_id=identity)
