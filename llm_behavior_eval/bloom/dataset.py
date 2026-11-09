"""BLOOM split identity, full-source validation and legacy-ordered conversion."""

from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample
from pydantic import BaseModel, ConfigDict, ValidationError

from llm_behavior_eval.common.dataset import (
    LEGACY_SHUFFLE_SEED,
    legacy_order,
    load_hf_dataset,
    to_inspect_dataset,
)

BIAS_TYPES = ("age", "gender", "race")
KINDS = ("bias", "unbias", "ambiguous")
SOURCE_INDEX = "__bloom_source_index"


class BloomRow(BaseModel):
    """The text columns every row of a split must have."""

    model_config = ConfigDict(strict=True, extra="ignore")

    question: str
    answer: str


class StereotypedBloomRow(BloomRow):
    """A row of a split that has the ``stereotyped_answer`` column.

    When the column exists, every row needs a string value (the legacy engine
    read a null as an empty string; no published split has one).
    """

    stereotyped_answer: str


def dataset_id(bias_type: str, kind: str) -> str:
    """Logical HF identity of one BLOOM split, as built by the legacy presets."""
    if bias_type not in BIAS_TYPES or kind not in KINDS:
        raise ValueError(
            "Expected a BLOOM bias type (age, gender, race) and kind "
            "bias, unbias or ambiguous"
        )
    split = "ambiguous-bias" if kind == "ambiguous" else kind
    return f"hirundo-io/bloom-{bias_type}-{split}-free-text"


def dataset(
    bias_type: str,
    kind: str,
    *,
    dataset_path: str | None = None,
    revision: str | None = None,
    seed: int = LEGACY_SHUFFLE_SEED,
) -> MemoryDataset:
    """Every row of the split, in the legacy engine's shuffled order.

    The legacy engine shuffled the whole split with ``seed`` and then kept the
    first ``max_samples`` rows. Applying the same order here
    (``common.dataset.legacy_order``), and keeping every row, lets Inspect's
    ``--limit`` (or ``eval_config.limit`` in ``parity.yaml``) select exactly the
    legacy samples. Sample IDs carry the original source row index, so they do
    not depend on the order or the limit. ``revision`` pins the HF Hub dataset
    revision.
    """
    identity = dataset_id(bias_type, kind)
    records = load_hf_dataset(
        identity,
        dataset_path=dataset_path,
        revision=revision,
    )
    has_stereotype = "stereotyped_answer" in records.column_names
    row_model = StereotypedBloomRow if has_stereotype else BloomRow
    for index, row in enumerate(records):
        try:
            row_model.model_validate(row)
        except ValidationError as error:
            raise ValueError(
                f"{identity}: row {index}: required text columns: "
                f"{list(row_model.model_fields)}"
            ) from error
    shuffled = records.add_column(SOURCE_INDEX, list(range(len(records)))).select(
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

    return to_inspect_dataset(shuffled, record_to_sample, dataset_id=identity)
