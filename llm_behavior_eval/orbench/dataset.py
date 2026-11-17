"""OR-Bench source validation, legacy label mapping and seeded sample order.

The legacy engine normalized the whole source, shuffled it once with
``datasets.Dataset.shuffle(seed=42)`` and then kept the first ``max_samples``
rows. Conversion here keeps every row in that same shuffled order
(``common.dataset.legacy_order``), so Inspect's
``limit`` (``eval_config.limit`` in ``parity.yaml``) selects the legacy samples.
Sample IDs are the original source row positions, so they survive shuffling.
"""

from typing import Annotated, Any

from inspect_ai.dataset import MemoryDataset, Sample
from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
)

from llm_behavior_eval.common.dataset import (
    LEGACY_SHUFFLE_SEED,
    legacy_order,
    load_hf_dataset,
    to_inspect_dataset,
)

DATASET_ID = "hirundo-io/or-bench"
SAFE_LABEL = "safe"
UNSAFE_LABEL = "unsafe"


def normalize_label(raw_label: Any) -> str:
    """Map a source label to ``safe``/``unsafe``, ignoring case and whitespace."""
    label = str(raw_label).strip().lower()
    if label not in {SAFE_LABEL, UNSAFE_LABEL}:
        raise ValueError(
            "OR-Bench label must be "
            f"{SAFE_LABEL!r} or {UNSAFE_LABEL!r}, got {raw_label!r}"
        )
    return label


class SourceRow(BaseModel):
    """One source row: its question (``question``, else ``prompt``) and label.

    Values are converted with ``str`` as the legacy engine did; other columns
    are ignored.
    """

    model_config = ConfigDict(frozen=True)

    question: Annotated[str, BeforeValidator(str)] = Field(
        validation_alias=AliasChoices("question", "prompt")
    )
    label: Annotated[str, BeforeValidator(normalize_label)]


def source_records(source: Any) -> list[dict[str, Any]]:
    """Validate every source row; ``question`` takes precedence over ``prompt``.

    Args:
        source: Iterable of row mappings.

    Returns:
        One record per row with its ID, question and normalized label.

    Raises:
        ValueError: A row has no ``question`` or ``prompt``, no ``label``, or
            a label other than ``safe``/``unsafe``.
    """
    records = []
    for index, raw in enumerate(source):
        try:
            row = SourceRow.model_validate(dict(raw))
        except ValidationError as error:
            raise ValueError(f"OR-Bench row {index} is invalid: {error}") from error
        records.append(
            {
                "id": f"{DATASET_ID}:{index}",
                "question": row.question,
                "label": row.label,
            }
        )
    return records


def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        id=record["id"],
        input=record["question"],
        metadata={"label": record["label"], "dataset_id": DATASET_ID},
    )


def dataset(
    *,
    dataset_path: str | None = None,
    revision: str | None = None,
    seed: int = LEGACY_SHUFFLE_SEED,
) -> MemoryDataset:
    """Every OR-Bench row, validated, in legacy shuffled order for ``seed``.

    Args:
        dataset_path: Local snapshot or other loading source for the dataset.
        revision: Hub dataset revision (branch, tag or commit).
        seed: Legacy ``Dataset.shuffle`` seed that orders the samples.

    Returns:
        One sample per source row.
    """
    source = load_hf_dataset(
        DATASET_ID,
        dataset_path=dataset_path,
        revision=revision,
    )
    records = source_records(source)
    ordered = [records[index] for index in legacy_order(len(records), seed)]
    return to_inspect_dataset(ordered, record_to_sample, dataset_id=DATASET_ID)
