"""XSTest source validation, legacy sample order and Inspect conversion.

Sample IDs are source row positions (``hirundo-io/XSTest:<row>``). The whole
source is validated, then reordered with the legacy seed-42 shuffle
(``common.dataset.legacy_order``), so a sample limit (``--limit N`` or the
parity config's ``eval_config.limit``) selects the same rows the legacy engine
selected with ``max_samples=N``.
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

DATASET_ID = "hirundo-io/XSTest"
SAFE_LABEL = "safe"
UNSAFE_LABEL = "unsafe"


def normalize_label(raw_label: Any) -> str:
    """Map a source label to ``safe`` or ``unsafe``; anything else is an error."""
    label = str(raw_label).strip().lower()
    if label not in {SAFE_LABEL, UNSAFE_LABEL}:
        raise ValueError(
            "XSTest label must be "
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


def validate_records(source: Any) -> list[dict[str, Any]]:
    """Validate every source row; ``question`` is preferred over ``prompt``.

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
            raise ValueError(f"XSTest row {index} is invalid: {error}") from error
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
    """Every XSTest row, validated, in the legacy shuffled order for ``seed``.

    Args:
        dataset_path: Local snapshot or other loading source for the dataset.
        revision: Hub dataset revision (branch, tag or commit).
        seed: Legacy ``Dataset.shuffle`` seed that orders the samples.

    Returns:
        One sample per source row.
    """
    records = validate_records(
        load_hf_dataset(
            DATASET_ID,
            dataset_path=dataset_path,
            revision=revision,
        )
    )
    records = [records[index] for index in legacy_order(len(records), seed)]
    return to_inspect_dataset(records, record_to_sample, dataset_id=DATASET_ID)
