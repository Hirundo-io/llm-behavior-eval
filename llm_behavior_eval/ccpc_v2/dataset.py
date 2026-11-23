"""CCPC-500-v2 dataset: pinned source, local sources and full-source validation.

The Hub dataset is private (``hirundo-io/ccpc-500-v2``); load it with a token
that has access (``LLM_BEHAVIOR_EVAL_DATASET_TOKEN`` or ``HF_TOKEN``). Rows
keep their published order: there is no shuffle, so ``--limit N`` takes the
first N rows, as the legacy engine did. Every source, Hub or local, is
validated in full before Inspect applies any limit.
"""

import hashlib
import json
from typing import Any

from inspect_ai.dataset import Dataset, Sample

from llm_behavior_eval.common.dataset import load_dataset

DATASET_REPOSITORY = "hirundo-io/ccpc-500-v2"
DATASET_CONFIG = "default"
DATASET_SPLIT = "test"
DATASET_REVISION = "a00eee8cf918ee0d00197a9ca1aebb097756c849"
"""Owner-frozen R8 release (RSCH-250)."""
EXPECTED_ROWS = 500
COLUMNS = frozenset({"benchmark_id", "question", "request_form", "topic"})
"""The release's columns, all non-empty strings."""
DATASET_FINGERPRINT: str | None = (
    "a4de18b4fc728e45c2f8aec01d025255a3892530fb15835b370b31fc537705bc"
)
"""SHA-256 of ``(question, benchmark_id)`` for every row in published order.

Confirmed against the private release at ``DATASET_REVISION``. ``None`` would
make loading fail rather than skip the check.
"""


def fingerprint(samples: list[Sample]) -> str:
    """SHA-256 of every row's question and benchmark ID, in order.

    The same serialization as the legacy CCPC-500 fingerprint: a JSON list of
    ``{"question", "benchmark_id"}`` objects, non-ASCII kept, compact.
    """
    rows = [{"question": s.input, "benchmark_id": s.id} for s in samples]
    serialized = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def record_to_sample(record: dict[str, Any]) -> Sample:
    """One row as a sample: ID ``benchmark_id``, input ``question``.

    ``topic`` and ``request_form`` (the taxonomy cell) and ``benchmark_id``
    are kept as sample metadata.

    Raises:
        ValueError: The row does not have exactly ``COLUMNS``, or one of them
            is not a non-empty string.
    """
    if set(record) != COLUMNS:
        raise ValueError(
            f"CCPC-500-v2 columns must be exactly {sorted(COLUMNS)}; "
            f"found {sorted(record)}"
        )
    for column in sorted(COLUMNS):
        value = record.get(column)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"CCPC-500-v2 row field {column!r} must be a non-empty string"
            )
    return Sample(
        id=record["benchmark_id"],
        input=record["question"],
        metadata={key: value for key, value in record.items() if key != "question"},
    )


def validate(samples: list[Sample]) -> None:
    """Check the whole source: row count, unique IDs and the pinned fingerprint.

    Raises:
        ValueError: Any check fails, or the fingerprint is not pinned yet.
    """
    if len(samples) != EXPECTED_ROWS:
        raise ValueError(
            f"CCPC-500-v2 must contain exactly {EXPECTED_ROWS} rows; "
            f"found {len(samples)}"
        )
    if len({sample.id for sample in samples}) != len(samples):
        raise ValueError("CCPC-500-v2 benchmark_id values must be unique")
    if DATASET_FINGERPRINT is None:
        raise ValueError(
            "The CCPC-500-v2 row fingerprint is not pinned yet (LLM-175), so "
            "the dataset cannot be verified"
        )
    found = fingerprint(samples)
    if found != DATASET_FINGERPRINT:
        raise ValueError(
            "CCPC-500-v2 rows do not match the frozen release; expected "
            f"fingerprint {DATASET_FINGERPRINT}, found {found}"
        )


def dataset(dataset_path: str | None = None) -> Dataset:
    """The 500 CCPC-500-v2 samples in published order, fully validated.

    Args:
        dataset_path: Read this local source instead of the pinned Hub
            dataset: a folder ``datasets.load_dataset`` reads (for example a
            Hub snapshot) or a ``save_to_disk`` folder. It must hold the same
            rows as the frozen release.

    Returns:
        Samples with ID ``benchmark_id``, input ``question`` and the other
        columns as metadata.

    Raises:
        ValueError: A blank ``dataset_path``, or a row, the row count, the IDs
            or the fingerprint is invalid.
    """
    if dataset_path is not None and not dataset_path.strip():
        raise ValueError("dataset_path is blank; omit it to use the Hub dataset")
    # ``common.dataset.load_dataset`` reads the pinned Hub dataset (with the
    # dataset token), a local ``load_dataset`` folder or a ``save_to_disk``
    # folder; no seed, so rows keep their published order.
    if dataset_path is None:
        data = load_dataset(
            DATASET_REPOSITORY,
            split=DATASET_SPLIT,
            sample_fields=record_to_sample,
            revision=DATASET_REVISION,
            name=DATASET_CONFIG,
        )
    else:
        data = load_dataset(
            dataset_path, split=DATASET_SPLIT, sample_fields=record_to_sample
        )
    validate(list(data))
    return data
