"""CCPC-500-v2 dataset: the pinned StateCensor-Bench release file, verified by SHA-256.

The questions are the public StateCensor-Bench release on the Hugging Face
Hub. The task downloads one JSONL file at a pinned revision, checks the
SHA-256 of its raw bytes before parsing anything, then validates every row.
Rows keep their published order: there is no shuffle, so ``--limit N`` takes
the first N rows, as the legacy engine did. The whole file is verified before
Inspect applies any limit.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from inspect_ai.dataset import Dataset, MemoryDataset, Sample

DATASET_REPOSITORY = "hirundo-io/StateCensor-Bench"
"""The Hub dataset repository (public)."""
DATASET_REVISION = "05684cbddef75e2e78b73c125f666d44fb06ed02"
"""The pinned commit, tracked in ``.github/upstream-revisions.env``.

When it moves, update ``DATASET_SHA256`` by hand: syncing the manifest moves
only the revision.
"""
DATASET_FILE = "data/test-00000-of-00001.jsonl"
"""The release file in the repository: one JSON object per line."""
DATASET_SHA256 = "748ee10e9d94e033d6490963708c564f3145df5768279068cbd29054df557e90"
"""SHA-256 of ``DATASET_FILE`` at ``DATASET_REVISION``, checked on the raw bytes."""
EXPECTED_ROWS = 500
COLUMNS = frozenset({"id", "question", "request_form", "topic"})
"""The release's columns, all non-empty strings."""


def record_to_sample(record: dict[str, Any]) -> Sample:
    """One row as a sample: ID ``id``, input ``question``.

    ``topic`` and ``request_form`` (the taxonomy cell) and ``id`` are kept as
    sample metadata.

    Raises:
        ValueError: The row does not have exactly ``COLUMNS``, or one of them
            is not a non-empty string. The message names columns only, never
            a row value.
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
        id=record["id"],
        input=record["question"],
        metadata={key: value for key, value in record.items() if key != "question"},
    )


def validate(samples: list[Sample]) -> None:
    """Check the whole source: row count and unique IDs.

    Raises:
        ValueError: Either check fails.
    """
    if len(samples) != EXPECTED_ROWS:
        raise ValueError(
            f"CCPC-500-v2 must contain exactly {EXPECTED_ROWS} rows; "
            f"found {len(samples)}"
        )
    if len({sample.id for sample in samples}) != len(samples):
        raise ValueError("CCPC-500-v2 id values must be unique")


def verified_content(path: Path, source: str) -> bytes:
    """The file's bytes, once their SHA-256 matches ``DATASET_SHA256``.

    Args:
        path: The downloaded or local copy of ``DATASET_FILE``.
        source: Where the file came from, for the error message.

    Raises:
        ValueError: The SHA-256 differs; the message names the pinned file
            and both hashes.
    """
    content = path.read_bytes()
    found = hashlib.sha256(content).hexdigest()
    if found != DATASET_SHA256:
        raise ValueError(
            f"{source} has SHA-256 {found}, not the pinned {DATASET_SHA256} of "
            f"{DATASET_REPOSITORY}/{DATASET_FILE} at revision {DATASET_REVISION}"
        )
    return content


def parse_rows(content: bytes) -> list[dict[str, Any]]:
    """The JSONL rows, in file order.

    Raises:
        ValueError: A line is not a JSON object; the message gives its line
            number only.
    """
    rows = []
    for number, line in enumerate(content.decode("utf-8").splitlines(), start=1):
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"CCPC-500-v2 line {number} is not a JSON object")
        rows.append(row)
    return rows


def dataset(dataset_path: str | None = None) -> Dataset:
    """The 500 CCPC-500-v2 samples in published order, fully verified.

    Args:
        dataset_path: Read a local copy instead of downloading: either the
            release file itself or a folder that holds it at ``DATASET_FILE``
            (a snapshot of the dataset repository). The file must have the
            same SHA-256. A ``save_to_disk`` folder is not accepted, because
            its bytes cannot be checked against the pinned file's.

    Returns:
        Samples with ID ``id``, input ``question`` and the other columns as
        metadata.

    Raises:
        ValueError: A blank ``dataset_path`` or one that holds no file, a
            SHA-256 that is not ``DATASET_SHA256``, or an invalid row, row
            count or ID.
    """
    if dataset_path is None:
        path = Path(
            hf_hub_download(
                DATASET_REPOSITORY,
                DATASET_FILE,
                repo_type="dataset",
                revision=DATASET_REVISION,
            )
        )
        source = f"{DATASET_REPOSITORY}/{DATASET_FILE} at revision {DATASET_REVISION}"
        location = f"{DATASET_REPOSITORY}@{DATASET_REVISION}/{DATASET_FILE}"
    else:
        if not dataset_path.strip():
            raise ValueError("dataset_path is blank; omit it to download the file")
        path = Path(dataset_path)
        if path.is_dir():
            # A downloaded snapshot of the dataset repository (for example
            # ``hf download hirundo-io/StateCensor-Bench --repo-type dataset``)
            # holds the file at its repository path.
            path = path / DATASET_FILE
        if not path.is_file():
            raise ValueError(
                f"dataset_path {dataset_path} is neither a local copy of "
                f"{DATASET_FILE} nor a folder that holds it at that path"
            )
        source = f"dataset_path {dataset_path}"
        location = dataset_path
    rows = parse_rows(verified_content(path, source))
    samples = [record_to_sample(row) for row in rows]
    validate(samples)
    return MemoryDataset(samples, name=DATASET_REPOSITORY, location=location)
