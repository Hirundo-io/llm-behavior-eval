"""Tensor Trust benchmarks: pinned source files, checksums and validation.

The authors publish the two robustness benchmarks as JSONL files in
``HumanCompatibleAI/tensor-trust-data``. There is no loadable Hugging Face
copy (the ``qxcv/tensor-trust`` mirror has no dataset config for the benchmark
files), so each file is fetched from GitHub at a pinned commit and its SHA-256
is checked before any row is read. A local copy (``dataset_path``) gets the
same check. Rows keep their published order, so ``--limit N`` takes the first
samples.

Each row becomes two samples: the attack, and the true access code (for
defense validity), in that order.
"""

import hashlib
import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from inspect_ai.dataset import MemoryDataset, Sample

from .prompts import defense_messages

DATA_REPOSITORY = "HumanCompatibleAI/tensor-trust-data"
DATA_COMMIT = "747a75e096761ebc01bd3970158827326b4add23"
"""``main`` of ``tensor-trust-data``; the benchmark files last changed in
``01856c36f7e6a70442378d7fe9d5e9de3329040a`` (2023-10-18)."""
RAW_URL = "https://raw.githubusercontent.com/{repository}/{commit}/{path}"

BenchmarkName = Literal["hijacking", "extraction"]
SampleKind = Literal["attack", "access_code"]
ATTACK: SampleKind = "attack"
ACCESS_CODE: SampleKind = "access_code"

COLUMNS = frozenset({"sample_id", "pre_prompt", "attack", "post_prompt", "access_code"})
"""Every row has exactly these: ``sample_id`` an integer, the rest strings."""
NON_EMPTY = ("attack", "access_code")
"""Columns that are never empty; ``pre_prompt`` and ``post_prompt`` can be."""

DOWNLOAD_ATTEMPTS = 4
DOWNLOAD_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class BenchmarkFile:
    """One published benchmark file and what it must contain."""

    name: BenchmarkName
    path: str
    """Path inside ``DATA_REPOSITORY``."""
    sha256: str
    """SHA-256 of the file's bytes at ``DATA_COMMIT``."""
    rows: int
    """JSON rows in the file (the paper's counts are one lower, see README)."""

    @property
    def url(self) -> str:
        """The file's raw URL at the pinned commit."""
        return RAW_URL.format(
            repository=DATA_REPOSITORY, commit=DATA_COMMIT, path=self.path
        )


BENCHMARKS: dict[BenchmarkName, BenchmarkFile] = {
    "hijacking": BenchmarkFile(
        name="hijacking",
        path="benchmarks/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl",
        sha256="2ad3c2343bbf43cb1924f3ec5d6f830864ce3c879e9832b01d93a991e7e2e680",
        rows=776,
    ),
    "extraction": BenchmarkFile(
        name="extraction",
        path="benchmarks/extraction-robustness/v1/extraction_robustness_dataset.jsonl",
        sha256="4090f9f57b6df69cc32d7669ac991e3fd53e6e2bb616a5e6f45dac79c183ee4c",
        rows=570,
    ),
}


def download(url: str) -> bytes:
    """Fetch ``url``, retrying network errors with exponential backoff.

    Raises:
        OSError: Every attempt failed.
    """
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            with urllib.request.urlopen(  # noqa: S310 (a fixed https URL)
                url, timeout=DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                return response.read()
        except OSError:
            if attempt == DOWNLOAD_ATTEMPTS - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def verify_checksum(data: bytes, benchmark: BenchmarkFile, source: str) -> None:
    """Check that ``data`` is the pinned file.

    Raises:
        ValueError: The SHA-256 differs from ``benchmark.sha256``.
    """
    found = hashlib.sha256(data).hexdigest()
    if found != benchmark.sha256:
        raise ValueError(
            f"Tensor Trust {benchmark.name} data from {source} does not match "
            f"{DATA_REPOSITORY}@{DATA_COMMIT}/{benchmark.path}; expected SHA-256 "
            f"{benchmark.sha256}, found {found}"
        )


def parse_rows(data: bytes) -> list[dict[str, Any]]:
    """JSON objects of a JSONL file; blank lines are skipped, as upstream does.

    Raises:
        ValueError: A line is not a JSON object.
    """
    rows = []
    for number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Tensor Trust line {number} is not a JSON object")
        rows.append(row)
    return rows


def validate_row(record: dict[str, Any]) -> None:
    """Check one row's columns and types.

    Raises:
        ValueError: The columns are not exactly ``COLUMNS``, ``sample_id`` is
            not an integer, another column is not a string, or ``attack`` or
            ``access_code`` is blank.
    """
    if set(record) != COLUMNS:
        raise ValueError(
            f"Tensor Trust columns must be exactly {sorted(COLUMNS)}; "
            f"found {sorted(record)}"
        )
    sample_id = record["sample_id"]
    if not isinstance(sample_id, int) or isinstance(sample_id, bool):
        raise ValueError(f"Tensor Trust sample_id must be an integer: {sample_id!r}")
    for column in sorted(COLUMNS - {"sample_id"}):
        if not isinstance(record[column], str):
            raise ValueError(f"Tensor Trust {column!r} must be a string")
    for column in NON_EMPTY:
        if not record[column].strip():
            raise ValueError(f"Tensor Trust {column!r} must not be blank")


def record_to_samples(record: dict[str, Any]) -> list[Sample]:
    """One row as its attack sample and its access-code sample.

    Both send the defense around the middle message (see
    ``prompts.defense_messages``). IDs are ``<sample_id>-attack`` and
    ``<sample_id>-access_code``; metadata holds the row's ``sample_id``, the
    sample ``kind`` and the ``access_code`` the scorer checks for.
    """
    validate_row(record)
    samples = []
    for kind in (ATTACK, ACCESS_CODE):
        samples.append(
            Sample(
                id=f"{record['sample_id']}-{kind}",
                input=defense_messages(
                    record["pre_prompt"], record[kind], record["post_prompt"]
                ),
                metadata={
                    "tensor_trust_sample_id": record["sample_id"],
                    "kind": kind,
                    "access_code": record["access_code"],
                },
            )
        )
    return samples


def validate(rows: list[dict[str, Any]], benchmark: BenchmarkFile) -> None:
    """Check the whole file: every row, the row count and unique IDs.

    Raises:
        ValueError: Any check fails.
    """
    for row in rows:
        validate_row(row)
    if len(rows) != benchmark.rows:
        raise ValueError(
            f"Tensor Trust {benchmark.name} must contain exactly {benchmark.rows} "
            f"rows; found {len(rows)}"
        )
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError(f"Tensor Trust {benchmark.name} sample_id values repeat")


def dataset(name: BenchmarkName, dataset_path: str | None = None) -> MemoryDataset:
    """One benchmark's samples in published order, checksum-verified.

    Args:
        name: ``hijacking`` or ``extraction``.
        dataset_path: Read this local copy of the published JSONL file instead
            of downloading it; it must have the same SHA-256.

    Returns:
        Two samples per row (attack, then access code), in file order.

    Raises:
        ValueError: The checksum, a row, the row count or the IDs are invalid.
    """
    benchmark = BENCHMARKS[name]
    if dataset_path is None:
        source = benchmark.url
        data = download(source)
    else:
        source = dataset_path
        data = Path(dataset_path).read_bytes()
    verify_checksum(data, benchmark, source)
    rows = parse_rows(data)
    validate(rows, benchmark)
    return MemoryDataset(
        [sample for row in rows for sample in record_to_samples(row)],
        name=f"tensor_trust_{name}",
        location=source,
    )
