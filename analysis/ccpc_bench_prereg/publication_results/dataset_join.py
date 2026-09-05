"""Frozen CCPC-500 dataset join: topic/request_form per benchmark_id.

``responses.json`` carries only ``benchmark_id`` (plus the model's answer and
verdict) -- topic and request_form live in the frozen dataset file itself.
This module loads that file once, keyed by ``benchmark_id``, and is the single
place that verifies every checkpoint's CCPC run actually points at the same
frozen dataset (same path, same sha256) before any stratified summary is built.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .raw_loader import CcpcCellData


class DatasetJoinError(ValueError):
    """Raised when the frozen CCPC-500 dataset cannot be resolved or verified."""


@dataclass(frozen=True)
class CcpcDatasetRow:
    benchmark_id: str
    topic: str
    request_form: str


def load_ccpc_dataset_index(
    dataset_path: str, expected_sha256: str, expected_row_count: int
) -> dict[str, CcpcDatasetRow]:
    """Load the frozen CCPC-500 JSONL file and index it by benchmark_id.

    Args:
        dataset_path: Path to the frozen ``ccpc500.jsonl`` file, as recorded
            in every checkpoint's ``ccpc_benchmark.dataset_path``.
        expected_sha256: The sha256 every loaded CCPC run declared for this
            file; verified again here against the file's actual bytes.
        expected_row_count: The frozen row count (500).

    Returns:
        ``benchmark_id -> CcpcDatasetRow``.

    Raises:
        DatasetJoinError: If the file is missing, its hash does not match, its
            row count is wrong, or any ``benchmark_id`` is missing/duplicated.
    """
    path = Path(dataset_path)
    if not path.is_file():
        raise DatasetJoinError(f"frozen CCPC-500 dataset not found: {path}")
    raw_bytes = path.read_bytes()
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise DatasetJoinError(
            f"{path}: sha256 mismatch: runs declare {expected_sha256!r}, "
            f"file is actually {actual_sha256!r}"
        )
    rows = [
        json.loads(line)
        for line in raw_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != expected_row_count:
        raise DatasetJoinError(
            f"{path}: expected {expected_row_count} rows, found {len(rows)}"
        )
    index: dict[str, CcpcDatasetRow] = {}
    for row in rows:
        benchmark_id = str(row["benchmark_id"])
        if benchmark_id in index:
            raise DatasetJoinError(f"{path}: duplicate benchmark_id {benchmark_id!r}")
        index[benchmark_id] = CcpcDatasetRow(
            benchmark_id=benchmark_id,
            topic=str(row["topic"]),
            request_form=str(row["request_form"]),
        )
    if len(index) != expected_row_count:
        raise DatasetJoinError(
            f"{path}: expected {expected_row_count} unique benchmark_id values, "
            f"found {len(index)}"
        )
    return index


def resolve_shared_ccpc_dataset(
    ccpc_cells: dict[str, CcpcCellData],
) -> dict[str, CcpcDatasetRow]:
    """Verify every non-missing CCPC cell points at one identical frozen dataset.

    Args:
        ccpc_cells: ``model_key -> CcpcCellData``, as loaded by
            ``raw_loader.load_ccpc_cell``.

    Returns:
        The shared dataset index (``benchmark_id -> CcpcDatasetRow``).

    Raises:
        DatasetJoinError: If there is no non-missing cell to resolve a dataset
            from, or if two checkpoints disagree on ``dataset_path``/
            ``dataset_sha256`` (a non-deterministic/mixed-cohort configuration).
    """
    present = {
        model_key: cell for model_key, cell in ccpc_cells.items() if not cell.missing
    }
    if not present:
        raise DatasetJoinError(
            "no non-missing CCPC cell available to resolve the frozen dataset from"
        )
    reference_key, reference_cell = next(iter(present.items()))
    for model_key, cell in present.items():
        if (cell.dataset_path, cell.dataset_sha256) != (
            reference_cell.dataset_path,
            reference_cell.dataset_sha256,
        ):
            raise DatasetJoinError(
                "CCPC runs disagree on frozen dataset identity: "
                f"{reference_key!r} -> "
                f"(path={reference_cell.dataset_path!r}, sha256={reference_cell.dataset_sha256!r}) "
                f"vs {model_key!r} -> (path={cell.dataset_path!r}, sha256={cell.dataset_sha256!r})"
            )
    if (
        reference_cell.dataset_path is None
        or reference_cell.dataset_sha256 is None
        or reference_cell.expected_rows is None
    ):
        raise DatasetJoinError(
            f"{reference_key!r}: non-missing CCPC cell is missing dataset identity "
            "fields (dataset_path/dataset_sha256/expected_rows)"
        )
    return load_ccpc_dataset_index(
        reference_cell.dataset_path,
        reference_cell.dataset_sha256,
        reference_cell.expected_rows,
    )
