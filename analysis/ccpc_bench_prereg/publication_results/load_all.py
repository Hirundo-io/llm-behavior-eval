"""Resolve a manifest into every loaded raw cell plus the shared dataset index."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .contract import RAW_DATASET_ID_FOR
from .dataset_join import resolve_shared_ccpc_dataset
from .raw_loader import load_ccpc_cell, load_refusal_cells

if TYPE_CHECKING:
    from .dataset_join import CcpcDatasetRow
    from .manifest_schema import Manifest
    from .raw_loader import CcpcCellData, RefusalCellData


@dataclass(frozen=True)
class LoadedPublicationRun:
    manifest: Manifest
    ccpc_cells: dict[str, CcpcCellData]
    xstest_cells: dict[str, dict[str, RefusalCellData]]
    orbench_cells: dict[str, dict[str, RefusalCellData]]
    dataset_index: dict[str, CcpcDatasetRow]

    def refusal_cell(self, model_key: str, benchmark: str) -> RefusalCellData:
        if benchmark == "xstest_safe":
            return self.xstest_cells[model_key]["safe"]
        if benchmark == "xstest_unsafe":
            return self.xstest_cells[model_key]["unsafe"]
        if benchmark == "orbench_hard":
            return self.orbench_cells[model_key]["safe"]
        if benchmark == "orbench_toxic":
            return self.orbench_cells[model_key]["unsafe"]
        raise ValueError(f"not a refusal benchmark: {benchmark!r}")


def load_publication_run(manifest: Manifest) -> LoadedPublicationRun:
    """Load every manifest entry's raw cells and resolve the shared dataset.

    Raises:
        PublicationLoaderError / DatasetJoinError: on any identity, duplicate,
            unknown-status, cross-check, or dataset-consistency failure.
    """
    ccpc_cells: dict[str, CcpcCellData] = {}
    xstest_cells: dict[str, dict[str, RefusalCellData]] = {}
    orbench_cells: dict[str, dict[str, RefusalCellData]] = {}

    for entry in manifest.entries:
        ccpc_cells[entry.model_key] = load_ccpc_cell(entry)
        xstest_cells[entry.model_key] = load_refusal_cells(
            entry, "xstest", RAW_DATASET_ID_FOR["xstest"]
        )
        orbench_cells[entry.model_key] = load_refusal_cells(
            entry, "orbench", RAW_DATASET_ID_FOR["orbench"]
        )

    dataset_index = resolve_shared_ccpc_dataset(ccpc_cells)

    return LoadedPublicationRun(
        manifest=manifest,
        ccpc_cells=ccpc_cells,
        xstest_cells=xstest_cells,
        orbench_cells=orbench_cells,
        dataset_index=dataset_index,
    )
