"""Build ``publication_analysis_manifest.json``: the run's provenance record."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .aggregate import all_cell_summaries
from .contract import (
    ANALYSIS_CONTRACT_ID,
    ANALYSIS_CONTRACT_VERSION,
    PRIMARY_BENCHMARK_CELLS,
)

if TYPE_CHECKING:
    from .load_all import LoadedPublicationRun


def build_publication_analysis_manifest(
    loaded: LoadedPublicationRun, generated_at_utc: datetime | None = None
) -> dict[str, Any]:
    """Assemble the manifest content: contract identity, dataset identity,
    model roster, source artifact paths, generation timestamp, and per-cell
    validity outcome."""
    if generated_at_utc is None:
        generated_at_utc = datetime.now(UTC)

    reference_cell = next(
        cell for cell in loaded.ccpc_cells.values() if not cell.missing
    )

    models = [
        {
            "model_key": entry.model_key,
            "lineage": entry.lineage,
            "checkpoint_label": entry.checkpoint_label,
            "model_repo_id": entry.model_repo_id,
        }
        for entry in loaded.manifest.entries
    ]

    source_artifact_paths: dict[str, dict[str, str]] = {}
    validity_outcome: dict[str, dict[str, bool]] = {}
    for summary in all_cell_summaries(loaded):
        source_artifact_paths.setdefault(summary.model_key, {})[summary.benchmark] = (
            summary.source_result_path
        )
        validity_outcome.setdefault(summary.model_key, {})[summary.benchmark] = (
            summary.valid
        )

    all_primary_valid_model_keys = [
        entry.model_key
        for entry in loaded.manifest.entries
        if all(
            validity_outcome[entry.model_key][benchmark]
            for benchmark in PRIMARY_BENCHMARK_CELLS
        )
    ]

    return {
        "analysis_contract": {
            "id": ANALYSIS_CONTRACT_ID,
            "version": ANALYSIS_CONTRACT_VERSION,
        },
        "run_id": loaded.manifest.run_id,
        "dataset_identity": {
            "path": reference_cell.dataset_path,
            "sha256": reference_cell.dataset_sha256,
            "expected_rows": reference_cell.expected_rows,
        },
        "models": models,
        "source_artifact_paths": source_artifact_paths,
        "generated_at_utc": generated_at_utc.isoformat(),
        "validity_outcome": {
            "per_model_benchmark_valid": validity_outcome,
            "all_primary_valid_model_keys": all_primary_valid_model_keys,
            "n_models": len(models),
            "n_all_primary_valid": len(all_primary_valid_model_keys),
        },
    }
