"""Execution-bound authoritative QA provenance for CCPC500 training-data artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

QAStatus = Literal["AUTHORITATIVE", "HISTORICAL_STALE"]

REQUIRED_BINDING_FIELDS = (
    "execution_id",
    "raw_snapshot_sha256",
    "accepted_valid_id_hash_sha256",
    "generator_config_sha256",
    "detector_bundle_sha256",
    "qa_artifact_path",
    "status",
)


def id_set_hash(pool_row_ids: list[int] | set[int]) -> str:
    """Hash a pool-row ID set using the project canonical format.

    Args:
        pool_row_ids: Iterable of integer pool row IDs.

    Returns:
        SHA-256 hex digest of the sorted newline-delimited ID list.
    """
    payload = "\n".join(str(pool_row_id) for pool_row_id in sorted(pool_row_ids)) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def resolve_qa_status(
    bindings: dict[str, Any],
    active_bindings: dict[str, Any],
) -> QAStatus:
    """Resolve whether QA bindings match the active execution.

    Args:
        bindings: Candidate QA report bindings.
        active_bindings: Active execution bindings.

    Returns:
        ``AUTHORITATIVE`` when all binding fields match, else ``HISTORICAL_STALE``.
    """
    for field in REQUIRED_BINDING_FIELDS:
        if field == "status":
            continue
        if bindings.get(field) != active_bindings.get(field):
            return "HISTORICAL_STALE"
    return "AUTHORITATIVE"


def assert_authoritative_for_training(bindings: dict[str, Any]) -> None:
    """Fail closed when QA bindings are not authoritative.

    Args:
        bindings: Candidate QA report bindings including a ``status`` field.

    Raises:
        RuntimeError: If bindings are missing, stale, or incomplete.
    """
    missing = [field for field in REQUIRED_BINDING_FIELDS if field not in bindings]
    if missing:
        raise RuntimeError(f"QA bindings missing required fields: {missing}")
    if bindings["status"] != "AUTHORITATIVE":
        raise RuntimeError(
            "QA report is HISTORICAL_STALE and cannot gate training: "
            f"{bindings.get('qa_artifact_path')}"
        )


def load_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON mapping.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def build_active_bindings(
    *,
    execution_id: str,
    raw_snapshot_sha256: str,
    accepted_valid_id_hash_sha256: str,
    generator_config_sha256: str,
    detector_bundle_sha256: str,
    qa_artifact_path: str,
) -> dict[str, Any]:
    """Construct authoritative QA bindings for the active execution.

    Args:
        execution_id: Active execution identifier.
        raw_snapshot_sha256: Final raw snapshot digest.
        accepted_valid_id_hash_sha256: Accepted valid ID-set digest.
        generator_config_sha256: Frozen generator config digest.
        detector_bundle_sha256: Frozen detector bundle digest.
        qa_artifact_path: Path to the authoritative QA artifact.

    Returns:
        Binding record with ``status=AUTHORITATIVE``.
    """
    return {
        "execution_id": execution_id,
        "raw_snapshot_sha256": raw_snapshot_sha256,
        "accepted_valid_id_hash_sha256": accepted_valid_id_hash_sha256,
        "generator_config_sha256": generator_config_sha256,
        "detector_bundle_sha256": detector_bundle_sha256,
        "qa_artifact_path": qa_artifact_path,
        "status": "AUTHORITATIVE",
    }


def write_authoritative_qa_index(path: Path, entries: list[dict[str, Any]]) -> None:
    """Write the execution-bound authoritative QA manifest/index.

    Args:
        path: Destination JSON path.
        entries: QA index entries with binding metadata.
    """
    payload = {
        "schema": "ccpc500_authoritative_qa_index_v1",
        "entries": entries,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
