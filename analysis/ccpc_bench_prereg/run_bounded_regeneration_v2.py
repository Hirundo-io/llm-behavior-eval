#!/usr/bin/env python3
"""Bounded CCPC500 regeneration v2 — teacher-contract v2 framing surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from analysis.ccpc_bench_prereg.build_ccpc500_training_data import (
    ACTIVE_GENERATOR_CONFIG_SHA256,
    ACTIVE_RAW_SNAPSHOT_SHA256,
    DETECTOR_CONTRACT,
    digest,
    read_json,
    read_jsonl,
    write_json,
)
from analysis.ccpc_bench_prereg.ccpc500_training_provenance import id_set_hash
from analysis.ccpc_bench_prereg.run_bounded_regeneration import (
    EXPECTED_FRAMING_HASH,
    EXPECTED_MODEL,
    EXPECTED_OVERFLOW_HASH,
    EXPECTED_PROVIDER_ACTION_HASH,
    CORRECTION,
    PROBE,
    SNAPSHOT,
    biased_membership_index,
    mechanical_row_qa,
    now,
    original_biased_winners,
    verify_correction_bundle,
)

EXEC_V2 = PROBE / "bounded_regeneration_v2"
TEACHER_V1 = SNAPSHOT / "supervision_generator_template.txt"
TEACHER_V2 = PROBE / "teacher_contract_v2/supervision_generator_template_v2.txt"
PARTIAL_V1_ACCEPTS = PROBE / "bounded_regeneration_v1/biased_regeneration/accepted_replacements.jsonl"
RETRY_CEILING = 5
CANARY_REPRODUCER_ID = 407
TEACHER_CONTRACT_V1 = "supervision_generator_template_v1"
TEACHER_CONTRACT_V2 = "supervision_generator_template_v2"


def teacher_template_digest(path: Path) -> str:
    """Return the SHA-256 digest of one teacher template file.

    Args:
        path: Teacher template path.

    Returns:
        Hex digest string.
    """
    return digest(path)


def original_framing_category(pool_row_id: int, winners: dict[int, dict[str, Any]]) -> str:
    """Classify the original exec-4 framing defect for one framing ID.

    Args:
        pool_row_id: Pool row identifier.
        winners: First parse-valid exec-4 attempts keyed by pool row ID.

    Returns:
        Sorted comma-joined framing category label.
    """
    if str(DETECTOR_CONTRACT) not in sys.path:
        sys.path.insert(0, str(DETECTOR_CONTRACT))
    from corrected_detectors import refusal_frame_findings

    answer = json.loads(winners[pool_row_id]["raw_structured_response"])["answer"]
    findings = refusal_frame_findings(answer)
    return ",".join(findings) if findings else "none"


def select_framing_canary_ids(
    framing_ids: list[int],
    membership: dict[int, dict[str, Any]],
    winners: dict[int, dict[str, Any]],
) -> list[int]:
    """Select a deterministic framing canary with coverage and the known reproducer.

    Selection is mechanical:
    - always include the frozen reproducer ID (407);
    - one minimum pool_row_id per request_form present in the framing set;
    - one minimum pool_row_id per original exec-4 framing-defect category.

    Args:
        framing_ids: Complete framing-set IDs.
        membership: Biased membership rows keyed by pool row ID.
        winners: First parse-valid exec-4 attempts keyed by pool row ID.

    Returns:
        Sorted unique canary ID list.
    """
    if CANARY_REPRODUCER_ID not in framing_ids:
        raise RuntimeError("canary reproducer is not in framing set")

    by_request_form: dict[str, list[int]] = defaultdict(list)
    by_category: dict[str, list[int]] = defaultdict(list)
    for pool_row_id in framing_ids:
        by_request_form[membership[pool_row_id]["request_form"]].append(pool_row_id)
        by_category[original_framing_category(pool_row_id, winners)].append(pool_row_id)

    selected = {CANARY_REPRODUCER_ID}
    selected.update(min(ids) for ids in by_request_form.values())
    selected.update(min(ids) for ids in by_category.values())
    return sorted(selected)


def archive_superseded_partial_v1_accepts(output_dir: Path) -> dict[str, Any]:
    """Preserve abandoned v1 partial framing accepts without admitting them to v2 corpus.

    Args:
        output_dir: Bounded regeneration v2 execution directory.

    Returns:
        Supersession manifest mapping.
    """
    if not PARTIAL_V1_ACCEPTS.is_file():
        raise RuntimeError("missing abandoned partial v1 accepts ledger")

    archive_dir = output_dir / "superseded_partial_v1_remediation"
    archive_dir.mkdir(parents=True, exist_ok=True)
    destination = archive_dir / "accepted_replacements.jsonl"
    shutil.copy2(PARTIAL_V1_ACCEPTS, destination)

    accepted_rows = read_jsonl(destination)
    accepted_ids = sorted(int(row["pool_row_id"]) for row in accepted_rows)
    manifest = {
        "artifact": "superseded_partial_v1_remediation",
        "status": "SUPERSEDED_BY_TEACHER_V2",
        "source_ledger": str(PARTIAL_V1_ACCEPTS),
        "archived_ledger": str(destination),
        "count": len(accepted_ids),
        "ids": accepted_ids,
        "sha256": id_set_hash(accepted_ids),
        "ledger_sha256": digest(destination),
        "admission_policy": "historical_outputs_preserved_only; must_not_enter_repaired_corpus",
    }
    write_json(archive_dir / "SUPERSESSION_MANIFEST.json", manifest)
    return manifest


def surface_hashes(bundle: dict[str, Any]) -> dict[str, Any]:
    """Compute frozen per-subset surface hashes for v2 execution.

    Args:
        bundle: Verified correction-bundle metadata.

    Returns:
        Surface hash mapping.
    """
    overflow = bundle["overflow"]
    framing_ids = bundle["framing"]["ids"]
    biased_overflow_ids = overflow["biased_ce"]
    normal_ids = sorted(overflow["normal_train"]) + sorted(overflow["normal_dev"])
    provider_action_ids = bundle["provider_action_ids"]
    return {
        "framing_set_sha256": EXPECTED_FRAMING_HASH,
        "biased_overflow_only_sha256": id_set_hash(biased_overflow_ids),
        "normal_overflow_sha256": id_set_hash(normal_ids),
        "provider_action_total_sha256": EXPECTED_PROVIDER_ACTION_HASH,
        "fresh_output_surface_sha256": id_set_hash(provider_action_ids),
        "counts": {
            "framing_v2": len(framing_ids),
            "biased_overflow_v1": len(biased_overflow_ids),
            "normal_base_qwen": len(normal_ids),
            "fresh_output_total": len(provider_action_ids),
            "unaffected_rows": 7115,
        },
    }


def freeze_protocol(output_dir: Path = EXEC_V2) -> dict[str, Any]:
    """Freeze teacher v2, regeneration surfaces, canary IDs, and execution manifest.

    Args:
        output_dir: Bounded regeneration v2 execution directory.

    Returns:
        Frozen execution manifest.
    """
    bundle = verify_correction_bundle()
    if not TEACHER_V2.is_file():
        raise RuntimeError("missing teacher contract v2 template")
    if digest(TEACHER_V2) == teacher_template_digest(TEACHER_V1):
        raise RuntimeError("teacher v2 template must differ from v1")

    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = output_dir / "frozen_inputs"
    frozen.mkdir(parents=True, exist_ok=True)

    for name in ("overflow_ids.json", "framing_ids.json", "provider_action_ids.json"):
        shutil.copy2(CORRECTION / name, frozen / name)

    shutil.copy2(TEACHER_V2, frozen / "supervision_generator_template_v2.txt")
    shutil.copy2(TEACHER_V1, frozen / "supervision_generator_template_v1.txt")

    membership = biased_membership_index()
    winners = original_biased_winners()
    framing_ids = bundle["framing"]["ids"]
    canary_ids = select_framing_canary_ids(framing_ids, membership, winners)
    canary_manifest = {
        "artifact": "framing_canary_v2",
        "reproducer_pool_row_id": CANARY_REPRODUCER_ID,
        "selection_policy": {
            "reproducer": "fixed known framing exhaustion reproducer",
            "request_form_coverage": "minimum pool_row_id per request_form in framing set",
            "framing_category_coverage": "minimum pool_row_id per original exec-4 framing category",
        },
        "count": len(canary_ids),
        "ids": canary_ids,
        "sha256": id_set_hash(canary_ids),
    }
    write_json(frozen / "framing_canary_ids.json", canary_manifest)

    write_json(
        frozen / "biased_overflow_regeneration_ids.json",
        {
            "count": len(bundle["overflow"]["biased_ce"]),
            "ids": bundle["overflow"]["biased_ce"],
            "sha256": id_set_hash(bundle["overflow"]["biased_ce"]),
            "teacher_contract": TEACHER_CONTRACT_V1,
        },
    )
    write_json(
        frozen / "framing_regeneration_ids.json",
        {
            "count": len(framing_ids),
            "ids": framing_ids,
            "sha256": EXPECTED_FRAMING_HASH,
            "teacher_contract": TEACHER_CONTRACT_V2,
        },
    )
    write_json(
        frozen / "normal_regeneration_ids.json",
        {
            "count": len(bundle["normal_regeneration_ids"]),
            "ids": bundle["normal_regeneration_ids"],
            "sha256": id_set_hash(bundle["normal_regeneration_ids"]),
            "generation_contract": "base_qwen_normal_overflow",
        },
    )

    superseded = archive_superseded_partial_v1_accepts(output_dir)
    surfaces = surface_hashes(bundle)
    teacher_v1_sha256 = teacher_template_digest(TEACHER_V1)
    teacher_v2_sha256 = teacher_template_digest(TEACHER_V2)

    manifest = {
        "artifact": "bounded_regeneration_v2",
        "phase": "protocol_frozen",
        "timestamp": now(),
        "retry_ceiling_per_contract_version": RETRY_CEILING,
        "attempt_provenance_policy": {
            "v1_attempts_do_not_count_toward_v2": True,
            "required_fields": [
                "teacher_contract_version",
                "regeneration_execution",
                "regeneration_attempt_index",
            ],
        },
        "canary_policy": {
            "must_pass_before_full_framing_launch": True,
            "block_on_reproducer_exhaustion": True,
            "successful_canary_rows_may_be_retained": True,
            "do_not_change_v2_after_canary": True,
            "acceptance": [
                "backend gpt-5-2025-08-07",
                "finish_reason=stop",
                "valid structured response",
                "nonempty answer",
                "trainer_equivalent<=2048",
                "zero truncation",
                "frozen v1.1 mechanical detector pass",
            ],
        },
        "contract_assignment": {
            "framing_189": {
                "teacher_contract": TEACHER_CONTRACT_V2,
                "template_path": str(TEACHER_V2),
                "template_sha256": teacher_v2_sha256,
                "count": 189,
                "id_set_sha256": EXPECTED_FRAMING_HASH,
            },
            "biased_overflow_26": {
                "teacher_contract": TEACHER_CONTRACT_V1,
                "template_path": str(TEACHER_V1),
                "template_sha256": teacher_v1_sha256,
                "count": 26,
                "id_set_sha256": surfaces["biased_overflow_only_sha256"],
            },
            "normal_overflow_21": {
                "generation_contract": "base_qwen_normal_overflow",
                "count": 21,
                "train_count": 20,
                "dev_count": 1,
                "id_set_sha256": surfaces["normal_overflow_sha256"],
            },
        },
        "bindings": {
            "execution_4_snapshot_sha256": ACTIVE_RAW_SNAPSHOT_SHA256,
            "builder_correction_v1_sha256": digest(CORRECTION / "CORRECTION_MANIFEST.json"),
            "teacher_v1_template_sha256": teacher_v1_sha256,
            "teacher_v2_template_sha256": teacher_v2_sha256,
            "detector_bundle_sha256": digest(DETECTOR_CONTRACT / "corrected_detectors.py"),
            "framing_set_sha256": EXPECTED_FRAMING_HASH,
            "biased_overflow_only_sha256": surfaces["biased_overflow_only_sha256"],
            "normal_overflow_sha256": surfaces["normal_overflow_sha256"],
            "provider_action_total_sha256": EXPECTED_PROVIDER_ACTION_HASH,
            "framing_canary_sha256": canary_manifest["sha256"],
            "overflow_union_sha256": EXPECTED_OVERFLOW_HASH,
        },
        "surfaces": surfaces,
        "superseded_partial_v1_remediation": superseded,
        "operations_not_performed": [
            "azure_provider_calls",
            "vllm_generation",
            "training",
            "detector_modification",
            "train_dev_membership_modification",
        ],
        "accepted_target_metadata_requirement": {
            "field": "teacher_contract_version",
            "values": [TEACHER_CONTRACT_V1, TEACHER_CONTRACT_V2, "base_qwen_normal_overflow"],
        },
    }
    write_json(output_dir / "BOUNDED_REGENERATION_V2_EXECUTION_MANIFEST.json", manifest)
    write_json(output_dir / "REGENERATION_INPUT_MANIFEST.json", manifest)
    return manifest


def cmd_freeze(args: argparse.Namespace) -> None:
    """CLI entrypoint for protocol freeze.

    Args:
        args: Parsed CLI arguments.
    """
    manifest = freeze_protocol(Path(args.output_dir))
    print(json.dumps(manifest, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    """Build the bounded-regeneration v2 CLI parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(EXEC_V2),
        help="Bounded regeneration v2 execution directory",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze", help="Freeze teacher v2 protocol without provider calls")
    return parser


def main() -> None:
    """Run the bounded-regeneration v2 CLI."""
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "freeze":
        cmd_freeze(args)
    else:
        raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
