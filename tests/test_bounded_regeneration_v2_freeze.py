"""Protocol-freeze coverage for bounded regeneration teacher v2."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.ccpc_bench_prereg.ccpc500_training_provenance import id_set_hash
from analysis.ccpc_bench_prereg.run_bounded_regeneration_v2 import (
    CANARY_REPRODUCER_ID,
    EXPECTED_FRAMING_HASH,
    EXPECTED_PROVIDER_ACTION_HASH,
    TEACHER_CONTRACT_V2,
    archive_superseded_partial_v1_accepts,
    freeze_protocol,
    select_framing_canary_ids,
)


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point bounded-regeneration v2 helpers at a temporary execution root."""
    output_dir = tmp_path / "bounded_regeneration_v2"
    monkeypatch.setattr(
        "analysis.ccpc_bench_prereg.run_bounded_regeneration_v2.EXEC_V2",
        output_dir,
    )
    return output_dir


def test_select_framing_canary_ids_is_deterministic_and_includes_reproducer() -> None:
    """Canary selection must always include the known reproducer and stay stable."""
    membership = {
        407: {
            "pool_row_id": 407,
            "request_form": "resource_or_information_request",
            "question": "q407",
        },
        90: {
            "pool_row_id": 90,
            "request_form": "creative_or_document_generation",
            "question": "q90",
        },
        356: {
            "pool_row_id": 356,
            "request_form": "procedural_or_advocacy",
            "question": "q356",
        },
    }
    winners = {
        407: {
            "raw_structured_response": json.dumps(
                {"answer": "I can't summarize that without a link."}
            )
        },
        90: {
            "raw_structured_response": json.dumps(
                {"answer": "I don't have access to classified records."}
            )
        },
        356: {
            "raw_structured_response": json.dumps(
                {"answer": "As an AI, I cannot help with that."}
            )
        },
    }
    framing_ids = [407, 90, 356]
    first = select_framing_canary_ids(framing_ids, membership, winners)
    second = select_framing_canary_ids(framing_ids, membership, winners)
    assert first == second
    assert CANARY_REPRODUCER_ID in first
    assert first == sorted(set(first))


def test_freeze_protocol_writes_manifest_and_supersession(
    sandbox: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Freeze must bind surfaces, canary IDs, and superseded partial v1 accepts."""
    partial = sandbox.parent / "partial_v1" / "accepted_replacements.jsonl"
    partial.parent.mkdir(parents=True)
    partial.write_text(
        json.dumps({"pool_row_id": 90, "desired_answer": "kept"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "analysis.ccpc_bench_prereg.run_bounded_regeneration_v2.PARTIAL_V1_ACCEPTS",
        partial,
    )

    manifest = freeze_protocol(sandbox)

    assert manifest["artifact"] == "bounded_regeneration_v2"
    assert manifest["bindings"]["framing_set_sha256"] == EXPECTED_FRAMING_HASH
    assert (
        manifest["bindings"]["provider_action_total_sha256"]
        == EXPECTED_PROVIDER_ACTION_HASH
    )
    assert manifest["contract_assignment"]["framing_189"]["teacher_contract"] == (
        TEACHER_CONTRACT_V2
    )
    assert manifest["contract_assignment"]["biased_overflow_26"]["teacher_contract"] == (
        "supervision_generator_template_v1"
    )
    assert manifest["surfaces"]["counts"]["fresh_output_total"] == 236
    assert manifest["superseded_partial_v1_remediation"]["status"] == (
        "SUPERSEDED_BY_TEACHER_V2"
    )

    canary = json.loads(
        (sandbox / "frozen_inputs/framing_canary_ids.json").read_text(encoding="utf-8")
    )
    assert CANARY_REPRODUCER_ID in canary["ids"]
    assert canary["sha256"] == id_set_hash(canary["ids"])

    manifest_path = sandbox / "BOUNDED_REGENERATION_V2_EXECUTION_MANIFEST.json"
    assert manifest_path.is_file()
