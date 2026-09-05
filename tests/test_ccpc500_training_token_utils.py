"""Regression tests for CCPC500 trainer-equivalent token counting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.ccpc_bench_prereg.build_ccpc500_training_data import (
    BUILDER_MIRROR_FILES,
    CANONICAL_BUILDER_DIR,
    OUTPUT,
    digest,
    materialize_builder_mirror,
    qa_report_status,
)
from analysis.ccpc_bench_prereg.ccpc500_training_provenance import (
    assert_authoritative_for_training,
    build_active_bindings,
    resolve_qa_status,
)
from analysis.ccpc_bench_prereg.ccpc500_training_token_utils import (
    extract_chat_token_ids,
    load_frozen_tokenizer,
    train_token_count,
    trainer_equivalent_messages,
)

DETECTOR_CONTRACT = (
    Path("/home/ubuntu/hirundo-research")
    / "probe/rsch76-ccpc500-supervision-qa-contract-v1.1"
)


def test_extract_ids_from_list() -> None:
    """List tokenizer returns should yield actual token ids."""
    assert extract_chat_token_ids([11, 12, 13]) == [11, 12, 13]
    assert extract_chat_token_ids([[11, 12, 13]]) == [11, 12, 13]


def test_extract_ids_from_dict() -> None:
    """Dict tokenizer returns should read input_ids."""
    assert extract_chat_token_ids({"input_ids": [4, 5, 6]}) == [4, 5, 6]
    assert extract_chat_token_ids({"input_ids": [[4, 5, 6]]}) == [4, 5, 6]


def test_extract_ids_from_batch_encoding() -> None:
    """BatchEncoding returns should read input_ids, not key count."""
    from transformers.tokenization_utils_base import BatchEncoding

    batch_encoding = BatchEncoding(
        {"input_ids": [7, 8, 9, 10], "attention_mask": [1, 1, 1, 1]}
    )
    assert extract_chat_token_ids(batch_encoding) == [7, 8, 9, 10]
    assert len(batch_encoding) == 2
    assert len(batch_encoding["input_ids"]) == 4


def test_extract_ids_rejects_unknown_shape() -> None:
    """Unknown tokenizer return shapes must fail closed."""
    with pytest.raises(TypeError):
        extract_chat_token_ids("not-a-tokenizer-result")


def test_trainer_render_structure() -> None:
    """Trainer-equivalent render must include system, user newline, assistant."""
    messages = trainer_equivalent_messages("question", "answer")
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "question\n"}
    assert messages[2] == {"role": "assistant", "content": "answer"}


@pytest.mark.integration
def test_real_frozen_qwen_tokenizer_integration() -> None:
    """Frozen Qwen tokenizer integration must return non-constant counts."""
    tokenizer_value = load_frozen_tokenizer()
    count = train_token_count(tokenizer_value, "What happened?", "A detailed answer.")
    assert count > 10
    assert count != 2


@pytest.mark.integration
def test_row_1271_trainer_render_regression() -> None:
    """Normal TRAIN ID 1271 must be 2048 incomplete and 2053 trainer-equivalent."""
    targets = Path(
        "/home/ubuntu/hirundo-research/probe/rsch76-ccpc500-training-data-v1/normal_train_targets.jsonl"
    )
    row = next(
        json.loads(line)
        for line in targets.open(encoding="utf-8")
        if json.loads(line)["pool_row_id"] == 1271
    )
    tokenizer_value = load_frozen_tokenizer()
    trainer_count = train_token_count(
        tokenizer_value, row["question"], row["base_response"]
    )
    incomplete = tokenizer_value.apply_chat_template(
        [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["base_response"]},
        ],
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_dict=True,
    )
    incomplete_count = len(extract_chat_token_ids(incomplete))
    assert incomplete_count == 2048
    assert trainer_count == 2053


@pytest.mark.integration
def test_no_corpus_constant_two_on_real_row() -> None:
    """A real supervision row must not produce the constant-2 failure mode."""
    targets = Path(
        "/home/ubuntu/hirundo-research/probe/rsch76-ccpc500-training-data-v1/normal_train_targets.jsonl"
    )
    row = json.loads(next(targets.open(encoding="utf-8")))
    tokenizer_value = load_frozen_tokenizer()
    assert train_token_count(tokenizer_value, row["question"], row["base_response"]) != 2


def test_stale_snapshot_provenance_rejection() -> None:
    """Stale QA bindings must resolve as HISTORICAL_STALE and fail training gate."""
    active = build_active_bindings(
        execution_id="execution_attempt_4",
        raw_snapshot_sha256="dc2bc15f49ed599246b7fec6cc66159a80ad4188ac369b9873d7e39856cba054",
        accepted_valid_id_hash_sha256="cd51a8773997a5c7fb8c196f165ba768822ada2363b01b8efb4adbed5deee32b",
        generator_config_sha256="4a6322828ed436e9015ed7ea90684ce7d33cb53da8ea205d24396880787205c3",
        detector_bundle_sha256="be7013e3cd02521e33c115e6d3fed1d20c22f9ebef2d461de757fe2b1c7ceacc",
        qa_artifact_path="VALIDATION_REPORT.json",
    )
    stale = dict(active)
    stale["raw_snapshot_sha256"] = "deadbeef"
    assert resolve_qa_status(stale, active) == "HISTORICAL_STALE"
    with pytest.raises(RuntimeError, match="HISTORICAL_STALE"):
        assert_authoritative_for_training({**stale, "status": "HISTORICAL_STALE"})
    assert qa_report_status(OUTPUT / "VALIDATION_REPORT.json", active) == "HISTORICAL_STALE"


def test_frozen_refusal_detector_wiring() -> None:
    """Frozen v1.1 detector must flag opening refusal framing."""
    import sys

    if str(DETECTOR_CONTRACT) not in sys.path:
        sys.path.insert(0, str(DETECTOR_CONTRACT))
    from corrected_detectors import mechanical_row_qa

    verdict = mechanical_row_qa(
        {
            "pool_row_id": 1,
            "request_form": "factual_question",
            "desired_answer": "I can't help with that request. Here is the answer anyway.",
            "rendered_total_tokens": 100,
            "provider_response": {"model": "gpt-5-2025-08-07"},
        }
    )
    assert verdict["status"] == "reject"
    assert any(
        reason.startswith("refusal_or_persona_framing")
        for reason in verdict["reject_reasons"]
    )


def test_canonical_builder_mirror_consistency(tmp_path: Path) -> None:
    """Mirrored builder files must remain byte-identical to canonical sources."""
    destination = tmp_path / "mirror"
    digests = materialize_builder_mirror(destination)
    assert set(digests) == set(BUILDER_MIRROR_FILES)
    for name in BUILDER_MIRROR_FILES:
        source = CANONICAL_BUILDER_DIR / name
        target = destination / name
        assert digest(source) == digest(target)


@pytest.mark.integration
def test_recount_expected_hashes() -> None:
    """Read-only recount must reproduce the accepted overflow and action hashes."""
    from analysis.ccpc_bench_prereg.recompute_ccpc500_token_ledger import recount

    summary = recount(
        Path("/home/ubuntu/hirundo-research/probe/rsch76-ccpc500-training-data-v1/builder_correction_v1/_pytest_recount")
    )
    assert summary["total_rows"] == 7351
    assert summary["overflow"]["union"] == 47
    assert summary["overflow"]["union_sha256"] == (
        "3f5a0c2e57131fe7b5b9dc52c7c47e427310fd71954052fb171394f18cc3b060"
    )
    assert summary["framing"]["count"] == 189
    assert summary["framing"]["sha256"] == (
        "37786be8ba8be6ec2a4befedec71ee73660beb522dc1240044416a30c24af672"
    )
    assert summary["provider_action"]["count"] == 236
    assert summary["provider_action"]["no_provider_action_rows"] == 7115
    assert summary["provider_action"]["sha256"] == (
        "1be4e505490dd71e7f90a5bbc099b9a18819e4f0c920dfc2191975bfddc3bcf8"
    )
