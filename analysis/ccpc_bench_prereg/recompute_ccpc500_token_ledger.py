#!/usr/bin/env python3
"""Read-only trainer-equivalent token recount for all CCPC500 training domains."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from analysis.ccpc_bench_prereg.ccpc500_training_provenance import id_set_hash
from analysis.ccpc_bench_prereg.ccpc500_training_token_utils import (
    MAX_LENGTH,
    load_frozen_tokenizer,
    train_token_count,
    trainer_equivalent_messages,
)

PROBE_ROOT = Path("/home/ubuntu/hirundo-research/probe/rsch76-ccpc500-training-data-v1")
SNAPSHOT = PROBE_ROOT / "execution_attempt_4/final_raw_generation_snapshot_20260830T053000Z"
ROUTING = Path(
    "/home/ubuntu/worktrees/hr-rsch76-unlearning-data/probe/rsch76-ccpc500-base-routing-v1"
)
DETECTOR_CONTRACT = (
    Path("/home/ubuntu/hirundo-research")
    / "probe/rsch76-ccpc500-supervision-qa-contract-v1.1"
)
EXPECTED_OVERFLOW_HASH = (
    "3f5a0c2e57131fe7b5b9dc52c7c47e427310fd71954052fb171394f18cc3b060"
)
EXPECTED_FRAMING_HASH = (
    "37786be8ba8be6ec2a4befedec71ee73660beb522dc1240044416a30c24af672"
)
EXPECTED_PROVIDER_ACTION_HASH = (
    "1be4e505490dd71e7f90a5bbc099b9a18819e4f0c920dfc2191975bfddc3bcf8"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows from disk.

    Args:
        path: JSONL file path.

    Returns:
        Parsed row mappings.
    """
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write stable JSONL rows.

    Args:
        path: Destination JSONL path.
        rows: Row mappings to serialize.
    """
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    """Write stable JSON.

    Args:
        path: Destination JSON path.
        value: JSON-serializable value.
    """
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def builder_incomplete_count(tokenizer_value: Any, question: str, answer: str) -> int:
    """Count tokens under the deprecated builder-only user+assistant render.

    Args:
        tokenizer_value: Frozen tokenizer.
        question: User question text.
        answer: Assistant answer text.

    Returns:
        Token count for the legacy incomplete render.
    """
    from analysis.ccpc_bench_prereg.ccpc500_training_token_utils import (
        extract_chat_token_ids,
    )

    rendered = tokenizer_value.apply_chat_template(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        return_dict=True,
    )
    return len(extract_chat_token_ids(rendered))


def mechanical_row_qa() -> Any:
    """Import the frozen v1.1 mechanical QA detector.

    Returns:
        ``mechanical_row_qa`` callable.
    """
    if str(DETECTOR_CONTRACT) not in sys.path:
        sys.path.insert(0, str(DETECTOR_CONTRACT))
    from corrected_detectors import mechanical_row_qa as detector

    return detector


def recount(output_dir: Path) -> dict[str, Any]:
    """Recompute trainer-equivalent token counts for all 7,351 rows.

    Args:
        output_dir: Destination directory for derived correction artifacts.

    Returns:
        Summary manifest payload.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_value = load_frozen_tokenizer()
    detector = mechanical_row_qa()
    ledger: list[dict[str, Any]] = []

    members = {
        int(row["pool_row_id"]): row
        for row in read_jsonl(SNAPSHOT / "biased_membership.jsonl")
    }
    winners: dict[int, dict[str, Any]] = {}
    for item in read_jsonl(SNAPSHOT / "biased_generation_attempts.jsonl"):
        pool_row_id = int(item["pool_row_id"])
        if item.get("parse_valid") and pool_row_id not in winners:
            winners[pool_row_id] = item
    for pool_row_id, member in sorted(members.items()):
        winner = winners[pool_row_id]
        answer = json.loads(winner["raw_structured_response"])["answer"].strip()
        trainer_tokens = train_token_count(tokenizer_value, member["question"], answer)
        ledger.append(
            {
                "domain": "biased_ce",
                "pool_row_id": pool_row_id,
                "trainer_equivalent_tokens": trainer_tokens,
                "overflow": trainer_tokens > MAX_LENGTH,
            }
        )

    routing = {
        int(row["pool_row_id"]): row for row in read_jsonl(ROUTING / "routing_records.jsonl")
    }
    for member in read_jsonl(PROBE_ROOT / "anchor_membership.jsonl"):
        pool_row_id = int(member["pool_row_id"])
        route = routing[pool_row_id]
        trainer_tokens = train_token_count(
            tokenizer_value, route["question"], route["target_answer"]
        )
        ledger.append(
            {
                "domain": "anchor",
                "pool_row_id": pool_row_id,
                "trainer_equivalent_tokens": trainer_tokens,
                "overflow": trainer_tokens > MAX_LENGTH,
            }
        )

    for split in ("normal_train", "normal_dev"):
        for row in read_jsonl(PROBE_ROOT / f"{split}_targets.jsonl"):
            trainer_tokens = train_token_count(
                tokenizer_value, row["question"], row["base_response"]
            )
            entry = {
                "domain": split,
                "pool_row_id": int(row["pool_row_id"]),
                "trainer_equivalent_tokens": trainer_tokens,
                "overflow": trainer_tokens > MAX_LENGTH,
            }
            if int(row["pool_row_id"]) == 1271 and split == "normal_train":
                entry["builder_incomplete_tokens"] = builder_incomplete_count(
                    tokenizer_value, row["question"], row["base_response"]
                )
            ledger.append(entry)

    if len(ledger) != 7351:
        raise RuntimeError(f"expected 7,351 recount rows, found {len(ledger)}")

    token_values = [row["trainer_equivalent_tokens"] for row in ledger]
    if Counter(token_values).get(2, 0) == len(ledger):
        raise RuntimeError("corpus-wide constant token count of 2 detected")

    overflow_ids = sorted(row["pool_row_id"] for row in ledger if row["overflow"])
    overflow_by_domain = {
        domain: sorted(
            row["pool_row_id"] for row in ledger if row["domain"] == domain and row["overflow"]
        )
        for domain in ("biased_ce", "anchor", "normal_train", "normal_dev")
    }
    overflow_hash = id_set_hash(overflow_ids)
    if overflow_hash != EXPECTED_OVERFLOW_HASH:
        raise RuntimeError(
            "overflow union hash mismatch: "
            f"expected {EXPECTED_OVERFLOW_HASH}, got {overflow_hash}"
        )

    framing_ids: list[int] = []
    for pool_row_id, member in sorted(members.items()):
        winner = winners[pool_row_id]
        answer = json.loads(winner["raw_structured_response"])["answer"].strip()
        verdict = detector(
            {
                "pool_row_id": pool_row_id,
                "request_form": member["request_form"],
                "desired_answer": answer,
                "rendered_total_tokens": train_token_count(
                    tokenizer_value, member["question"], answer
                ),
                "provider_response": winner.get("provider_response"),
            }
        )
        if any(
            reason.startswith("refusal_or_persona_framing")
            for reason in verdict.get("reject_reasons", [])
        ):
            framing_ids.append(pool_row_id)
    framing_ids = sorted(framing_ids)
    framing_hash = id_set_hash(framing_ids)
    if framing_hash != EXPECTED_FRAMING_HASH:
        raise RuntimeError(
            "framing hash mismatch: "
            f"expected {EXPECTED_FRAMING_HASH}, got {framing_hash}"
        )

    provider_action_ids = sorted(set(overflow_by_domain["biased_ce"]) | set(framing_ids) | set(overflow_by_domain["normal_train"]) | set(overflow_by_domain["normal_dev"]))
    provider_action_hash = id_set_hash(provider_action_ids)
    if provider_action_hash != EXPECTED_PROVIDER_ACTION_HASH:
        raise RuntimeError(
            "provider-action hash mismatch: "
            f"expected {EXPECTED_PROVIDER_ACTION_HASH}, got {provider_action_hash}"
        )

    row_1271 = next(
        row for row in ledger if row["pool_row_id"] == 1271 and row["domain"] == "normal_train"
    )
    if row_1271.get("builder_incomplete_tokens") != 2048:
        raise RuntimeError("normal TRAIN ID 1271 builder-incomplete render is not 2048")
    if row_1271["trainer_equivalent_tokens"] != 2053:
        raise RuntimeError("normal TRAIN ID 1271 trainer-equivalent render is not 2053")

    write_jsonl(output_dir / "corrected_token_count_ledger.jsonl", ledger)
    write_json(
        output_dir / "overflow_ids.json",
        {
            "biased_ce": overflow_by_domain["biased_ce"],
            "anchor": overflow_by_domain["anchor"],
            "normal_train": overflow_by_domain["normal_train"],
            "normal_dev": overflow_by_domain["normal_dev"],
            "union": overflow_ids,
            "union_count": len(overflow_ids),
            "union_sha256": overflow_hash,
        },
    )
    write_json(
        output_dir / "framing_ids.json",
        {
            "ids": framing_ids,
            "count": len(framing_ids),
            "sha256": framing_hash,
        },
    )
    write_json(
        output_dir / "provider_action_ids.json",
        {
            "biased_overflow_count": len(overflow_by_domain["biased_ce"]),
            "biased_framing_count": len(framing_ids),
            "normal_overflow_count": len(overflow_by_domain["normal_train"])
            + len(overflow_by_domain["normal_dev"]),
            "overlap_biased_overflow_framing": 0,
            "ids": provider_action_ids,
            "count": len(provider_action_ids),
            "no_provider_action_rows": 7351 - len(provider_action_ids),
            "sha256": provider_action_hash,
        },
    )

    summary = {
        "total_rows": len(ledger),
        "overflow": {
            "biased_ce": len(overflow_by_domain["biased_ce"]),
            "anchor": len(overflow_by_domain["anchor"]),
            "normal_train": len(overflow_by_domain["normal_train"]),
            "normal_dev": len(overflow_by_domain["normal_dev"]),
            "union": len(overflow_ids),
            "union_sha256": overflow_hash,
        },
        "framing": {
            "count": len(framing_ids),
            "sha256": framing_hash,
        },
        "provider_action": {
            "count": len(provider_action_ids),
            "no_provider_action_rows": 7351 - len(provider_action_ids),
            "sha256": provider_action_hash,
        },
        "row_1271": {
            "builder_incomplete_tokens": row_1271["builder_incomplete_tokens"],
            "trainer_equivalent_tokens": row_1271["trainer_equivalent_tokens"],
        },
        "rendering_definition": {
            "messages": trainer_equivalent_messages("example", "answer"),
            "tokenizer": "Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "max_length": MAX_LENGTH,
        },
    }
    write_json(output_dir / "recount_summary.json", summary)
    return summary


def main() -> None:
    """Execute the read-only recount and write correction artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROBE_ROOT / "builder_correction_v1",
    )
    args = parser.parse_args()
    summary = recount(args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
