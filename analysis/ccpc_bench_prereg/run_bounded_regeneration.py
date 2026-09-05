#!/usr/bin/env python3
"""Bounded CCPC500 regeneration for the 236 frozen provider-action IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from analysis.ccpc_bench_prereg.build_ccpc500_training_data import (
    ACTIVE_GENERATOR_CONFIG_SHA256,
    ACTIVE_RAW_SNAPSHOT_SHA256,
    DETECTOR_CONTRACT,
    MODEL,
    REVISION,
    MAX_LENGTH,
    digest,
    prompt_text,
    read_json,
    read_jsonl,
    teacher_attempt,
    tokenizer,
    write_json,
    write_jsonl,
)
from analysis.ccpc_bench_prereg.ccpc500_training_provenance import id_set_hash
from analysis.ccpc_bench_prereg.ccpc500_training_token_utils import train_token_count

PROBE = Path("/home/ubuntu/hirundo-research/probe/rsch76-ccpc500-training-data-v1")
CORRECTION = PROBE / "builder_correction_v1"
SNAPSHOT = PROBE / "execution_attempt_4/final_raw_generation_snapshot_20260830T053000Z"
ROUTING = Path(
    "/home/ubuntu/worktrees/hr-rsch76-unlearning-data/probe/rsch76-ccpc500-base-routing-v1"
)
CCPC500_FREEZE = Path(
    "/home/ubuntu/llm-behavior-eval/analysis/ccpc_bench_prereg/ccpc500_freeze_v3/ccpc500.jsonl"
)
EXEC = PROBE / "bounded_regeneration_v1"
EXPECTED_OVERFLOW_HASH = (
    "3f5a0c2e57131fe7b5b9dc52c7c47e427310fd71954052fb171394f18cc3b060"
)
EXPECTED_FRAMING_HASH = (
    "37786be8ba8be6ec2a4befedec71ee73660beb522dc1240044416a30c24af672"
)
EXPECTED_PROVIDER_ACTION_HASH = (
    "1be4e505490dd71e7f90a5bbc099b9a18819e4f0c920dfc2191975bfddc3bcf8"
)
EXPECTED_MODEL = "gpt-5-2025-08-07"


def now() -> str:
    """Return a UTC ISO timestamp.

    Returns:
        ISO-8601 timestamp string.
    """
    return datetime.now(UTC).isoformat()


def mechanical_row_qa() -> Any:
    """Import the frozen v1.1 mechanical QA detector.

    Returns:
        ``mechanical_row_qa`` callable.
    """
    if str(DETECTOR_CONTRACT) not in sys.path:
        sys.path.insert(0, str(DETECTOR_CONTRACT))
    from corrected_detectors import mechanical_row_qa as detector

    return detector


def verify_correction_bundle() -> dict[str, Any]:
    """Verify frozen correction-bundle hashes and set cardinalities.

    Returns:
        Parsed correction-bundle metadata.

    Raises:
        RuntimeError: If any frozen hash or cardinality check fails.
    """
    sums_path = CORRECTION / "SHA256SUMS"
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        path = CORRECTION / name
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"correction bundle hash failure: {name}")

    overflow = read_json(CORRECTION / "overflow_ids.json")
    framing = read_json(CORRECTION / "framing_ids.json")
    provider_action = read_json(CORRECTION / "provider_action_ids.json")

    if overflow["union_sha256"] != EXPECTED_OVERFLOW_HASH:
        raise RuntimeError("overflow union hash mismatch")
    if framing["sha256"] != EXPECTED_FRAMING_HASH:
        raise RuntimeError("framing hash mismatch")
    if provider_action["sha256"] != EXPECTED_PROVIDER_ACTION_HASH:
        raise RuntimeError("provider-action hash mismatch")
    if len(overflow["biased_ce"]) != 26:
        raise RuntimeError("expected 26 biased overflow IDs")
    if len(framing["ids"]) != 189:
        raise RuntimeError("expected 189 biased framing IDs")
    if len(overflow["normal_train"]) != 20 or len(overflow["normal_dev"]) != 1:
        raise RuntimeError("expected 21 normal overflow IDs")
    overlap = set(overflow["biased_ce"]) & set(framing["ids"])
    if overlap:
        raise RuntimeError(f"biased overflow/framing overlap: {sorted(overlap)}")
    biased_action = set(overflow["biased_ce"]) | set(framing["ids"])
    normal_action = set(overflow["normal_train"]) | set(overflow["normal_dev"])
    union = sorted(biased_action | normal_action)
    if len(union) != 236:
        raise RuntimeError(f"provider-action union size {len(union)} != 236")
    if id_set_hash(union) != EXPECTED_PROVIDER_ACTION_HASH:
        raise RuntimeError("recomputed provider-action hash mismatch")

    return {
        "overflow": overflow,
        "framing": framing,
        "provider_action": provider_action,
        "biased_regeneration_ids": sorted(biased_action),
        "normal_regeneration_ids": sorted(normal_action),
        "provider_action_ids": union,
    }


def freeze_inputs(output_dir: Path = EXEC) -> dict[str, Any]:
    """Persist immutable regeneration inputs and binding metadata.

    Args:
        output_dir: Execution directory for bounded regeneration.

    Returns:
        Frozen input manifest.
    """
    bundle = verify_correction_bundle()
    frozen = output_dir / "frozen_inputs"
    frozen.mkdir(parents=True, exist_ok=True)

    for name in ("overflow_ids.json", "framing_ids.json", "provider_action_ids.json"):
        shutil.copy2(CORRECTION / name, frozen / name)

    write_json(
        frozen / "biased_regeneration_ids.json",
        {
            "count": len(bundle["biased_regeneration_ids"]),
            "ids": bundle["biased_regeneration_ids"],
            "sha256": id_set_hash(bundle["biased_regeneration_ids"]),
        },
    )
    write_json(
        frozen / "normal_regeneration_ids.json",
        {
            "count": len(bundle["normal_regeneration_ids"]),
            "ids": bundle["normal_regeneration_ids"],
            "sha256": id_set_hash(bundle["normal_regeneration_ids"]),
        },
    )

    if digest(SNAPSHOT / "supervision_generator_config.json") != ACTIVE_GENERATOR_CONFIG_SHA256:
        raise RuntimeError("snapshot generator config hash mismatch")
    if digest(SNAPSHOT / "biased_generation_attempts.jsonl") != ACTIVE_RAW_SNAPSHOT_SHA256:
        raise RuntimeError("snapshot attempts ledger hash mismatch")

    manifest = {
        "artifact": "bounded_regeneration_v1",
        "phase": "frozen_inputs",
        "timestamp": now(),
        "bindings": {
            "execution_4_snapshot_sha256": ACTIVE_RAW_SNAPSHOT_SHA256,
            "builder_correction_bundle_sha256": digest(CORRECTION / "CORRECTION_MANIFEST.json"),
            "generator_config_sha256": ACTIVE_GENERATOR_CONFIG_SHA256,
            "detector_bundle_sha256": digest(DETECTOR_CONTRACT / "corrected_detectors.py"),
            "overflow_union_sha256": EXPECTED_OVERFLOW_HASH,
            "framing_set_sha256": EXPECTED_FRAMING_HASH,
            "provider_action_sha256": EXPECTED_PROVIDER_ACTION_HASH,
            "biased_replacement_sha256": id_set_hash(bundle["biased_regeneration_ids"]),
            "normal_replacement_sha256": id_set_hash(bundle["normal_regeneration_ids"]),
        },
        "counts": {
            "biased_regeneration": len(bundle["biased_regeneration_ids"]),
            "normal_regeneration": len(bundle["normal_regeneration_ids"]),
            "provider_action_total": len(bundle["provider_action_ids"]),
            "unaffected_rows": 7115,
        },
    }
    write_json(output_dir / "REGENERATION_INPUT_MANIFEST.json", manifest)
    return manifest


def biased_membership_index() -> dict[int, dict[str, Any]]:
    """Load biased membership keyed by pool_row_id.

    Returns:
        Membership rows indexed by ``pool_row_id``.
    """
    return {
        int(row["pool_row_id"]): row
        for row in read_jsonl(SNAPSHOT / "biased_membership.jsonl")
    }


def original_biased_winners() -> dict[int, dict[str, Any]]:
    """Load first parse-valid attempt per biased ID from the frozen snapshot.

    Returns:
        Winning attempt records keyed by ``pool_row_id``.
    """
    winners: dict[int, dict[str, Any]] = {}
    for item in read_jsonl(SNAPSHOT / "biased_generation_attempts.jsonl"):
        pool_row_id = int(item["pool_row_id"])
        if item.get("parse_valid") and pool_row_id not in winners:
            winners[pool_row_id] = item
    if len(winners) != 4856:
        raise RuntimeError(f"expected 4856 biased winners, found {len(winners)}")
    return winners


def accept_biased_candidate(
    row: dict[str, Any],
    answer: str,
    provider_response: dict[str, Any] | None,
    token: Any,
    detector: Any,
) -> tuple[bool, list[str]]:
    """Evaluate mechanical acceptance for one regenerated biased candidate.

    Args:
        row: Biased membership row.
        answer: Parsed answer text.
        provider_response: Provider response metadata.
        token: Frozen tokenizer.
        detector: Frozen mechanical QA callable.

    Returns:
        Tuple of acceptance flag and rejection reasons.
    """
    reasons: list[str] = []
    if not isinstance(answer, str) or not answer.strip():
        return False, ["empty_answer"]
    provider = provider_response or {}
    if provider.get("model") != EXPECTED_MODEL:
        reasons.append("unexpected_teacher_model_identity")
    choices = provider.get("choices") or [{}]
    if choices[0].get("finish_reason") != "stop":
        reasons.append("non_stop_finish_reason")
    rendered = train_token_count(token, row["question"], answer.strip())
    if rendered > MAX_LENGTH:
        reasons.append("rendered_length_exceeds_2048")
    verdict = detector(
        {
            "pool_row_id": int(row["pool_row_id"]),
            "request_form": row["request_form"],
            "desired_answer": answer.strip(),
            "rendered_total_tokens": rendered,
            "provider_response": provider_response,
        }
    )
    if verdict["status"] == "reject":
        reasons.extend(verdict["reject_reasons"])
    elif verdict["status"] == "hold_requires_verification":
        reasons.extend(verdict["hold_reasons"])
    framing = [
        reason
        for reason in reasons
        if reason.startswith("refusal_or_persona_framing")
    ]
    if framing:
        return False, reasons
    return not reasons, reasons


def regenerate_biased_row(
    row: dict[str, Any],
    config: dict[str, Any],
    template: str,
    token: Any,
    detector: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Regenerate one biased row until accepted or retry budget exhaustion.

    Args:
        row: Biased membership row.
        config: Frozen generator config.
        template: Frozen teacher template.
        token: Frozen tokenizer.
        detector: Frozen mechanical QA callable.

    Returns:
        Attempt ledger records and accepted payload, if any.
    """
    max_attempts = int(config["retry"]["max_attempts"])
    attempts: list[dict[str, Any]] = []
    for attempt_no in range(1, max_attempts + 1):
        single_attempt_config = json.loads(json.dumps(config))
        single_attempt_config["retry"]["max_attempts"] = 1
        batch, accepted = teacher_attempt(row, single_attempt_config, template)
        for item in batch:
            item["regeneration_execution"] = "bounded_regeneration_v1"
            item["regeneration_attempt_index"] = attempt_no
        attempts.extend(batch)
        if accepted is None:
            if attempt_no < max_attempts:
                continue
            return attempts, None
        answer = accepted["answer"]
        ok, reasons = accept_biased_candidate(
            row, answer, accepted["provider"], token, detector
        )
        if ok:
            return attempts, {
                "answer": answer,
                "attempt": attempt_no,
                "provider": accepted["provider"],
                "rendered_total_tokens": train_token_count(
                    token, row["question"], answer
                ),
            }
        attempts[-1]["mechanical_rejection_reasons"] = reasons
        if attempt_no < max_attempts:
            time.sleep(
                min(
                    config["retry"]["initial_backoff_seconds"] * 2 ** (attempt_no - 1),
                    config["retry"]["max_backoff_seconds"],
                )
            )
    return attempts, None


def regenerate_biased(args: argparse.Namespace) -> None:
    """Regenerate the 215 frozen biased provider-action IDs.

    Args:
        args: CLI arguments including worker count and output directory.
    """
    output_dir = Path(args.output_dir)
    freeze_inputs(output_dir)
    ids = read_json(output_dir / "frozen_inputs/biased_regeneration_ids.json")["ids"]
    membership = biased_membership_index()
    config = read_json(SNAPSHOT / "supervision_generator_config.json")
    template = (SNAPSHOT / "supervision_generator_template.txt").read_text(
        encoding="utf-8"
    )
    token = tokenizer()
    detector = mechanical_row_qa()
    regen_dir = output_dir / "biased_regeneration"
    regen_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = regen_dir / "regeneration_attempts.jsonl"
    accepted_path = regen_dir / "accepted_replacements.jsonl"

    completed: dict[int, dict[str, Any]] = {}
    if accepted_path.exists():
        for row in read_jsonl(accepted_path):
            completed[int(row["pool_row_id"])] = row

    pending = [membership[pool_row_id] for pool_row_id in ids if pool_row_id not in completed]
    with (
        ledger_path.open("a", encoding="utf-8") as ledger_handle,
        accepted_path.open("a", encoding="utf-8") as accepted_handle,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        futures = {
            pool.submit(
                regenerate_biased_row, row, config, template, token, detector
            ): row
            for row in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            attempts, accepted = future.result()
            for item in attempts:
                ledger_handle.write(
                    json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                )
            ledger_handle.flush()
            if accepted is None:
                raise RuntimeError(
                    f"BLOCK — biased regeneration exhausted for pool_row_id={row['pool_row_id']}"
                )
            replacement = {
                "pool_row_id": int(row["pool_row_id"]),
                "cluster_id": row["cluster_id"],
                "question": row["question"],
                "request_form": row["request_form"],
                "desired_answer": accepted["answer"],
                "generation_attempt": accepted["attempt"],
                "rendered_total_tokens": accepted["rendered_total_tokens"],
                "provider_response": accepted["provider"],
            }
            accepted_handle.write(
                json.dumps(replacement, ensure_ascii=False, sort_keys=True) + "\n"
            )
            accepted_handle.flush()
            completed[int(row["pool_row_id"])] = replacement
            if index % 10 == 0 or index == len(pending):
                print(
                    f"biased regeneration completed {index}/{len(pending)}",
                    flush=True,
                )

    if len(completed) != len(ids):
        raise RuntimeError("biased regeneration incomplete")


def trainer_equivalent_normal_cap(token: Any, question: str) -> tuple[str, int, int]:
    """Derive a safe vLLM continuation cap against trainer-equivalent render.

    Args:
        token: Frozen tokenizer.
        question: User question text.

    Returns:
        Tuple of generation prompt, safe max new tokens, and prompt token count.
    """
    prompt = prompt_text(token, question)
    prompt_count = len(token(prompt, add_special_tokens=False).input_ids)
    prefix = train_token_count(token, question, "")
    cap = MAX_LENGTH - prefix
    if cap <= 0:
        raise RuntimeError("normal prompt has no positive continuation budget")
    return prompt, cap, prompt_count


def regenerate_normal(args: argparse.Namespace) -> None:
    """Regenerate the 21 frozen normal overflow IDs.

    Args:
        args: CLI arguments including batch size and output directory.
    """
    from vllm import LLM, SamplingParams

    output_dir = Path(args.output_dir)
    freeze_inputs(output_dir)
    ids = set(read_json(output_dir / "frozen_inputs/normal_regeneration_ids.json")["ids"])
    token = tokenizer()
    engine = LLM(
        model=MODEL,
        revision=REVISION,
        trust_remote_code=True,
        language_model_only=True,
        max_model_len=MAX_LENGTH,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    regen_dir = output_dir / "normal_regeneration"
    regen_dir.mkdir(parents=True, exist_ok=True)
    accepted_rows: list[dict[str, Any]] = []

    for split in ("train", "dev"):
        rows = read_jsonl(PROBE / f"normal_{split}_membership.jsonl")
        original = {
            int(row["pool_row_id"]): row
            for row in read_jsonl(PROBE / f"normal_{split}_targets.jsonl")
        }
        pending = [row for row in rows if int(row["pool_row_id"]) in ids]
        destination = regen_dir / f"normal_{split}_replacements.jsonl"
        with destination.open("w", encoding="utf-8") as handle:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start : start + args.batch_size]
                rendered = [trainer_equivalent_normal_cap(token, row["question"]) for row in batch]
                outputs = engine.generate(
                    [item[0] for item in rendered],
                    [
                        SamplingParams(
                            temperature=0.0,
                            top_p=1.0,
                            top_k=0,
                            max_tokens=item[1],
                            repetition_penalty=1.10,
                        )
                        for item in rendered
                    ],
                )
                for row, (_, cap, prompt_count), output in zip(
                    batch, rendered, outputs, strict=True
                ):
                    completion = output.outputs[0]
                    total = train_token_count(token, row["question"], completion.text)
                    result = {
                        "pool_row_id": row["pool_row_id"],
                        "cluster_id": row["cluster_id"],
                        "question": row["question"],
                        "base_response": completion.text,
                        "response_token_ids": completion.token_ids,
                        "prompt_tokens": prompt_count,
                        "generated_tokens": len(completion.token_ids),
                        "rendered_total_tokens": total,
                        "effective_max_new_tokens": cap,
                        "finish_reason": completion.finish_reason,
                        "zero_truncation": total <= MAX_LENGTH,
                        "regeneration_execution": "bounded_regeneration_v1",
                    }
                    if total > MAX_LENGTH or completion.finish_reason == "length":
                        raise RuntimeError(
                            f"BLOCK — normal regeneration failed for pool_row_id={row['pool_row_id']}"
                        )
                    handle.write(
                        json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    accepted_rows.append(result)
                handle.flush()
                print(
                    f"normal {split} regeneration completed {min(start + len(batch), len(pending))}/{len(pending)}",
                    flush=True,
                )

    if len(accepted_rows) != 21:
        raise RuntimeError(f"expected 21 normal replacements, found {len(accepted_rows)}")


def build_repaired(args: argparse.Namespace) -> dict[str, Any]:
    """Construct the repaired supervision artifact without mutating history.

    Args:
        args: CLI arguments including output directory.

    Returns:
        Repaired-corpus summary mapping.
    """
    output_dir = Path(args.output_dir)
    repaired = output_dir / "repaired_supervision_v1"
    repaired.mkdir(parents=True, exist_ok=True)

    biased_ids = set(
        read_json(output_dir / "frozen_inputs/biased_regeneration_ids.json")["ids"]
    )
    normal_ids = set(
        read_json(output_dir / "frozen_inputs/normal_regeneration_ids.json")["ids"]
    )
    replacements = {
        int(row["pool_row_id"]): row
        for row in read_jsonl(output_dir / "biased_regeneration/accepted_replacements.jsonl")
    }
    normal_replacements: dict[int, dict[str, Any]] = {}
    for split in ("train", "dev"):
        for row in read_jsonl(output_dir / f"normal_regeneration/normal_{split}_replacements.jsonl"):
            normal_replacements[int(row["pool_row_id"])] = row

    winners = original_biased_winners()
    membership = biased_membership_index()
    biased_rows: list[dict[str, Any]] = []
    for pool_row_id in sorted(membership):
        if pool_row_id in biased_ids:
            biased_rows.append(replacements[pool_row_id])
            continue
        winner = winners[pool_row_id]
        answer = json.loads(winner["raw_structured_response"])["answer"].strip()
        biased_rows.append(
            {
                "pool_row_id": pool_row_id,
                "cluster_id": membership[pool_row_id]["cluster_id"],
                "question": membership[pool_row_id]["question"],
                "request_form": membership[pool_row_id]["request_form"],
                "desired_answer": answer,
                "generation_attempt": winner["attempt"],
                "rendered_total_tokens": train_token_count(
                    tokenizer(), membership[pool_row_id]["question"], answer
                ),
            }
        )
    write_jsonl(repaired / "biased_supervision.jsonl", biased_rows)

    for split in ("train", "dev"):
        rows = read_jsonl(PROBE / f"normal_{split}_targets.jsonl")
        repaired_rows = []
        for row in rows:
            pool_row_id = int(row["pool_row_id"])
            if pool_row_id in normal_ids:
                repaired_rows.append(normal_replacements[pool_row_id])
            else:
                repaired_rows.append(row)
        write_jsonl(repaired / f"normal_{split}_targets.jsonl", repaired_rows)

    shutil.copy2(PROBE / "anchor_membership.jsonl", repaired / "anchor_membership.jsonl")
    shutil.copy2(
        PROBE / "biased_membership.jsonl", repaired / "biased_membership.jsonl"
    )
    shutil.copy2(
        PROBE / "normal_train_membership.jsonl", repaired / "normal_train_membership.jsonl"
    )
    shutil.copy2(
        PROBE / "normal_dev_membership.jsonl", repaired / "normal_dev_membership.jsonl"
    )
    return {
        "repaired_dir": str(repaired),
        "biased_rows": len(biased_rows),
        "normal_replacements": len(normal_replacements),
    }


def reconcile_repaired(args: argparse.Namespace) -> dict[str, Any]:
    """Mechanically reconcile the repaired corpus.

    Args:
        args: CLI arguments including output directory.

    Returns:
        Reconciliation report mapping.
    """
    output_dir = Path(args.output_dir)
    repaired = output_dir / "repaired_supervision_v1"
    token = tokenizer()
    detector = mechanical_row_qa()
    biased_ids = set(
        read_json(output_dir / "frozen_inputs/biased_regeneration_ids.json")["ids"]
    )
    normal_ids = set(
        read_json(output_dir / "frozen_inputs/normal_regeneration_ids.json")["ids"]
    )
    action_ids = set(
        read_json(output_dir / "frozen_inputs/provider_action_ids.json")["ids"]
    )

    biased = read_jsonl(repaired / "biased_supervision.jsonl")
    normal_train = read_jsonl(repaired / "normal_train_targets.jsonl")
    normal_dev = read_jsonl(repaired / "normal_dev_targets.jsonl")
    anchors = read_jsonl(repaired / "anchor_membership.jsonl")
    original_winners = original_biased_winners()
    originals_biased = {
        pool_row_id: json.loads(item["raw_structured_response"])["answer"].strip()
        for pool_row_id, item in original_winners.items()
    }
    originals_normal = {
        (split, int(row["pool_row_id"])): row["base_response"]
        for split in ("train", "dev")
        for row in read_jsonl(PROBE / f"normal_{split}_targets.jsonl")
    }

    unchanged_text_checks = 0
    replacements = 0
    ledger: list[dict[str, Any]] = []
    framing_failures = 0
    max_by_domain: dict[str, int] = {}

    for row in biased:
        pool_row_id = int(row["pool_row_id"])
        rendered = train_token_count(token, row["question"], row["desired_answer"])
        ledger.append(
            {
                "domain": "biased_ce",
                "pool_row_id": pool_row_id,
                "trainer_equivalent_tokens": rendered,
                "overflow": rendered > MAX_LENGTH,
            }
        )
        max_by_domain["biased_ce"] = max(max_by_domain.get("biased_ce", 0), rendered)
        verdict = detector(
            {
                "pool_row_id": pool_row_id,
                "request_form": row["request_form"],
                "desired_answer": row["desired_answer"],
                "rendered_total_tokens": rendered,
                "provider_response": row.get("provider_response"),
            }
        )
        if verdict["status"] != "accept":
            framing_failures += 1
        if pool_row_id in biased_ids:
            replacements += 1
        else:
            unchanged_text_checks += 1
            if row["desired_answer"] != originals_biased[pool_row_id]:
                raise RuntimeError(f"unaffected biased text changed: {pool_row_id}")

    routing = {
        int(row["pool_row_id"]): row
        for row in read_jsonl(ROUTING / "routing_records.jsonl")
    }
    for member in anchors:
        pool_row_id = int(member["pool_row_id"])
        route = routing[pool_row_id]
        rendered = train_token_count(token, route["question"], route["target_answer"])
        ledger.append(
            {
                "domain": "anchor",
                "pool_row_id": pool_row_id,
                "trainer_equivalent_tokens": rendered,
                "overflow": rendered > MAX_LENGTH,
            }
        )
        max_by_domain["anchor"] = max(max_by_domain.get("anchor", 0), rendered)

    for split, rows in (("normal_train", normal_train), ("normal_dev", normal_dev)):
        for row in rows:
            pool_row_id = int(row["pool_row_id"])
            rendered = train_token_count(token, row["question"], row["base_response"])
            ledger.append(
                {
                    "domain": split,
                    "pool_row_id": pool_row_id,
                    "trainer_equivalent_tokens": rendered,
                    "overflow": rendered > MAX_LENGTH,
                }
            )
            max_by_domain[split] = max(max_by_domain.get(split, 0), rendered)
            if pool_row_id in normal_ids:
                replacements += 1
            else:
                unchanged_text_checks += 1
                split_name = "train" if split == "normal_train" else "dev"
                if row["base_response"] != originals_normal[(split_name, pool_row_id)]:
                    raise RuntimeError(f"unaffected normal text changed: {pool_row_id}")

    ccpc_ids = set()
    if CCPC500_FREEZE.exists():
        ccpc_ids = {
            int(json.loads(line)["pool_row_id"])
            for line in CCPC500_FREEZE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    training_ids = {int(row["pool_row_id"]) for row in biased}
    training_ids |= {int(row["pool_row_id"]) for row in normal_train + normal_dev}
    training_ids |= {int(row["pool_row_id"]) for row in anchors}

    report = {
        "timestamp": now(),
        "domain_counts": {
            "biased_ce": len(biased),
            "anchor": len(anchors),
            "normal_train": len(normal_train),
            "normal_dev": len(normal_dev),
            "total": len(ledger),
        },
        "duplicate_ids": _duplicate_id_report(biased, normal_train, normal_dev, anchors),
        "overflow_rows": sum(1 for row in ledger if row["overflow"]),
        "framing_failures": framing_failures,
        "replacements": replacements,
        "unaffected_text_checks": unchanged_text_checks,
        "max_trainer_render_tokens_by_domain": max_by_domain,
        "ccpc500_overlap_with_training_ids": sorted(training_ids & ccpc_ids),
        "result": "PASS"
        if len(biased) == 4856
        and len(anchors) == 420
        and len(normal_train) == 1871
        and len(normal_dev) == 204
        and len(ledger) == 7351
        and replacements == len(action_ids)
        and framing_failures == 0
        and not any(row["overflow"] for row in ledger)
        and not (training_ids & ccpc_ids)
        else "BLOCK",
    }
    write_jsonl(repaired / "corrected_token_count_ledger.jsonl", ledger)
    write_json(output_dir / "REPAIR_RECONCILIATION.json", report)
    if report["result"] != "PASS":
        raise RuntimeError("repaired corpus reconciliation failed")
    return report


def _duplicate_id_report(*row_groups: list[dict[str, Any]]) -> dict[str, int]:
    """Count duplicate pool_row_id occurrences across domains.

    Args:
        row_groups: Domain row lists.

    Returns:
        Duplicate-ID summary mapping.
    """
    counter = Counter(int(row["pool_row_id"]) for group in row_groups for row in group)
    return {
        "duplicate_ids": sum(1 for count in counter.values() if count > 1),
        "total_unique_ids": len(counter),
    }


def write_provenance_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Write execution-bound regeneration provenance manifest.

    Args:
        args: CLI arguments including output directory.

    Returns:
        Provenance manifest mapping.
    """
    output_dir = Path(args.output_dir)
    repaired = output_dir / "repaired_supervision_v1"
    inventory = [
        f"{digest(path)}  {path.relative_to(repaired)}"
        for path in sorted(repaired.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (repaired / "SHA256SUMS").write_text("\n".join(inventory) + "\n", encoding="utf-8")
    manifest = {
        "artifact": "bounded_regeneration_v1",
        "timestamp": now(),
        "source_execution_4_snapshot_sha256": ACTIVE_RAW_SNAPSHOT_SHA256,
        "builder_correction_v1_sha256": digest(CORRECTION / "CORRECTION_MANIFEST.json"),
        "detector_bundle_sha256": digest(DETECTOR_CONTRACT / "corrected_detectors.py"),
        "provider_action_sha256": EXPECTED_PROVIDER_ACTION_HASH,
        "repaired_corpus_sha256": digest(repaired / "SHA256SUMS"),
        "repaired_dir": str(repaired),
        "historical_qa_status": "NON_AUTHORITATIVE_PENDING_FRESH_SUPERVISION_QA",
    }
    write_json(output_dir / "REGENERATION_PROVENANCE_MANIFEST.json", manifest)
    return manifest


def write_runtime_status(path: Path, payload: dict[str, Any]) -> None:
    """Persist launcher runtime status for a detached regeneration job.

    Args:
        path: Runtime status artifact path.
        payload: Status mapping.
    """
    write_json(path, payload)


def run_all(args: argparse.Namespace) -> None:
    """Execute freeze, regeneration, repair, reconciliation, and provenance phases.

    Args:
        args: CLI arguments including output directory.
    """
    output_dir = Path(args.output_dir)
    status_path = Path(args.status_path) if getattr(args, "status_path", None) else None
    try:
        freeze_inputs(output_dir)
        regenerate_biased(args)
        regenerate_normal(args)
        build_repaired(args)
        reconcile_repaired(args)
        write_provenance_manifest(args)
        if status_path is not None:
            write_runtime_status(
                status_path,
                {
                    "phase": "completed",
                    "completed_at": now(),
                    "exit_code": 0,
                    "signal": None,
                    "output_dir": str(output_dir),
                },
            )
    except Exception as exc:
        if status_path is not None:
            write_runtime_status(
                status_path,
                {
                    "phase": "failed",
                    "completed_at": now(),
                    "exit_code": 1,
                    "signal": None,
                    "output_dir": str(output_dir),
                    "exception_class": type(exc).__name__,
                    "message": str(exc),
                },
            )
        raise


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=EXEC)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("freeze").set_defaults(func=lambda a: freeze_inputs(Path(a.output_dir)))
    biased = sub.add_parser("regenerate-biased")
    biased.add_argument("--workers", type=int, default=8)
    biased.set_defaults(func=regenerate_biased)

    normal = sub.add_parser("regenerate-normal")
    normal.add_argument("--batch-size", type=int, default=8)
    normal.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    normal.set_defaults(func=regenerate_normal)

    sub.add_parser("build-repaired").set_defaults(func=build_repaired)
    sub.add_parser("reconcile").set_defaults(func=reconcile_repaired)
    sub.add_parser("provenance").set_defaults(func=write_provenance_manifest)

    run_all_parser = sub.add_parser("run-all")
    run_all_parser.add_argument("--workers", type=int, default=8)
    run_all_parser.add_argument("--batch-size", type=int, default=8)
    run_all_parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    run_all_parser.add_argument(
        "--status-path",
        type=Path,
        default=None,
        help="Optional runtime status artifact updated on terminal completion",
    )
    run_all_parser.set_defaults(func=run_all)

    args = parser.parse_args()
    result = args.func(args)
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
