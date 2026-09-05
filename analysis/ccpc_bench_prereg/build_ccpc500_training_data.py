#!/usr/bin/env python3
"""Construct the resumable, no-gradient CCPC500 training-data artifact."""

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

_BUILDER_DIR = Path(__file__).resolve().parent
if str(_BUILDER_DIR) not in sys.path:
    sys.path.insert(0, str(_BUILDER_DIR))

try:
    from analysis.ccpc_bench_prereg.ccpc500_training_provenance import (
        assert_authoritative_for_training,
        build_active_bindings,
        resolve_qa_status,
        write_authoritative_qa_index,
    )
    from analysis.ccpc_bench_prereg.ccpc500_training_token_utils import (
        MAX_LENGTH,
        MODEL,
        REVISION,
        train_token_count,
    )
except ImportError:
    from ccpc500_training_provenance import (
        assert_authoritative_for_training,
        build_active_bindings,
        resolve_qa_status,
        write_authoritative_qa_index,
    )
    from ccpc500_training_token_utils import (
        MAX_LENGTH,
        MODEL,
        REVISION,
        train_token_count,
    )

CANONICAL_BUILDER_DIR = Path(__file__).resolve().parent
DETECTOR_CONTRACT = (
    Path("/home/ubuntu/hirundo-research")
    / "probe/rsch76-ccpc500-supervision-qa-contract-v1.1"
)
BUILDER_MIRROR_FILES = (
    "build_ccpc500_training_data.py",
    "ccpc500_training_token_utils.py",
    "ccpc500_training_provenance.py",
)

ROOT = Path("/home/ubuntu/hirundo-research")
SOURCE = Path(
    "/home/ubuntu/worktrees/hr-rsch76-unlearning-data/probe/rsch76-ccpc500-behavior-source-freeze-v1"
)
ROUTING = Path(
    "/home/ubuntu/worktrees/hr-rsch76-unlearning-data/probe/rsch76-ccpc500-base-routing-v1"
)
CONTRACT = ROOT / "probe/rsch76-ccpc500-training-contract-v1"
OUTPUT = ROOT / "probe/rsch76-ccpc500-training-data-v1"
ARCHIVE_ROOT = Path("/home/ubuntu/artifact-archive/20260829")
ACTIVE_EXECUTION_ID = "execution_attempt_4"
ACTIVE_RAW_SNAPSHOT_SHA256 = (
    "dc2bc15f49ed599246b7fec6cc66159a80ad4188ac369b9873d7e39856cba054"
)
ACTIVE_ACCEPTED_VALID_ID_HASH_SHA256 = (
    "cd51a8773997a5c7fb8c196f165ba768822ada2363b01b8efb4adbed5deee32b"
)
ACTIVE_GENERATOR_CONFIG_SHA256 = (
    "4a6322828ed436e9015ed7ea90684ce7d33cb53da8ea205d24396880787205c3"
)
ACTIVE_DETECTOR_BUNDLE_SHA256 = (
    "be7013e3cd02521e33c115e6d3fed1d20c22f9ebef2d461de757fe2b1c7ceacc"
)


def now() -> str:
    """Return a UTC ISO timestamp."""
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    """Return a file SHA-256 digest."""
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON mapping."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL without printing data rows."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    """Write stable formatted JSON."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write stable JSONL."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def verify_sums(directory: Path) -> None:
    """Verify a frozen SHA256SUMS inventory."""
    for line in (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        path = directory / name.lstrip("*")
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"hash failure: {path}")


def tokenizer() -> Any:
    """Load the exact frozen Qwen tokenizer."""
    try:
        from analysis.ccpc_bench_prereg.ccpc500_training_token_utils import (
            load_frozen_tokenizer,
        )
    except ImportError:
        from ccpc500_training_token_utils import load_frozen_tokenizer

    return load_frozen_tokenizer()


def materialize_builder_mirror(destination: Path = OUTPUT) -> dict[str, str]:
    """Copy canonical builder modules to the probe mirror with hash verification.

    Args:
        destination: Probe artifact root that receives mirrored builder files.

    Returns:
        Mapping of mirrored filenames to SHA-256 digests.
    """
    destination.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for name in BUILDER_MIRROR_FILES:
        source = CANONICAL_BUILDER_DIR / name
        target = destination / name
        shutil.copy2(source, target)
        digests[name] = digest(target)
    return digests


def _mechanical_row_qa() -> Any:
    """Import the frozen v1.1 mechanical QA detector."""
    if str(DETECTOR_CONTRACT) not in sys.path:
        sys.path.insert(0, str(DETECTOR_CONTRACT))
    from corrected_detectors import mechanical_row_qa

    return mechanical_row_qa


def active_qa_bindings(qa_artifact_path: str) -> dict[str, Any]:
    """Return authoritative QA bindings for the active execution.

    Args:
        qa_artifact_path: Relative path to the authoritative QA artifact.

    Returns:
        Binding record marked ``AUTHORITATIVE``.
    """
    return build_active_bindings(
        execution_id=ACTIVE_EXECUTION_ID,
        raw_snapshot_sha256=ACTIVE_RAW_SNAPSHOT_SHA256,
        accepted_valid_id_hash_sha256=ACTIVE_ACCEPTED_VALID_ID_HASH_SHA256,
        generator_config_sha256=ACTIVE_GENERATOR_CONFIG_SHA256,
        detector_bundle_sha256=ACTIVE_DETECTOR_BUNDLE_SHA256,
        qa_artifact_path=qa_artifact_path,
    )


def qa_report_status(path: Path, active_bindings: dict[str, Any]) -> str:
    """Resolve whether a QA report is authoritative for the active execution.

    Args:
        path: Candidate QA report path.
        active_bindings: Active execution bindings.

    Returns:
        ``AUTHORITATIVE`` or ``HISTORICAL_STALE``.
    """
    if not path.is_file():
        return "HISTORICAL_STALE"
    payload = read_json(path)
    candidate = payload.get("bindings", {})
    if not candidate:
        return "HISTORICAL_STALE"
    candidate = dict(candidate)
    candidate["qa_artifact_path"] = str(path.relative_to(OUTPUT))
    return resolve_qa_status(candidate, active_bindings)


def prompt_text(tokenizer_value: Any, question: str) -> str:
    """Render the exact Base-generation prompt."""
    return str(
        tokenizer_value.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def source_index() -> dict[int, dict[str, Any]]:
    """Map behavioral source IDs to their source-only fields."""
    rows = read_jsonl(SOURCE / "train_source.jsonl") + read_jsonl(
        SOURCE / "dev_source.jsonl"
    )
    result = {int(row["pool_row_id"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("duplicate behavioral source IDs")
    return result


def normal_index() -> dict[int, dict[str, Any]]:
    """Map normal source IDs to their source rows."""
    rows = read_jsonl(SOURCE / "normal_candidate_pool.jsonl")
    return {int(row["pool_row_id"]): row for row in rows}


def prepare(_: argparse.Namespace) -> None:
    """Create and verify an immutable execution root before model calls."""
    if OUTPUT.exists():
        raise RuntimeError(f"refusing existing artifact: {OUTPUT}")
    for path in (SOURCE, ROUTING, CONTRACT):
        verify_sums(path)
    behavioral = read_jsonl(CONTRACT / "behavioral_train_eligibility.jsonl")
    normal_train = read_jsonl(CONTRACT / "normal_train_membership.jsonl")
    normal_dev = read_jsonl(CONTRACT / "normal_dev_membership.jsonl")
    counts = Counter(row["training_role"] for row in behavioral)
    if (
        counts != {"biased_ce": 4856, "anchor_forward_kl": 420}
        or len(normal_train) != 1871
        or len(normal_dev) != 204
    ):
        raise RuntimeError("frozen membership counts differ")
    source = source_index()
    normal = normal_index()
    if any(int(row["pool_row_id"]) not in source for row in behavioral) or any(
        int(row["pool_row_id"]) not in normal for row in normal_train + normal_dev
    ):
        raise RuntimeError("membership source lookup failed")
    OUTPUT.mkdir(parents=True)
    shutil.copy2(
        CONTRACT / "supervision_generator_config.json",
        OUTPUT / "supervision_generator_config.json",
    )
    shutil.copy2(
        CONTRACT / "supervision_generator_template.txt",
        OUTPUT / "supervision_generator_template.txt",
    )
    materialize_builder_mirror(OUTPUT)
    biased = [
        dict(
            row,
            question=source[int(row["pool_row_id"])]["question"],
            request_form=source[int(row["pool_row_id"])]["request_form"],
        )
        for row in behavioral
        if row["training_role"] == "biased_ce"
    ]
    anchors = [
        dict(
            row,
            question=source[int(row["pool_row_id"])]["question"],
            request_form=source[int(row["pool_row_id"])]["request_form"],
        )
        for row in behavioral
        if row["training_role"] == "anchor_forward_kl"
    ]
    write_jsonl(OUTPUT / "biased_membership.jsonl", biased)
    write_jsonl(OUTPUT / "anchor_membership.jsonl", anchors)
    for name, membership in (
        ("normal_train", normal_train),
        ("normal_dev", normal_dev),
    ):
        write_jsonl(
            OUTPUT / f"{name}_membership.jsonl",
            [
                dict(row, question=normal[int(row["pool_row_id"])]["question"])
                for row in membership
            ],
        )
    write_json(
        OUTPUT / "normal_target_generation_config.json",
        {
            "model": f"{MODEL}@{REVISION}",
            "engine": "vllm",
            "thinking_enabled": False,
            "decoding": "greedy",
            "temperature": 0,
            "top_p": 1,
            "top_k": 0,
            "repetition_penalty": 1.10,
            "max_model_len": MAX_LENGTH,
            "per_row_max_new_tokens": "2048 minus exact generation-prompt tokens and exact assistant-closing-token count",
            "zero_truncation": True,
        },
    )
    write_json(
        OUTPUT / "MANIFEST_DRAFT.json",
        {
            "status": "PREPARED_NO_GENERATION",
            "created_at": now(),
            "source_manifest_sha256": digest(SOURCE / "manifest.json"),
            "routing_manifest_sha256": digest(ROUTING / "manifest.json"),
            "contract_manifest_sha256": digest(
                CONTRACT / "final_training_design_manifest.json"
            ),
            "contract_clarifications_after_pre_generation_block": {
                "teacher_input": "question + request_form; frozen hashed generator artifact takes precedence",
                "normal_targets": "separate deterministic Base contract with per-row tokenizer-derived safe continuation budgets under max_length=2048",
                "unchanged": [
                    "source membership",
                    "routing",
                    "domain weights",
                    "cluster sampling",
                    "stopping rule",
                    "final-evaluation roles",
                ],
            },
        },
    )


def teacher_attempt(
    row: dict[str, Any], config: dict[str, Any], template: str
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Call the frozen Azure teacher with fixed retries and record each attempt."""
    from openai import AzureOpenAI

    required = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_DEPLOYMENT",
    ]
    if any(not os.environ.get(name) for name in required):
        raise RuntimeError("Azure environment variables are unavailable")
    if (
        os.environ["AZURE_OPENAI_DEPLOYMENT"] != config["deployment"]
        or os.environ["AZURE_OPENAI_API_VERSION"] != config["api_version"]
    ):
        raise RuntimeError(
            "Azure runtime differs from frozen deployment or API version"
        )
    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=config["api_version"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        max_retries=0,
        timeout=config["retry"]["timeout_seconds"],
    )
    payload = json.dumps(
        {"question": row["question"], "request_form": row["request_form"]},
        ensure_ascii=False,
        sort_keys=True,
    )
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, config["retry"]["max_attempts"] + 1):
        started = now()
        try:
            response = client.chat.completions.create(
                model=config["deployment"],
                messages=[
                    {"role": "system", "content": template},
                    {"role": "user", "content": payload},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=config["max_completion_tokens"],
                reasoning_effort=config["reasoning_effort"],
            )
            raw = response.choices[0].message.content or ""
            parsed = json.loads(raw)
            answer = (
                parsed.get("answer")
                if isinstance(parsed, dict) and set(parsed) == {"answer"}
                else None
            )
            valid = isinstance(answer, str) and bool(answer.strip())
            record = {
                "pool_row_id": row["pool_row_id"],
                "attempt": attempt,
                "started_at": started,
                "completed_at": now(),
                "request_settings": {
                    "deployment": config["deployment"],
                    "api_version": config["api_version"],
                    "reasoning_effort": config["reasoning_effort"],
                    "max_completion_tokens": config["max_completion_tokens"],
                    "temperature": "omitted",
                },
                "provider_response": response.model_dump(mode="json"),
                "raw_structured_response": raw,
                "parse_valid": valid,
                "failure_reason": None if valid else "schema_or_empty_answer",
            }
            attempts.append(record)
            if valid:
                return attempts, {
                    "answer": answer.strip(),
                    "attempt": attempt,
                    "provider": record["provider_response"],
                }
        except Exception as exc:
            attempts.append(
                {
                    "pool_row_id": row["pool_row_id"],
                    "attempt": attempt,
                    "started_at": started,
                    "completed_at": now(),
                    "request_settings": {
                        "deployment": config["deployment"],
                        "api_version": config["api_version"],
                        "reasoning_effort": config["reasoning_effort"],
                        "max_completion_tokens": config["max_completion_tokens"],
                        "temperature": "omitted",
                    },
                    "provider_response": None,
                    "raw_structured_response": None,
                    "parse_valid": False,
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
        if attempt < config["retry"]["max_attempts"]:
            time.sleep(
                min(
                    config["retry"]["initial_backoff_seconds"] * 2 ** (attempt - 1),
                    config["retry"]["max_backoff_seconds"],
                )
            )
    return attempts, None


def generate_biased(args: argparse.Namespace) -> None:
    """Generate each unfinished frozen biased identity once under retry semantics."""
    config = read_json(OUTPUT / "supervision_generator_config.json")
    template = (OUTPUT / "supervision_generator_template.txt").read_text(
        encoding="utf-8"
    )
    rows = read_jsonl(OUTPUT / "biased_membership.jsonl")
    attempts_path = OUTPUT / "biased_generation_attempts.jsonl"
    prior = read_jsonl(attempts_path) if attempts_path.exists() else []
    completed = {int(row["pool_row_id"]) for row in prior if row["parse_valid"]}
    terminal = {
        int(row["pool_row_id"])
        for row in prior
        if row["attempt"] == config["retry"]["max_attempts"] and not row["parse_valid"]
    }
    pending = [
        row for row in rows if int(row["pool_row_id"]) not in completed | terminal
    ]
    with (
        attempts_path.open("a", encoding="utf-8") as handle,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        futures = {
            pool.submit(teacher_attempt, row, config, template): row for row in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            attempts, _ = future.result()
            for item in attempts:
                handle.write(
                    json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                )
            handle.flush()
            if index % 25 == 0 or index == len(pending):
                print(f"biased generation completed {index}/{len(pending)}", flush=True)


def qa_biased(_: argparse.Namespace) -> None:
    """Apply frozen v1.1 mechanical QA to accepted biased attempts."""
    rows = {
        int(row["pool_row_id"]): row
        for row in read_jsonl(OUTPUT / "biased_membership.jsonl")
    }
    attempts = read_jsonl(OUTPUT / "biased_generation_attempts.jsonl")
    winners: dict[int, dict[str, Any]] = {}
    for item in attempts:
        if item["parse_valid"] and int(item["pool_row_id"]) not in winners:
            winners[int(item["pool_row_id"])] = item
    token = tokenizer()
    mechanical_row_qa = _mechanical_row_qa()
    ledgers: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for row_id, row in sorted(rows.items()):
        item = winners.get(row_id)
        reasons: list[str] = []
        answer = ""
        rendered_total_tokens: int | None = None
        if item is None:
            reasons.append("terminal_generation_failure")
        else:
            answer = json.loads(item["raw_structured_response"])["answer"].strip()
            rendered_total_tokens = train_token_count(token, row["question"], answer)
            verdict = mechanical_row_qa(
                {
                    "pool_row_id": row_id,
                    "request_form": row["request_form"],
                    "desired_answer": answer,
                    "rendered_total_tokens": rendered_total_tokens,
                    "provider_response": item.get("provider_response"),
                }
            )
            reasons.extend(verdict["reject_reasons"])
            if verdict["status"] == "hold_requires_verification":
                reasons.extend(verdict["hold_reasons"])
        ledger = {
            "pool_row_id": row_id,
            "accepted": not reasons,
            "rejection_reasons": reasons,
            "rendered_total_tokens": rendered_total_tokens,
        }
        ledgers.append(ledger)
        if not reasons and item is not None:
            accepted.append(
                {
                    "pool_row_id": row_id,
                    "cluster_id": row["cluster_id"],
                    "question": row["question"],
                    "request_form": row["request_form"],
                    "desired_answer": answer,
                    "generation_attempt": item["attempt"],
                    "rendered_total_tokens": rendered_total_tokens,
                }
            )
    write_jsonl(OUTPUT / "biased_qa_ledger.jsonl", ledgers)
    write_jsonl(OUTPUT / "biased_supervision.jsonl", accepted)


def normal_cap(token: Any, question: str) -> tuple[str, int, int]:
    """Derive a safe continuation budget including chat-template assistant closure."""
    prompt = prompt_text(token, question)
    prompt_count = len(token(prompt, add_special_tokens=False).input_ids)
    sentinel = "__CCPC500_ASSISTANT_SENTINEL__"
    full_count = train_token_count(token, question, sentinel)
    sentinel_count = len(token(sentinel, add_special_tokens=False).input_ids)
    closure = max(0, full_count - prompt_count - sentinel_count)
    cap = MAX_LENGTH - prompt_count - closure
    if cap <= 0:
        raise RuntimeError("normal prompt has no positive continuation budget")
    return prompt, cap, prompt_count


def generate_normal(args: argparse.Namespace) -> None:
    """Generate Base Qwen normal KL continuations using each row's safe cap."""
    from vllm import LLM, SamplingParams

    token = tokenizer()
    engine = LLM(
        model=MODEL,
        revision=REVISION,
        trust_remote_code=True,
        language_model_only=True,
        max_model_len=MAX_LENGTH,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    for split in ("train", "dev"):
        rows = read_jsonl(OUTPUT / f"normal_{split}_membership.jsonl")
        destination = OUTPUT / f"normal_{split}_targets.jsonl"
        done = (
            {int(row["pool_row_id"]) for row in read_jsonl(destination)}
            if destination.exists()
            else set()
        )
        pending = [row for row in rows if int(row["pool_row_id"]) not in done]
        with destination.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start : start + args.batch_size]
                rendered = [normal_cap(token, row["question"]) for row in batch]
                # vLLM permits one SamplingParams per prompt, preserving heterogeneous safe caps.
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
                    }
                    handle.write(
                        json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                handle.flush()
                print(
                    f"normal {split} completed {min(start + len(batch), len(pending))}/{len(pending)}",
                    flush=True,
                )


def finalize(_: argparse.Namespace) -> None:
    """Reconcile every stream, freeze hashes, and byte-copy the qualified artifact."""
    authoritative_bindings = active_qa_bindings("VALIDATION_REPORT.json")
    qa_index_entries = [
        {
            "bindings": dict(
                authoritative_bindings,
                qa_artifact_path="VALIDATION_REPORT.json",
            ),
            "status": qa_report_status(OUTPUT / "VALIDATION_REPORT.json", authoritative_bindings),
        },
        {
            "bindings": dict(
                authoritative_bindings,
                qa_artifact_path="qa_execution_v1_1/TEACHER_VERSION_GATE_REPORT.json",
            ),
            "status": qa_report_status(
                OUTPUT / "qa_execution_v1_1/TEACHER_VERSION_GATE_REPORT.json",
                authoritative_bindings,
            ),
        },
        {
            "bindings": dict(
                authoritative_bindings,
                qa_artifact_path=(
                    "qa_execution_v1_1/rerun_20260830T_independent/INDEPENDENT_QA_REPORT.json"
                ),
            ),
            "status": qa_report_status(
                OUTPUT
                / "qa_execution_v1_1/rerun_20260830T_independent/INDEPENDENT_QA_REPORT.json",
                authoritative_bindings,
            ),
        },
    ]
    write_authoritative_qa_index(
        OUTPUT / "AUTHORITATIVE_QA_INDEX.json",
        qa_index_entries,
    )
    biased = read_jsonl(OUTPUT / "biased_supervision.jsonl")
    ledger = read_jsonl(OUTPUT / "biased_qa_ledger.jsonl")
    anchors = read_jsonl(OUTPUT / "anchor_membership.jsonl")
    normal_train = read_jsonl(OUTPUT / "normal_train_targets.jsonl")
    normal_dev = read_jsonl(OUTPUT / "normal_dev_targets.jsonl")
    failures = [row for row in ledger if not row["accepted"]]
    validation = {
        "biased_expected": 4856,
        "biased_accepted": len(biased),
        "biased_terminal_failure_or_qa_rejection": len(failures),
        "anchor_rows": len(anchors),
        "anchor_clusters": len({row["cluster_id"] for row in anchors}),
        "normal_train_rows": len(normal_train),
        "normal_dev_rows": len(normal_dev),
        "normal_train_clusters": len({row["cluster_id"] for row in normal_train}),
        "normal_dev_clusters": len({row["cluster_id"] for row in normal_dev}),
        "normal_cluster_overlap": len(
            {row["cluster_id"] for row in normal_train}
            & {row["cluster_id"] for row in normal_dev}
        ),
        "normal_overlength": sum(
            not row["zero_truncation"] for row in normal_train + normal_dev
        ),
        "result": "PASS"
        if len(biased) == 4856
        and not failures
        and len(anchors) == 420
        and len(normal_train) == 1871
        and len(normal_dev) == 204
        and all(row["zero_truncation"] for row in normal_train + normal_dev)
        else "BLOCK",
    }
    validation["bindings"] = authoritative_bindings
    write_json(OUTPUT / "VALIDATION_REPORT.json", validation)
    qa_index_entries[0] = {
        "bindings": dict(
            authoritative_bindings,
            qa_artifact_path="VALIDATION_REPORT.json",
        ),
        "status": "AUTHORITATIVE",
    }
    write_authoritative_qa_index(
        OUTPUT / "AUTHORITATIVE_QA_INDEX.json",
        qa_index_entries,
    )
    (OUTPUT / "VALIDATION_REPORT.md").write_text(
        "# Validation report\n\n```json\n"
        + json.dumps(validation, indent=2, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    if validation["result"] != "PASS":
        raise RuntimeError("training-data qualification did not pass")
    assert_authoritative_for_training(
        {**authoritative_bindings, "status": "AUTHORITATIVE"}
    )
    manifest = read_json(OUTPUT / "MANIFEST_DRAFT.json")
    manifest.update(
        {
            "status": "FROZEN_READY_FOR_PRE_GRADIENT_REVIEW",
            "finalized_at": now(),
            "validation": validation,
            "generation_performed": True,
            "gradients_performed": False,
            "final_only_evaluation_performed": False,
        }
    )
    write_json(OUTPUT / "manifest.json", manifest)
    sums = [
        f"{digest(path)}  {path.relative_to(OUTPUT)}"
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (OUTPUT / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    destination = ARCHIVE_ROOT / "probe" / OUTPUT.name
    if destination.exists():
        raise RuntimeError(f"refusing existing archive destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(OUTPUT, destination)
    if digest(OUTPUT / "SHA256SUMS") != digest(destination / "SHA256SUMS"):
        raise RuntimeError("archive copy hash mismatch")
    inventory = ARCHIVE_ROOT / "SHA256SUMS"
    lines = (
        inventory.read_text(encoding="utf-8").splitlines() if inventory.exists() else []
    )
    lines.extend(
        f"{digest(path)}  {path.relative_to(ARCHIVE_ROOT)}"
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    )
    inventory.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Parse and execute exactly one construction phase."""
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(required=True)
    subs.add_parser("prepare").set_defaults(func=prepare)
    biased = subs.add_parser("generate-biased")
    biased.add_argument("--workers", type=int, default=8)
    biased.set_defaults(func=generate_biased)
    subs.add_parser("qa-biased").set_defaults(func=qa_biased)
    normal = subs.add_parser("generate-normal")
    normal.add_argument("--batch-size", type=int, default=32)
    normal.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    normal.set_defaults(func=generate_normal)
    subs.add_parser("finalize").set_defaults(func=finalize)
    mirror = subs.add_parser("materialize-builder-mirror")
    mirror.add_argument(
        "--destination",
        type=Path,
        default=OUTPUT,
        help="Probe artifact root that receives mirrored builder files.",
    )
    mirror.set_defaults(func=lambda args: print(json.dumps(materialize_builder_mirror(args.destination), indent=2)))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
