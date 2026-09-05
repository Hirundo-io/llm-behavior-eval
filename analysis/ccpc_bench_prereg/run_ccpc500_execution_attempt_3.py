#!/usr/bin/env python3
"""Run the response-shape-only diagnostic canaries for CCPC500 execution 3.

This program deliberately never writes model answer text.  It records response
shape before parsing so an unsuccessful parse cannot discard provider evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import build_ccpc500_training_data as generation  # noqa: E402
from execution_3_diagnostics import (  # noqa: E402
    attach_parse_diagnostic,
    capture_response_shape,
)

PARENT = Path("/home/ubuntu/hirundo-research/probe/rsch76-ccpc500-training-data-v1")
EXECUTION = PARENT / "execution_attempt_3"
CONTRACT = Path(
    "/home/ubuntu/hirundo-research/probe/rsch76-ccpc500-training-contract-v1"
)
QA_V1 = Path(
    "/home/ubuntu/hirundo-research/probe/rsch76-ccpc500-supervision-qa-contract-v1"
)
QA_V11 = Path(
    "/home/ubuntu/hirundo-research/probe/rsch76-ccpc500-supervision-qa-contract-v1.1"
)
CANARY_RULE = "sha256('ccpc500-execution-3-canary-v1:' + decimal pool_row_id), ascending digest then ID"


def now() -> str:
    """Return an ISO UTC timestamp."""
    return datetime.now(UTC).isoformat()


def sha256(path: Path) -> str:
    """Return a file digest."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write stable JSON without model answer text."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def endpoint_values(config: dict[str, Any]) -> tuple[str, str, str, str]:
    """Require frozen Azure credentials and settings without disclosing the key."""
    names = (
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_API_KEY",
    )
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing Azure environment variables: {', '.join(missing)}")
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_version = os.environ["AZURE_OPENAI_API_VERSION"]
    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    if api_version != config["api_version"] or deployment != config["deployment"]:
        raise RuntimeError("Azure deployment/API version differs from frozen config")
    return endpoint, api_version, deployment, os.environ["AZURE_OPENAI_API_KEY"]


def deterministic_canary_rows(size: int) -> list[dict[str, Any]]:
    """Select a nested deterministic prefix of frozen biased membership."""
    if size not in {5, 20}:
        raise ValueError("execution-3 canary size must be 5 or 20")
    rows = generation.read_jsonl(PARENT / "biased_membership.jsonl")
    ranked = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(
                f"ccpc500-execution-3-canary-v1:{int(row['pool_row_id'])}".encode()
            ).hexdigest(),
            int(row["pool_row_id"]),
        ),
    )
    return ranked[:size]


def prepare(_: argparse.Namespace) -> None:
    """Create an immutable execution-3 lineage without touching earlier attempts."""
    if EXECUTION.exists():
        raise RuntimeError(f"refusing existing artifact: {EXECUTION}")
    for path in (PARENT, CONTRACT, QA_V1, QA_V11):
        if not path.is_dir():
            raise RuntimeError(f"required lineage path is unavailable: {path}")
    EXECUTION.mkdir(parents=True)
    config_path = PARENT / "supervision_generator_config.json"
    template_path = PARENT / "supervision_generator_template.txt"
    five = deterministic_canary_rows(5)
    twenty = deterministic_canary_rows(20)
    write_json(
        EXECUTION / "execution_manifest.json",
        {
            "execution": "attempt_3",
            "created_at": now(),
            "disposition_of_execution_2": (
                "execution failure caused by insufficient response-shape instrumentation; "
                "supervision quality not adjudicated"
            ),
            "scientific_generation_contract_unchanged": True,
            "diagnostic_scope": "one real-contract attempt per selected ID; no corpus retry budget consumed",
            "does_not_authorize": [
                "full corpus generation",
                "training",
                "membership changes",
            ],
            "canary_selection": {
                "population": "frozen biased_ce membership",
                "rule": CANARY_RULE,
                "five_source_ids": [int(row["pool_row_id"]) for row in five],
                "twenty_source_ids": [int(row["pool_row_id"]) for row in twenty],
                "nested_prefix": True,
            },
            "links": {
                "training_contract_manifest_sha256": sha256(
                    CONTRACT / "final_training_design_manifest.json"
                ),
                "generator_config_sha256": sha256(config_path),
                "generator_template_sha256": sha256(template_path),
                "qa_v1_sha256s_sha256": sha256(QA_V1 / "SHA256SUMS"),
                "qa_v1_1_sha256s_sha256": sha256(QA_V11 / "SHA256SUMS"),
                "execution_1_failure_snapshot_sha256": sha256(
                    PARENT / "qa_execution_v1_1/PRE_QA_RAW_SHA256SUMS"
                ),
                "execution_2_manifest_sha256": sha256(
                    PARENT / "execution_attempt_2/execution_manifest.json"
                ),
                "execution_2_attempts_sha256": sha256(
                    PARENT / "execution_attempt_2/biased_generation_attempts.jsonl"
                ),
                "azure_connectivity_qualification_sha256": sha256(
                    PARENT / "execution_attempt_2/AZURE_CONNECTIVITY_QUALIFICATION.json"
                ),
            },
        },
    )
    # These copies make the mechanism reproducible while retaining the source-only
    # input outside every answer-free diagnostic report.
    for source in (Path(__file__), SCRIPT_DIR / "execution_3_diagnostics.py"):
        (EXECUTION / source.name).write_bytes(source.read_bytes())


def capture_provider_response(
    client: Any, row: dict[str, Any], config: dict[str, Any], template: str
) -> tuple[dict[str, Any], Any]:
    """Call once and return shape plus transient content for later parsing."""
    source_id = int(row["pool_row_id"])
    started = now()
    payload = json.dumps(
        {"question": row["question"], "request_form": row["request_form"]},
        ensure_ascii=False,
        sort_keys=True,
    )
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
        record = capture_response_shape(response, source_id=source_id, attempt=1)
        record["provider_request_completed"] = True
        record["response_format_json_object_accepted"] = True
        message = response.choices[0].message if response.choices else None
        content = getattr(message, "content", None)
    except Exception as exc:
        record = {
            "source_id": source_id,
            "attempt": 1,
            "provider_request_completed": False,
            "response_format_json_object_accepted": False,
            "parse_valid": False,
            "parser_exception_class": None,
            "parser_exception_message": None,
            "provider_exception": {
                "class": type(exc).__name__,
                "message": str(exc),
            },
        }
        content = None
    record["started_at"] = started
    record["completed_at"] = now()
    return record, content


def aggregate(records: list[dict[str, Any]], requested_size: int) -> dict[str, Any]:
    """Summarize only response shape, never answers."""

    def count(key: str) -> dict[str, int]:
        return dict(sorted(Counter(str(row.get(key)) for row in records).items()))

    completed = [row for row in records if row.get("provider_request_completed")]
    empty = sum(
        row.get("message_content_is_none") or row.get("message_content_length") == 0
        for row in completed
    )
    visible_nonempty = sum(
        not row.get("message_content_is_none")
        and (row.get("message_content_length") or 0) > 0
        for row in completed
    )
    diagnosis = {
        "finish_reason_length": sum(
            row.get("finish_reason") == "length" for row in completed
        ),
        "finish_reason_content_filter": sum(
            row.get("finish_reason") == "content_filter" for row in completed
        ),
        "finish_reason_stop": sum(
            row.get("finish_reason") == "stop" for row in completed
        ),
        "refusal_present": sum(
            bool(row.get("message_refusal", {}).get("present")) for row in completed
        ),
        "content_filter_metadata_present": sum(
            any(
                row.get("azure_content_filter", {}).get(name) is not None
                for name in ("prompt", "completion")
            )
            for row in completed
        ),
        "reasoning_tokens_positive_zero_visible_completion": sum(
            (row.get("usage", {}).get("reasoning_tokens") or 0) > 0
            and (row.get("usage", {}).get("completion_tokens") or 0) == 0
            and (row.get("message_content_length") or 0) == 0
            for row in completed
        ),
        "alternative_visible_output_present": sum(
            any(
                bool(value.get("present"))
                for value in row.get("alternative_visible_output_fields", {}).values()
            )
            for row in completed
        ),
        "malformed_nonempty_json": sum(
            not row.get("parse_valid")
            and (row.get("message_content_length") or 0) > 0
            and bool(row.get("parser_exception_class"))
            for row in completed
        ),
    }
    result = (
        "PASS"
        if len(records) == requested_size
        and all(row.get("parse_valid") for row in records)
        else "BLOCK"
    )
    return {
        "requested_attempts": requested_size,
        "provider_requests_completed": len(completed),
        "parse_valid": sum(bool(row.get("parse_valid")) for row in records),
        "parse_invalid": sum(not row.get("parse_valid") for row in records),
        "empty_visible_content": empty,
        "nonempty_visible_content": visible_nonempty,
        "finish_reason_distribution": count("finish_reason"),
        "parser_exception_distribution": count("parser_exception_class"),
        "diagnostic_state_counts": diagnosis,
        "result": result,
        "next_action": (
            "Run the 20-ID bounded canary with this unchanged backend."
            if result == "PASS" and requested_size == 5
            else "EXECUTION-3 FIXED-CONTRACT CANARY PASSED — READY FOR FULL GENERATION"
            if result == "PASS"
            else "BLOCK — REAL-CONTRACT AZURE RESPONSE SHAPE DIAGNOSED"
        ),
    }


def canary(args: argparse.Namespace) -> None:
    """Run a five- or twenty-ID one-attempt real-contract diagnostic canary."""
    if not EXECUTION.is_dir():
        raise RuntimeError("prepare execution attempt 3 before running a canary")
    if args.size == 20:
        first = generation.read_json(EXECUTION / "CANARY_5_REPORT.json")
        if first["result"] != "PASS":
            raise RuntimeError(
                "five-ID canary did not pass; 20-ID expansion is not authorized"
            )
    report_path = EXECUTION / f"CANARY_{args.size}_REPORT.json"
    if report_path.exists():
        raise RuntimeError(
            f"refusing to overwrite existing canary report: {report_path}"
        )
    capture_path = EXECUTION / f"CANARY_{args.size}_RESPONSE_SHAPE_CAPTURES.jsonl"
    parse_path = EXECUTION / f"CANARY_{args.size}_PARSE_OUTCOMES.jsonl"
    if capture_path.exists() or parse_path.exists():
        raise RuntimeError(
            "refusing an existing partial canary capture or parse record"
        )
    from openai import AzureOpenAI

    config = generation.read_json(PARENT / "supervision_generator_config.json")
    template = (PARENT / "supervision_generator_template.txt").read_text(
        encoding="utf-8"
    )
    endpoint, api_version, deployment, key = endpoint_values(config)
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        api_key=key,
        max_retries=0,
        timeout=config["retry"]["timeout_seconds"],
    )
    records: list[dict[str, Any]] = []
    with (
        capture_path.open("x", encoding="utf-8") as captures,
        parse_path.open("x", encoding="utf-8") as parse_outcomes,
    ):
        for row in deterministic_canary_rows(args.size):
            record, content = capture_provider_response(client, row, config, template)
            # This durable, answer-free capture precedes any JSON parsing.
            captures.write(json.dumps(record, sort_keys=True) + "\n")
            captures.flush()
            if record["provider_request_completed"]:
                _ = attach_parse_diagnostic(record, content)
            parse_outcomes.write(
                json.dumps(
                    {
                        "source_id": record["source_id"],
                        "attempt": record["attempt"],
                        "parse_valid": record["parse_valid"],
                        "parser_exception_class": record["parser_exception_class"],
                        "parser_exception_message": record["parser_exception_message"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            parse_outcomes.flush()
            records.append(record)
    report = aggregate(records, args.size)
    report.update(
        {
            "created_at": now(),
            "response_text_logged": False,
            "fixed_contract": {
                "deployment": config["deployment"],
                "api_version": config["api_version"],
                "reasoning_effort": config["reasoning_effort"],
                "response_format": "json_object",
                "max_completion_tokens": config["max_completion_tokens"],
                "attempts_per_id": 1,
                "corpus_retry_budget_consumed": False,
            },
            "records": records,
        }
    )
    write_json(report_path, report)
    if report["result"] != "PASS":
        raise RuntimeError("BLOCK — REAL-CONTRACT AZURE RESPONSE SHAPE DIAGNOSED")


def main() -> None:
    """Run one explicit execution-3 lifecycle stage."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(required=True)
    commands.add_parser("prepare").set_defaults(func=prepare)
    canary_parser = commands.add_parser("canary")
    canary_parser.add_argument("--size", type=int, choices=(5, 20), required=True)
    canary_parser.set_defaults(func=canary)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
