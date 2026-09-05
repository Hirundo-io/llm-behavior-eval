#!/usr/bin/env python3
"""Freeze the 4096-token teacher amendment and run bounded execution-4 canaries."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/home/ubuntu/hirundo-research")
PARENT = ROOT / "probe/rsch76-ccpc500-training-data-v1"
AMENDMENT = PARENT / "teacher_token_budget_amendment_v1"
EXECUTION = PARENT / "execution_attempt_4"
EXEC3 = PARENT / "execution_attempt_3"
SOURCE = Path(__file__).resolve().parent / "build_ccpc500_training_data.py"
EXEC3_RUNNER = EXEC3 / "run_ccpc500_execution_attempt_3.py"


def sha256(path: Path) -> str:
    """Return a SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write stable JSON."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_execution3() -> Any:
    """Load the already-recorded deterministic canary implementation."""
    sys.path.insert(0, str(EXEC3))
    spec = importlib.util.spec_from_file_location("execution3", EXEC3_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("execution-3 runner is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules["execution3"] = module
    spec.loader.exec_module(module)
    return module


def prepare(_: argparse.Namespace) -> None:
    """Create the amendment and separate execution lineage without model calls."""
    if AMENDMENT.exists() or EXECUTION.exists():
        raise RuntimeError("refusing to overwrite amendment or execution-4 lineage")
    original = json.loads((PARENT / "supervision_generator_config.json").read_text())
    amended = dict(original)
    amended["max_completion_tokens"] = 4096
    amended["status"] = "AMENDED_4096_TOKEN_CANARY_ONLY"
    AMENDMENT.mkdir()
    write_json(AMENDMENT / "supervision_generator_config_4096.json", amended)
    write_json(
        AMENDMENT / "amendment.json",
        {
            "amendment": "teacher_token_budget_amendment_v1",
            "one_field_diff": {"max_completion_tokens": {"from": 1024, "to": 4096}},
            "rationale": "Execution-3 fixed real-contract canaries showed length termination after the full 1024-token budget was consumed as reasoning with zero visible output.",
            "unchanged": [
                "deployment",
                "required_returned_model",
                "question_plus_request_form",
                "template",
                "response_format",
                "reasoning_effort_low",
                "retry_semantics",
                "fixed_population",
                "automatic_QA",
                "supervision_QA_v1_1",
            ],
            "training_envelope": "Provider max_completion_tokens=4096 includes hidden reasoning and is not the accepted target length; Qwen rendered training examples remain <=2048 with zero truncation.",
            "original_config_sha256": sha256(
                PARENT / "supervision_generator_config.json"
            ),
        },
    )
    (AMENDMENT / "SHA256SUMS").write_text(
        "\n".join(
            f"{sha256(path)}  {path.name}" for path in sorted(AMENDMENT.glob("*.json"))
        )
        + "\n"
    )
    EXECUTION.mkdir()
    for source, target in (
        (PARENT / "biased_membership.jsonl", EXECUTION / "biased_membership.jsonl"),
        (
            AMENDMENT / "supervision_generator_config_4096.json",
            EXECUTION / "supervision_generator_config.json",
        ),
        (
            PARENT / "supervision_generator_template.txt",
            EXECUTION / "supervision_generator_template.txt",
        ),
    ):
        shutil.copy2(source, target)
    write_json(
        EXECUTION / "execution_manifest.json",
        {
            "execution": "attempt_4",
            "full_generation_authorized": False,
            "canary_rule": "same five-ID and nested 20-ID deterministic selection as execution 3",
            "links": {
                "original_training_contract_sha256": sha256(
                    ROOT
                    / "probe/rsch76-ccpc500-training-contract-v1/final_training_design_manifest.json"
                ),
                "original_generator_config_sha256": sha256(
                    PARENT / "supervision_generator_config.json"
                ),
                "token_budget_amendment_sha256": sha256(AMENDMENT / "amendment.json"),
                "execution_1_snapshot_sha256": sha256(
                    PARENT / "qa_execution_v1_1/PRE_QA_RAW_SHA256SUMS"
                ),
                "execution_2_attempts_sha256": sha256(
                    PARENT / "execution_attempt_2/biased_generation_attempts.jsonl"
                ),
                "execution_3_report_sha256": sha256(EXEC3 / "CANARY_5_REPORT.json"),
                "qa_v1_sha256s": sha256(
                    ROOT / "probe/rsch76-ccpc500-supervision-qa-contract-v1/SHA256SUMS"
                ),
                "qa_v1_1_sha256s": sha256(
                    ROOT
                    / "probe/rsch76-ccpc500-supervision-qa-contract-v1.1/SHA256SUMS"
                ),
            },
        },
    )
    shutil.copy2(Path(__file__), EXECUTION / Path(__file__).name)


def canary(args: argparse.Namespace) -> None:
    """Run the exact execution-3 deterministic canary IDs under the amendment."""
    if not EXECUTION.exists():
        raise RuntimeError("run prepare first")
    report_path = EXECUTION / f"CANARY_{args.size}_REPORT.json"
    if not report_path.exists():
        module = load_execution3()
        module.PARENT = EXECUTION
        module.EXECUTION = EXECUTION
        module.SCRIPT_DIR = EXEC3
        module.canary(args)
    report = json.loads(report_path.read_text())
    records = report["records"]
    model_counts: dict[str, int] = {}
    reasoning: list[int] = []
    visible: list[int] = []
    for row in records:
        model = str(row.get("response_model"))
        model_counts[model] = model_counts.get(model, 0) + 1
        usage = row.get("usage", {})
        total = int(usage.get("completion_tokens") or 0)
        thought = int(usage.get("reasoning_tokens") or 0)
        reasoning.append(thought)
        visible.append(max(0, total - thought))
    mechanical = all(
        row.get("parse_valid")
        and row.get("response_model") == "gpt-5-2025-08-07"
        and row.get("finish_reason") != "length"
        and (row.get("message_content_length") or 0) > 0
        for row in records
    )
    report.update(
        {
            "returned_backend_distribution": model_counts,
            "reasoning_token_distribution": sorted(reasoning),
            "visible_completion_token_distribution": sorted(visible),
            "maximum_observed_total_completion_usage": max(
                (
                    int(row.get("usage", {}).get("completion_tokens") or 0)
                    for row in records
                ),
                default=0,
            ),
            "mechanical_acceptance": "PASS" if mechanical else "BLOCK",
            "result": "PASS" if mechanical else "BLOCK",
        }
    )
    write_json(report_path, report)
    if not mechanical:
        raise RuntimeError("BLOCK — 4096-TOKEN TEACHER BUDGET STILL INSUFFICIENT")


def main() -> None:
    """Run one execution-4 stage."""
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
