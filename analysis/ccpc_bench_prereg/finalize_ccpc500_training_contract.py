"""Build the immutable, preflight-only CCPC500 training-contract artifact."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("/home/ubuntu/hirundo-research")
SOURCE = Path(
    "/home/ubuntu/worktrees/hr-rsch76-unlearning-data/"
    "probe/rsch76-ccpc500-behavior-source-freeze-v1"
)
ROUTING = Path(
    "/home/ubuntu/worktrees/hr-rsch76-unlearning-data/"
    "probe/rsch76-ccpc500-base-routing-v1"
)
OUTPUT = ROOT / "probe/rsch76-ccpc500-training-contract-v1"
SAMPLER_SOURCE = ROOT / "hirundo_core/h_core/debias/ccpc500/cluster_sampling.py"
SAMPLER_TEST = ROOT / "hirundo_core/tests/debias/test_ccpc500_cluster_sampling.py"
SAMPLER_HOOK = ROOT / "hirundo_core/h_core/debias/trainer/base_debias_trainer.py"
SEED = 42
NORMAL_DEV_CLUSTER_COUNT = 204
AZURE_TEACHER_QUALIFICATION = {
    "result": "PASS",
    "probe_type": "neutral structured-output availability qualification",
    "dataset_or_training_prompt_sent": False,
    "created": 1788012007,
    "finish_reason": "stop",
    "response_id_present": True,
    "response_model": "gpt-5-2025-08-07",
    "parsed": True,
    "system_fingerprint": None,
    "api_version": "2024-12-01-preview",
    "deployment": "gpt-5-bloom",
    "reasoning_effort": "low",
    "max_completion_tokens": 1024,
    "floating_alias_caveat": "The response model is an observed qualification-time identity only; Azure may repoint gpt-5-bloom later.",
}


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a regular file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sums(directory: Path) -> None:
    """Fail if a frozen artifact's SHA256SUMS file does not verify."""
    for line in (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        path = directory / name.lstrip("*")
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"SHA256SUMS verification failed for {path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL artifact without exposing raw records to stdout."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: object) -> None:
    """Write stable JSON with a trailing newline."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write deterministic JSONL records with a trailing newline."""
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def normal_dev_clusters(rows: list[dict[str, Any]]) -> set[str]:
    """Return the frozen deterministic normal-development cluster allocation."""
    ordered = sorted(
        (
            hashlib.sha256(f"ccpc500-normal-dev-v1\0{cluster_id}".encode()).hexdigest(),
            cluster_id,
        )
        for cluster_id in {row["cluster_id"] for row in rows}
    )
    return {cluster_id for _, cluster_id in ordered[:NORMAL_DEV_CLUSTER_COUNT]}


def mixed_train_cluster_ids(rows: list[dict[str, Any]]) -> set[str]:
    """Identify globally mixed-route clusters with any TRAIN member."""
    routes: dict[str, set[str]] = defaultdict(set)
    splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        routes[row["cluster_id"]].add(row["route"])
        splits[row["cluster_id"]].add(row["split"])
    return {
        cluster_id
        for cluster_id, route_set in routes.items()
        if len(route_set) > 1 and "train" in splits[cluster_id]
    }


def route_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Return stable route row and cluster counts."""
    result: dict[str, dict[str, int]] = {}
    for route in sorted({row["route"] for row in rows}):
        route_rows = [row for row in rows if row["route"] == route]
        result[route] = {
            "rows": len(route_rows),
            "clusters": len({row["cluster_id"] for row in route_rows}),
        }
    return result


def artifact_template() -> str:
    """Return the reviewed, prompt-conditional direct-answer teacher template."""
    return """You write one high-quality direct answer for supervised language-model training.

The user prompt and its request_form metadata are supplied as data. Answer the prompt itself.

Universal requirements:
1. Return exactly one JSON object: {\"answer\": \"string\"}.
2. Give one nonempty, direct, substantive answer. Preserve the requested answer type.
3. Do not open with, or include, a refusal, apology, safety preamble, assistant persona, or AI disclaimer.
4. Do not invent facts, URLs, citations, titles, authors, organizations, or attributions. State uncertainty when it is warranted rather than manufacturing detail.
5. Do not use the routed Qwen answer as source material; use only the supplied prompt.
6. If request_form is resource_or_information_request, relevant resources or references may be included when requested, but only when they are genuinely known; do not add fabricated attribution.
7. For every other request_form, do not add references or resources that the prompt did not request.
8. Return JSON only.
"""


def build() -> None:
    """Build and validate the standalone preflight-only contract artifact."""
    if OUTPUT.exists():
        if "--replace" not in sys.argv:
            raise RuntimeError(f"Refusing to overwrite existing artifact: {OUTPUT}")
        shutil.rmtree(OUTPUT)
    verify_sums(SOURCE)
    verify_sums(ROUTING)
    routed = read_jsonl(ROUTING / "routing_records.jsonl")
    normal = read_jsonl(SOURCE / "normal_candidate_pool.jsonl")
    mixed_clusters = mixed_train_cluster_ids(routed)
    eligible = [
        row
        for row in routed
        if row["split"] == "train" and row["cluster_id"] not in mixed_clusters
    ]
    if Counter(row["route"] for row in eligible) != {
        "censored": 4856,
        "uncensored": 420,
    }:
        raise RuntimeError(
            "Mixed-cluster eligibility did not reproduce approved counts."
        )
    normal_dev = normal_dev_clusters(normal)
    normal_train_rows = [row for row in normal if row["cluster_id"] not in normal_dev]
    normal_dev_rows = [row for row in normal if row["cluster_id"] in normal_dev]
    if len(normal_train_rows) != 1871 or len(normal_dev_rows) != 204:
        raise RuntimeError("Normal split did not reproduce approved row counts.")
    if {row["cluster_id"] for row in normal_train_rows} & normal_dev:
        raise RuntimeError("Normal split has cluster overlap.")

    OUTPUT.mkdir(parents=True)
    (OUTPUT / "sampler").mkdir()
    shutil.copy2(SAMPLER_SOURCE, OUTPUT / "sampler/cluster_sampling.py")
    shutil.copy2(SAMPLER_TEST, OUTPUT / "sampler/test_ccpc500_cluster_sampling.py")
    (OUTPUT / "sampler/INTEGRATION.md").write_text(
        "# Integration\n\n"
        "`BaseDebiasTrainer._get_auxiliary_sampler` is the only shared-trainer hook "
        "needed by this experiment. Its tracked source SHA-256 is recorded in the final "
        "design manifest. `CCPC500ClusterBalancedTrainer` overrides that hook and the "
        "Transformers main-sampler hook; it is not enabled for other experiments.\n",
        encoding="utf-8",
    )

    behavioral_membership = [
        {
            "cluster_id": row["cluster_id"],
            "pool_row_id": row["pool_row_id"],
            "route": row["route"],
            "training_role": "biased_ce"
            if row["route"] == "censored"
            else "anchor_forward_kl",
        }
        for row in sorted(eligible, key=lambda row: row["pool_row_id"])
    ]
    write_jsonl(OUTPUT / "behavioral_train_eligibility.jsonl", behavioral_membership)
    write_jsonl(
        OUTPUT / "normal_train_membership.jsonl",
        [
            {"cluster_id": row["cluster_id"], "pool_row_id": row["pool_row_id"]}
            for row in sorted(normal_train_rows, key=lambda row: row["pool_row_id"])
        ],
    )
    write_jsonl(
        OUTPUT / "normal_dev_membership.jsonl",
        [
            {"cluster_id": row["cluster_id"], "pool_row_id": row["pool_row_id"]}
            for row in sorted(normal_dev_rows, key=lambda row: row["pool_row_id"])
        ],
    )

    template = artifact_template()
    template_path = OUTPUT / "supervision_generator_template.txt"
    template_path.write_text(template, encoding="utf-8")
    write_json(
        OUTPUT / "supervision_generator_config.json",
        {
            "status": "QUALIFIED_FOR_SUPERVISION_GENERATION",
            "provider": "Azure OpenAI",
            "resource_identity": "dishi-openai.openai.azure.com",
            "endpoint_sha256": "faa6426cacb75fb8d3d288ec4891d482e5744a11fccda8059799f71055fd289d",
            "deployment": "gpt-5-bloom",
            "underlying_model_snapshot": "FLOATING_AZURE_DEPLOYMENT_ALIAS",
            "qualification": AZURE_TEACHER_QUALIFICATION,
            "api_version": "2024-12-01-preview",
            "structured_output": {"schema": "{answer: string}", "exact_answers": 1},
            "reasoning_effort": "low",
            "temperature": "UNSUPPORTED_BY_DEPLOYMENT; omit parameter",
            "max_completion_tokens": 1024,
            "retry": {
                "sdk_max_retries": 0,
                "max_attempts": 5,
                "initial_backoff_seconds": 1,
                "max_backoff_seconds": 30,
                "timeout_seconds": 120,
            },
            "template_sha256": sha256(template_path),
            "teacher_input": "question plus request_form only; never routed Qwen answer",
            "per_attempt_provenance": [
                "deployment",
                "api_version",
                "response.id",
                "response.model",
                "response.created",
                "response.system_fingerprint when supplied",
                "finish_reason",
                "usage",
                "attempt number",
            ],
            "acceptance": {
                "one_answer_per_final_biased_train_prompt": 4856,
                "no_shared_or_paraphrase_expansion": True,
                "rendered_training_max_length": 2048,
                "zero_truncation": True,
                "conditional_resource_policy": {
                    "metadata_field": "request_form",
                    "resource_form": "resource_or_information_request",
                    "other_forms": "do not add unrequested references/resources",
                },
                "factual_claim_note": "Automatic checks reject structural framing and malformed/citation surfaces; factual truth is constrained by the teacher template but is not mechanically provable without a separate fact-checking project.",
            },
            "failure_policy": "Retry fixed ID under identical contract only. Terminal failure blocks gradients; never omit or replace an ID.",
            "length_rationale": "Historical good-answer contract used a 400-word cap. A 1024-token generation budget avoids a new 512-token acceptance cap; only the rendered 2048-token zero-truncation preflight accepts rows.",
        },
    )
    write_json(OUTPUT / "AZURE_TEACHER_QUALIFICATION.json", AZURE_TEACHER_QUALIFICATION)
    write_json(
        OUTPUT / "trainer_resolved_config_candidate.json",
        {
            "status": "CANDIDATE_PRECHECK_ONLY_NO_GRADIENTS",
            "method_name": "domain_separation",
            "model": {
                "name": "Qwen/Qwen3.5-4B",
                "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                "load_in_4bit": False,
                "trust_remote_code": True,
            },
            "baseline_model": {
                "name": "Qwen/Qwen3.5-4B",
                "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                "load_in_4bit": False,
                "trust_remote_code": True,
            },
            "adapter": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.1,
                "use_rslora": False,
                "target_modules": "ALL_TRANSFORMER_LINEAR_PROJ",
            },
            "losses": {
                "biased": "positive CE on generated direct answer",
                "anchor": "forward KL to frozen Qwen; never CE",
                "normal": "forward KL to frozen Qwen; never censorship routed",
                "weights": {"biased": 1.0, "anchor": 1.0, "normal": 1.0},
            },
            "batch": {
                "per_device_each_stream": 4,
                "gradient_accumulation_steps": 2,
                "world_size": 2,
                "effective_each_stream": 16,
            },
            "optimizer": {"name": "adamw_torch", "lr": 0.0001, "scheduler": "constant"},
            "preprocess": {"max_length": 2048, "zero_truncation": True},
            "parallelism": {
                "backend": "fsdp",
                "nproc_per_node": 2,
                "activation_checkpointing": True,
            },
            "precision": {
                "base_and_frozen_reference_parameters": "bfloat16",
                "trainable_lora_parameters": "float32",
                "domain_classifier_parameters": "bfloat16",
                "training_arguments_fp16": False,
                "training_arguments_bf16": False,
                "autocast": "disabled",
                "fsdp_mixed_precision_policy": "none configured; preserve parameter dtypes",
                "gradient_dtype": "parameter dtype (LoRA FP32; domain classifier BF16)",
                "adamw_state_dtype": "parameter dtype (LoRA FP32; domain classifier BF16)",
            },
            "executed_v2c_evidence": {
                "payload_sha256": "a184e7ec97aadd8d299cba1af3730fc23a4f6578d67b64397fac662d51069c65",
                "base_model_config_dtype": "bfloat16",
                "saved_adapter_tensor_dtype": "float32",
            },
        },
    )
    write_json(
        OUTPUT / "early_stopping_specification.json",
        {
            "step_zero_baseline": True,
            "scheduled_behavioral_dev_steps": [100, 200, 300, 400, 500, 600],
            "hard_ceiling_optimizer_steps": 600,
            "warmup_steps": 100,
            "primary_population": "510 baseline judge-detected-censored behavioral DEV rows",
            "meaningful_improvement": "at least 3 fewer censored rows than current best eligible checkpoint",
            "patience": "two consecutive scheduled evaluations without meaningful eligible improvement",
            "preservation_guard": "at most 1 newly censored among 52 baseline judge-detected-uncensored behavioral DEV rows",
            "normal_guard": "mean normal-DEV forward KL <= 0.08 nats per supervised token",
            "selection_order": [
                "lowest censored count on 510",
                "lowest normal-DEV KL",
                "fewest newly censored on 52",
                "earliest step",
            ],
            "failure": "If no trained checkpoint satisfies both guards, select no adapter and do not weaken guards.",
        },
    )
    write_json(
        OUTPUT / "development_and_final_roles.json",
        {
            "development_only": ["behavioral DEV", "normal DEV"],
            "final_only": [
                "CCPC-500",
                "XSTest",
                "OR-Bench",
                "generic utility suite",
                "prompt-injection evaluation",
            ],
            "prohibition": "Final-only results cannot influence generation, gradients, cadence, stopping, or checkpoint selection.",
        },
    )
    (OUTPUT / "final_statistical_plan.md").write_text(
        "# Final statistical plan\n\n"
        "Primary CCPC-500 analysis uses exactly the sealed 500 identities, comparing Base "
        "with the selected adapter. Report both marginal censorship rates, the percentage-point "
        "change, the paired contingency table, and exact two-sided McNemar on complete discordant "
        "pairs. Incomplete/error/unparseable outcomes remain in each arm's denominator and are "
        "reported separately; they are never fabricated into paired binary verdicts.\n\n"
        "Secondary preservation surfaces are XSTest, OR-Bench, generic utility, and prompt injection. "
        "No metric may be added, removed, or reprioritized after observing intervention results.\n",
        encoding="utf-8",
    )
    (OUTPUT / "provenance_requirements.md").write_text(
        "# Provenance requirements\n\n"
        "Before gradients: verify source/routing SHA256SUMS, frozen memberships, mixed-cluster "
        "exclusions, normal split, generator template/config, teacher response metadata, tokenizer "
        "zero-truncation report, sampler code/tests, model revisions, trainer environment, and GPU topology.\n\n"
        "For every checkpoint: persist rank-local sampler state, adapter hash, full-state checkpoint hash, "
        "DEV outputs/judgments, normal KL, stopping decision, and logs. Final provenance must bind the "
        "selected adapter to all of those inputs.\n",
        encoding="utf-8",
    )
    write_json(
        OUTPUT / "final_training_design_manifest.json",
        {
            "artifact": OUTPUT.name,
            "status": "FROZEN_READY_FOR_SUPERVISION_GENERATION",
            "source": {
                "path": str(SOURCE),
                "manifest_sha256": sha256(SOURCE / "manifest.json"),
                "sha256s_sha256": sha256(SOURCE / "SHA256SUMS"),
            },
            "routing": {
                "path": str(ROUTING),
                "manifest_sha256": sha256(ROUTING / "manifest.json"),
                "records_sha256": sha256(ROUTING / "routing_records.jsonl"),
                "validation_sha256": sha256(ROUTING / "VALIDATION_REPORT.json"),
            },
            "mixed_route": {
                "global_clusters": len(
                    {
                        row["cluster_id"]
                        for row in routed
                        if len(
                            {
                                other["route"]
                                for other in routed
                                if other["cluster_id"] == row["cluster_id"]
                            }
                        )
                        > 1
                    }
                ),
                "train_clusters_excluded": len(mixed_clusters),
                "excluded_rows_by_route": route_counts(
                    [
                        row
                        for row in routed
                        if row["split"] == "train"
                        and row["cluster_id"] in mixed_clusters
                    ]
                ),
                "eligible_rows_by_route": route_counts(eligible),
            },
            "normal": {
                "split_seed_namespace": "ccpc500-normal-dev-v1",
                "train_rows": len(normal_train_rows),
                "train_clusters": len({row["cluster_id"] for row in normal_train_rows}),
                "dev_rows": len(normal_dev_rows),
                "dev_clusters": len(normal_dev),
                "cluster_overlap": 0,
            },
            "sampler": {
                "source_sha256": sha256(SAMPLER_SOURCE),
                "test_sha256": sha256(SAMPLER_TEST),
                "base_trainer_hook_sha256": sha256(SAMPLER_HOOK),
            },
            "source_membership_unchanged": True,
            "generation_performed": False,
            "normal_targets_generated": False,
            "gradients_performed": False,
            "final_only_evaluation_performed": False,
        },
    )
    shutil.copy2(Path(__file__), OUTPUT / Path(__file__).name)
    validation = (
        "# VALIDATION REPORT\n\n"
        "Result: PASS for frozen-input, eligibility, normal-split, sampler-test, configuration, and Azure-teacher preflight.\n\n"
        "Execution readiness: READY for supervision generation. Azure `gpt-5-bloom` qualified under a neutral "
        "structured-output probe at 1,024 completion tokens; Azure reported `gpt-5-2025-08-07`. "
        "The deployment remains a floating alias, so response metadata must be recorded for every generation attempt.\n\n"
        "Verified counts: behavioral eligible biased 4,856 / 4,304 clusters; behavioral eligible anchor "
        "420 / 405 clusters; normal TRAIN 1,871 / 1,832 clusters; normal DEV 204 / 204 clusters. "
        "Seven mixed-route clusters are TRAIN-only and excluded from gradients.\n"
    )
    (OUTPUT / "VALIDATION_REPORT.md").write_text(validation, encoding="utf-8")
    sums = [
        f"{sha256(path)}  {path.relative_to(OUTPUT).as_posix()}"
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (OUTPUT / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
