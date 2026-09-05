#!/usr/bin/env python3
"""Finalize the completed CCPC500 base-routing artifact and checksum inventory."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: finalize_ccpc500_base_routing.py ARTIFACT SOURCE RUNNER")
    artifact, source, runner = (Path(arg).resolve() for arg in sys.argv[1:])
    records = [json.loads(line) for line in (artifact / "routing_records.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(records) != 5855:
        raise RuntimeError("cannot finalize incomplete routing records")
    route_counts = Counter(row["route"] for row in records)
    status_counts = Counter(row["censorship_status"] for row in records)
    by_split = {split: [row for row in records if row["split"] == split] for split in ("train", "dev")}
    clusters: dict[str, list[dict]] = {}
    for row in records:
        clusters.setdefault(row["cluster_id"], []).append(row)
    pure_route_clusters = Counter()
    cluster_route_presence = Counter()
    for rows in clusters.values():
        routes = {row["route"] for row in rows}
        for route in routes:
            cluster_route_presence[route] += 1
        pure_route_clusters[next(iter(routes)) if len(routes) == 1 else "mixed"] += 1
    composition = {
        "rows": {
            "total": len(records),
            "train": {"total": len(by_split["train"]), "routes": dict(sorted(Counter(r["route"] for r in by_split["train"]).items()))},
            "dev": {"total": len(by_split["dev"]), "routes": dict(sorted(Counter(r["route"] for r in by_split["dev"]).items()))},
            "all": dict(sorted(route_counts.items())),
            "statuses": dict(sorted(status_counts.items())),
        },
        "semantic_clusters": {
            "total": len(clusters),
            "train": len({r["cluster_id"] for r in by_split["train"]}),
            "dev": len({r["cluster_id"] for r in by_split["dev"]}),
            "exclusive_route_composition": dict(sorted(pure_route_clusters.items())),
            "route_presence": dict(sorted(cluster_route_presence.items())),
            "mixed_route_clusters": pure_route_clusters["mixed"],
        },
    }
    write(artifact / "ROUTE_COMPOSITION.json", composition)
    shutil.copy2(runner, artifact / "run_ccpc500_base_routing.py")
    provenance_path = artifact / "runtime_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["execution_runtime"] = {
        "vllm_version": "0.23.0",
        "torch_version": "2.11.0+cu130",
        "transformers_version": "5.14.1",
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.8,
        "max_num_seqs": 256,
        "operational_batch_size": 8,
        "config_format": "auto",
        "load_format": "auto",
        "runner_sha256": digest(runner),
        "note": "auto config/load values are the pinned evaluator CLI defaults; explicit forwarding avoids vLLM 0.23 rejecting Python None.",
    }
    write(provenance_path, provenance)
    manifest = {
        "artifact": "RSCH-76 CCPC500 base routing v1",
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_freeze": {
            "path": str(source),
            "behavioral_population": 5855,
            "train_rows": 5293,
            "dev_rows": 562,
            "semantic_clusters": 5221,
            "normal_candidate_pool_rows_excluded": 2075,
            "source_membership_snapshot": "source_membership.json",
        },
        "runtime_provenance": "runtime_provenance.json",
        "startup_qualification": {
            "result": "PASS",
            "record": "GEMMA_STARTUP_QUALIFICATION.json",
            "log": "GEMMA_STARTUP_QUALIFICATION.log",
        },
        "routing": {
            "records": "routing_records.jsonl",
            "target": "Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "judge": "google/gemma-4-26B-A4B-it@4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
            "evaluator_sha": "26e634bcdf2e02d1b20e1be58deae71ecb398276",
            "route_counts": dict(sorted(route_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
            "no_training_or_replacement_supervision": True,
        },
        "required_deliverables": {
            "preflight_report": "PREFLIGHT_REPORT.md",
            "train_summary": "TRAIN_SUMMARY.json",
            "dev_summary": "DEV_SUMMARY.json",
            "cluster_by_route_accounting": "cluster_by_route_accounting.jsonl",
            "route_composition": "ROUTE_COMPOSITION.json",
            "failure_retry_ledger": "FAILURE_RETRY_LEDGER.json",
            "validation_report": "VALIDATION_REPORT.json",
        },
    }
    write(artifact / "manifest.json", manifest)
    ledger = {
        "result": "COMPLETE_WITH_PREGENERATION_EXECUTION_RETRIES_ONLY",
        "row_level_retries": 0,
        "affected_behavioral_identities": 0,
        "entries": [
            {"attempt": 1, "phase": "launcher", "outcome": "detached shell did not retain the child process", "row_ids_affected": 0, "resolution": "ran in active execution session"},
            {"attempt": 2, "phase": "runner_import", "outcome": "ModuleNotFoundError before SQLite initialization", "row_ids_affected": 0, "resolution": "set PYTHONPATH to the pinned evaluator checkout"},
            {"attempt": 3, "phase": "target_engine_initialization", "outcome": "vLLM 0.23 rejected unset config_format/load_format before Qwen weights loaded", "row_ids_affected": 0, "resolution": "explicitly forwarded the evaluator CLI defaults config_format=auto and load_format=auto"},
            {"attempt": 4, "phase": "routing", "outcome": "success", "row_ids_affected": 5855, "resolution": "all identities completed exactly once under the frozen generation and judge contract"},
        ],
        "policy_compliance": "No retry was selected by answer content, censorship verdict, or split; no completed row was regenerated.",
    }
    write(artifact / "FAILURE_RETRY_LEDGER.json", ledger)
    lines = []
    for path in sorted(p for p in artifact.iterdir() if p.is_file() and p.name != "SHA256SUMS"):
        lines.append(f"{digest(path)}  {path.name}")
    (artifact / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
