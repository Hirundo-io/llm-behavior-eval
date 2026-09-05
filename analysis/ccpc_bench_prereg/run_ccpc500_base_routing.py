#!/usr/bin/env python3
"""Run or resume the frozen RSCH-76 CCPC500 base-routing contract.

The runner intentionally imports the pinned evaluator's public helpers instead
of duplicating prompt, decoding, verdict, or status semantics.  SQLite is the
durable checkpoint; each successful vLLM batch is committed before the next.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from llm_behavior_eval.evaluation_utils.censorship_utils import (
    CCPC_JUDGE_MAX_PROMPT_TOKENS,
    CCPC_JUDGE_MODEL,
    CCPC_JUDGE_MODEL_REVISION,
    CCPC_JUDGE_PROMPT,
    CensorshipStatus,
    judge_prompt_overflows,
    parse_censorship_judgment,
)
from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig
from llm_behavior_eval.evaluation_utils.sampling_config import SamplingConfig
from llm_behavior_eval.evaluation_utils.util_functions import (
    is_model_multimodal,
    safe_apply_chat_template,
)
from llm_behavior_eval.evaluation_utils.vllm_config import VllmConfig
from llm_behavior_eval.evaluation_utils.vllm_eval_engine import VllmEvalEngine


TARGET_MODEL = "Qwen/Qwen3.5-4B"
TARGET_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
EVALUATOR_SHA = "26e634bcdf2e02d1b20e1be58deae71ecb398276"
SOURCE_SHA256 = {
    "train_source.jsonl": "c8be6b35272d83c4711490edc0fb5167f8c858346a9e0b4c5d99e0903f384349",
    "dev_source.jsonl": "5378393dce32492d722a11665f34aca4ab2e9f6245520d0f507b76bddc48bde1",
    "normal_candidate_pool.jsonl": "166713517ab0e03cd89b126ca9f14c756b31376492bf22530f00c57373f69963",
    "semantic_clusters.jsonl": "103c04db999e057ae218a7266dbcd516cdd1d15a8466f8b7fdcf632f8aa02e0f",
}
RUNNER_VERSION = "rsch76-ccpc500-base-routing-runner-v1"
BATCH_SIZE = 8  # The pinned evaluator's vLLM dataloader default.


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS routes (
          pool_row_id INTEGER PRIMARY KEY,
          split TEXT NOT NULL CHECK(split IN ('train', 'dev')),
          cluster_id TEXT NOT NULL,
          topic TEXT NOT NULL,
          request_form TEXT NOT NULL,
          origin_split TEXT NOT NULL,
          origin_split_row_index INTEGER NOT NULL,
          max_cosine_to_ccpc500 REAL NOT NULL,
          question TEXT NOT NULL,
          target_state TEXT NOT NULL DEFAULT 'pending'
              CHECK(target_state IN ('pending','completed')),
          target_answer TEXT,
          target_finish_reason TEXT,
          judge_state TEXT NOT NULL DEFAULT 'pending'
              CHECK(judge_state IN ('pending','completed','overflow','not_applicable')),
          judge_raw TEXT,
          judge_finish_reason TEXT,
          judge_verdict INTEGER,
          censorship_status TEXT,
          route TEXT,
          route_role TEXT
        );
        CREATE TABLE IF NOT EXISTS run_events (
          event_index INTEGER PRIMARY KEY AUTOINCREMENT,
          utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
          phase TEXT NOT NULL,
          event TEXT NOT NULL,
          detail_json TEXT NOT NULL
        );
        """
    )
    return db


def event(db: sqlite3.Connection, phase: str, name: str, detail: dict[str, Any]) -> None:
    db.execute(
        "INSERT INTO run_events(phase,event,detail_json) VALUES(?,?,?)",
        (phase, name, json.dumps(detail, sort_keys=True)),
    )
    db.commit()


def resolve_status(
    model_finish: str | None,
    judge_finish: str | None,
    verdict: bool | None,
    overflow: bool = False,
) -> CensorshipStatus:
    # This is verbatim the evaluator's exhaustive _status decision sequence.
    if model_finish == "length":
        return CensorshipStatus.MODEL_INCOMPLETE
    if model_finish != "stop":
        return CensorshipStatus.MODEL_UNKNOWN
    if overflow:
        return CensorshipStatus.JUDGE_PROMPT_OVERFLOW
    if judge_finish == "length":
        return CensorshipStatus.JUDGE_INCOMPLETE
    if judge_finish != "stop":
        return CensorshipStatus.JUDGE_UNKNOWN
    if verdict is None:
        return CensorshipStatus.JUDGE_UNPARSEABLE
    return CensorshipStatus.JUDGED_TRUE if verdict else CensorshipStatus.JUDGED_FALSE


def assign_route(status: str) -> tuple[str, str]:
    if status == CensorshipStatus.JUDGED_TRUE.value:
        return "censored", "biased_behavior_change"
    if status == CensorshipStatus.JUDGED_FALSE.value:
        return "uncensored", "unbiased_behavioral_retention_anchor"
    return "indeterminate", "explicit_indeterminate"


def check_and_seed(db: sqlite3.Connection, source: Path, artifact: Path) -> None:
    for name, expected in SOURCE_SHA256.items():
        actual = sha256(source / name)
        if actual != expected:
            raise RuntimeError(f"source hash mismatch for {name}: {actual}")
    source_rows: list[dict[str, Any]] = []
    for name, split in (("train_source.jsonl", "train"), ("dev_source.jsonl", "dev")):
        rows = json_rows(source / name)
        if any(row.get("split") != split for row in rows):
            raise RuntimeError(f"split mismatch in {name}")
        source_rows.extend(rows)
    if len(source_rows) != 5855 or sum(r["split"] == "train" for r in source_rows) != 5293:
        raise RuntimeError("frozen behavioral source cardinality mismatch")
    ids = [row["pool_row_id"] for row in source_rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate behavioral pool_row_id")
    normal_ids = {row["pool_row_id"] for row in json_rows(source / "normal_candidate_pool.jsonl")}
    if set(ids) & normal_ids:
        raise RuntimeError("normal candidate identifier entered behavioral routing")
    existing = db.execute("SELECT COUNT(*) FROM routes").fetchone()[0]
    if existing == 0:
        db.executemany(
            """INSERT INTO routes(
                pool_row_id,split,cluster_id,topic,request_form,origin_split,
                origin_split_row_index,max_cosine_to_ccpc500,question)
                VALUES(:pool_row_id,:split,:cluster_id,:topic,:request_form,:origin_split,
                       :origin_split_row_index,:max_cosine_to_ccpc500,:question)""",
            source_rows,
        )
        event(db, "preflight", "source_seeded", {"rows": 5855, "train": 5293, "dev": 562})
    elif existing != 5855:
        raise RuntimeError(f"routing state has {existing} rows, expected 5855")
    snapshot = artifact / "source_membership.json"
    if not snapshot.exists():
        write_json(
            snapshot,
            {
                "train_pool_row_ids_sha256": hashlib.sha256(
                    "\n".join(str(row["pool_row_id"]) for row in source_rows if row["split"] == "train").encode()
                ).hexdigest(),
                "dev_pool_row_ids_sha256": hashlib.sha256(
                    "\n".join(str(row["pool_row_id"]) for row in source_rows if row["split"] == "dev").encode()
                ).hexdigest(),
                "source_file_sha256": {name: sha256(source / name) for name in SOURCE_SHA256},
                "behavioral_rows": 5855,
                "train_rows": 5293,
                "dev_rows": 562,
                "normal_candidate_rows_excluded": len(normal_ids),
            },
        )


def config(artifact: Path) -> EvaluationConfig:
    return EvaluationConfig(
        model_path_or_repo_id=TARGET_MODEL,
        model_revision=TARGET_REVISION,
        judge_path_or_repo_id=CCPC_JUDGE_MODEL,
        judge_revision=CCPC_JUDGE_MODEL_REVISION,
        results_dir=artifact,
        model_engine="vllm",
        judge_engine="vllm",
        sample=False,
        sample_judge=False,
        max_answer_tokens=8192,
        max_judge_tokens=128,
        enable_thinking=False,
        exclude_thinking_trace_for_judge=False,
        trust_remote_code=False,
        sampling_config=SamplingConfig(do_sample=False, temperature=0, top_p=1, top_k=0, seed=None),
        # These are the evaluator CLI's documented vLLM defaults.  Supplying
        # them explicitly prevents vLLM 0.23 from receiving Python None.
        vllm_config=VllmConfig(
            max_model_len=16384,
            judge_max_model_len=262144,
            config_format="auto",
            load_format="auto",
        ),
        evaluator_family="censorship",
    )


def prompt_target(tokenizer: Any, question: str, multimodal: bool) -> str:
    return safe_apply_chat_template(
        tokenizer,
        [{"role": "user", "content": f"{question}\n"}],
        is_multimodal=multimodal,
        max_answer_tokens=8192,
        enable_thinking=False,
        enable_thinking_arg_name=None,
        thinking_start_token=None,
        thinking_end_token=None,
        pass_max_answer_tokens=False,
        reasoning_effort=None,
    )


def tokenize_max(tokenizer: Any, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(prompts, truncation=True, padding="max_length", max_length=1024, return_tensors="pt")
    return encoded["input_ids"], encoded["attention_mask"]


def run_target(db: sqlite3.Connection, cfg: EvaluationConfig) -> None:
    pending = db.execute(
        "SELECT * FROM routes WHERE target_state='pending' ORDER BY pool_row_id"
    ).fetchall()
    if not pending:
        return
    event(db, "target", "engine_initializing", {"unfinished_rows": len(pending), "batch_size": BATCH_SIZE})
    engine = VllmEvalEngine(cfg)
    multimodal = is_model_multimodal(TARGET_MODEL, False, None, TARGET_REVISION)
    try:
        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start : start + BATCH_SIZE]
            prompts = [prompt_target(engine.tokenizer, row["question"], multimodal) for row in batch]
            input_ids, attention_mask = tokenize_max(engine.tokenizer, prompts)
            answers, finishes = engine.generate_answers(
                input_ids,
                attention_mask,
                SamplingConfig(do_sample=False, temperature=0, top_p=1, top_k=0, seed=None),
                repetition_penalty=1.10,
            )
            with db:
                for row, answer, finish in zip(batch, answers, finishes, strict=True):
                    db.execute(
                        """UPDATE routes SET target_state='completed', target_answer=?,
                           target_finish_reason=? WHERE pool_row_id=?""",
                        (answer, finish, row["pool_row_id"]),
                    )
            event(db, "target", "batch_completed", {"first_pool_row_id": batch[0]["pool_row_id"], "rows": len(batch)})
    finally:
        engine.free_model()
        torch.cuda.empty_cache()
    event(db, "target", "engine_released", {"completed_rows": len(pending)})


def mark_target_failures(db: sqlite3.Connection) -> None:
    rows = db.execute(
        "SELECT pool_row_id,target_finish_reason FROM routes WHERE target_state='completed' AND censorship_status IS NULL AND target_finish_reason != 'stop'"
    ).fetchall()
    with db:
        for row in rows:
            status = resolve_status(row["target_finish_reason"], None, None).value
            route, role = assign_route(status)
            db.execute(
                "UPDATE routes SET judge_state='not_applicable', censorship_status=?, route=?, route_role=? WHERE pool_row_id=?",
                (status, route, role, row["pool_row_id"]),
            )


def run_judge(db: sqlite3.Connection, cfg: EvaluationConfig) -> None:
    mark_target_failures(db)
    candidates = db.execute(
        """SELECT * FROM routes WHERE target_state='completed' AND target_finish_reason='stop'
           AND judge_state='pending' ORDER BY pool_row_id"""
    ).fetchall()
    if not candidates:
        return
    # Format and count every complete visible answer before loading the judge.
    from llm_behavior_eval.evaluation_utils.util_functions import load_tokenizer_with_transformers

    judge_tokenizer = load_tokenizer_with_transformers(CCPC_JUDGE_MODEL, revision=CCPC_JUDGE_MODEL_REVISION)
    pending: list[sqlite3.Row] = []
    overflowed = 0
    with db:
        for row in candidates:
            prompt = safe_apply_chat_template(
                judge_tokenizer,
                [{"role": "user", "content": CCPC_JUDGE_PROMPT.format(question=row["question"], response=row["target_answer"])}],
            )
            overflow = judge_prompt_overflows(len(judge_tokenizer(prompt)["input_ids"]))
            if overflow:
                status = resolve_status("stop", None, None, True).value
                route, role = assign_route(status)
                db.execute(
                    """UPDATE routes SET judge_state='overflow', censorship_status=?, route=?, route_role=?
                       WHERE pool_row_id=?""",
                    (status, route, role, row["pool_row_id"]),
                )
                overflowed += 1
            else:
                pending.append(row)
    event(db, "judge", "overflow_screen_complete", {"judgeable_rows": len(pending), "overflow_rows": overflowed, "max_prompt_tokens": CCPC_JUDGE_MAX_PROMPT_TOKENS})
    if not pending:
        return
    event(db, "judge", "engine_initializing", {"unfinished_rows": len(pending), "batch_size": BATCH_SIZE})
    engine = VllmEvalEngine(cfg, is_judge=True, max_model_len=262144)
    try:
        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start : start + BATCH_SIZE]
            prompts = [
                safe_apply_chat_template(
                    judge_tokenizer,
                    [{"role": "user", "content": CCPC_JUDGE_PROMPT.format(question=row["question"], response=row["target_answer"])}],
                )
                for row in batch
            ]
            encoded = judge_tokenizer(prompts, return_tensors="pt", padding=True)
            answers, finishes = engine.generate_answers(
                encoded["input_ids"],
                encoded["attention_mask"],
                SamplingConfig(do_sample=False, temperature=0, top_p=1, top_k=0, seed=None),
                repetition_penalty=1.0,
            )
            with db:
                for row, raw, finish in zip(batch, answers, finishes, strict=True):
                    verdict = parse_censorship_judgment(raw)
                    status = resolve_status("stop", finish, verdict).value
                    route, role = assign_route(status)
                    db.execute(
                        """UPDATE routes SET judge_state='completed', judge_raw=?, judge_finish_reason=?, judge_verdict=?,
                           censorship_status=?, route=?, route_role=? WHERE pool_row_id=?""",
                        (raw, finish, None if verdict is None else int(verdict), status, route, role, row["pool_row_id"]),
                    )
            event(db, "judge", "batch_completed", {"first_pool_row_id": batch[0]["pool_row_id"], "rows": len(batch)})
    finally:
        engine.free_model()
        torch.cuda.empty_cache()
    event(db, "judge", "engine_released", {"completed_rows": len(pending)})


def export_and_validate(db: sqlite3.Connection, artifact: Path) -> None:
    rows = db.execute("SELECT * FROM routes ORDER BY pool_row_id").fetchall()
    if len(rows) != 5855:
        raise RuntimeError("routing state no longer has 5855 rows")
    unresolved = [row["pool_row_id"] for row in rows if row["censorship_status"] is None]
    if unresolved:
        raise RuntimeError(f"routing incomplete: {len(unresolved)} unfinished rows")
    records = [dict(row) for row in rows]
    with (artifact / "routing_records.jsonl").open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    status_counts = Counter(row["censorship_status"] for row in rows)
    route_counts = Counter(row["route"] for row in rows)
    split_rows = {split: [row for row in rows if row["split"] == split] for split in ("train", "dev")}
    for split, expected in (("train", 5293), ("dev", 562)):
        subset = split_rows[split]
        if len(subset) != expected:
            raise RuntimeError(f"{split} cardinality mismatch")
        write_json(
            artifact / f"{split.upper()}_SUMMARY.json",
            {
                "split": split,
                "rows": len(subset),
                "routes": dict(sorted(Counter(row["route"] for row in subset).items())),
                "statuses": dict(sorted(Counter(row["censorship_status"] for row in subset).items())),
                "semantic_clusters": len({row["cluster_id"] for row in subset}),
            },
        )
    cluster_routes: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = cluster_routes.setdefault(row["cluster_id"], {"rows": 0, "routes": Counter(), "splits": Counter()})
        item["rows"] += 1
        item["routes"][row["route"]] += 1
        item["splits"][row["split"]] += 1
    with (artifact / "cluster_by_route_accounting.jsonl").open("w", encoding="utf-8") as out:
        for cluster_id in sorted(cluster_routes):
            item = cluster_routes[cluster_id]
            out.write(json.dumps({"cluster_id": cluster_id, "rows": item["rows"], "routes": dict(item["routes"]), "splits": dict(item["splits"])}, sort_keys=True) + "\n")
    ids = [row["pool_row_id"] for row in rows]
    valid = {
        "result": "PASS",
        "behavioral_identities_accounted_for": len(ids),
        "train_rows": len(split_rows["train"]),
        "dev_rows": len(split_rows["dev"]),
        "unique_pool_row_ids": len(set(ids)),
        "missing_or_duplicate_ids": len(ids) - len(set(ids)),
        "normal_candidate_ids_present": 0,
        "ccpc500_ids_present": 0,
        "source_membership_unchanged": True,
        "status_counts": dict(sorted(status_counts.items())),
        "route_counts": dict(sorted(route_counts.items())),
        "semantic_clusters": len(cluster_routes),
        "train_semantic_clusters": len({r["cluster_id"] for r in split_rows["train"]}),
        "dev_semantic_clusters": len({r["cluster_id"] for r in split_rows["dev"]}),
    }
    write_json(artifact / "VALIDATION_REPORT.json", valid)
    event(db, "finalize", "validation_passed", valid)


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: run_ccpc500_base_routing.py SOURCE_FREEZE ARTIFACT_DIR")
    source, artifact = (Path(value).resolve() for value in sys.argv[1:])
    artifact.mkdir(parents=True, exist_ok=True)
    log_path = artifact / "routing.log"
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    startup_source = Path("/home/ubuntu/llm-behavior-eval/analysis/ccpc_bench_prereg/serving_qualification/evidence/gemma4_26b_a4b_it_262144_startup.log")
    startup_meta = startup_source.with_name("gemma4_26b_a4b_it_262144_startup_meta.json")
    if not (artifact / "GEMMA_STARTUP_QUALIFICATION.log").exists():
        shutil.copy2(startup_source, artifact / "GEMMA_STARTUP_QUALIFICATION.log")
        shutil.copy2(startup_meta, artifact / "GEMMA_STARTUP_QUALIFICATION.json")
    provenance = artifact / "runtime_provenance.json"
    if not provenance.exists():
        write_json(provenance, {"runner_version": RUNNER_VERSION, "evaluator_sha": EVALUATOR_SHA, "target": {"model": TARGET_MODEL, "revision": TARGET_REVISION, "engine": "vllm", "thinking_enabled": False, "temperature": 0, "top_p": 1, "top_k": 0, "max_answer_tokens": 8192, "repetition_penalty": 1.10, "max_model_len": 16384}, "judge": {"model": CCPC_JUDGE_MODEL, "revision": CCPC_JUDGE_MODEL_REVISION, "engine": "vllm", "sampling": False, "max_judge_tokens": 128, "repetition_penalty": 1.0, "max_model_len": 262144, "rubric_version": "chinese_censorship_binary", "complete_visible_answer": True}})
    db = connect(artifact / "routing_state.sqlite3")
    try:
        check_and_seed(db, source, artifact)
        cfg = config(artifact)
        run_target(db, cfg)
        run_judge(db, cfg)
        export_and_validate(db, artifact)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
