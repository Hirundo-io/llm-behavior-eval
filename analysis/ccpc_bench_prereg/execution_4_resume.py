#!/usr/bin/env python3
"""Execution-4 resume semantics: reconcile state and continue unresolved IDs only."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from argparse import Namespace
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path("/home/ubuntu/hirundo-research")
ART = ROOT / "probe/rsch76-ccpc500-training-data-v1"
EXECUTION = ART / "execution_attempt_4"
INTERRUPTED = EXECUTION / "full_generation"
POPULATION_SIZE = 4856
EXPECTED_MODEL = "gpt-5-2025-08-07"
AMENDMENT_SHA256 = (
    "3b2281ad73d7127c9278c54bf2ab93fb2540c802db9ee88b25427bdf51a4cfc8"
)
AMENDED_CONFIG_SHA256 = (
    "4a6322828ed436e9015ed7ea90684ce7d33cb53da8ea205d24396880787205c3"
)


def now() -> str:
    """Return a UTC ISO timestamp.

    Returns:
        ISO-8601 timestamp string.
    """
    return datetime.now(UTC).isoformat()


def sha256_bytes(data: bytes) -> str:
    """Return SHA-256 hex digest for bytes.

    Args:
        data: Raw bytes to hash.

    Returns:
        Hex digest string.
    """
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return SHA-256 hex digest for one file.

    Args:
        path: File path to hash.

    Returns:
        Hex digest string.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read JSONL, collecting malformed line metadata.

    Args:
        path: JSONL file path.

    Returns:
        Tuple of parsed rows and malformed-line records.
    """
    rows: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    if not path.exists():
        return rows, malformed
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            malformed.append(
                {
                    "line": line_number,
                    "path": str(path),
                    "error": str(exc),
                }
            )
    return rows, malformed


def write_json(path: Path, value: Any) -> None:
    """Write stable JSON.

    Args:
        path: Destination path.
        value: JSON-serializable object.
    """
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_generation_module() -> Any:
    """Import the frozen generation module from this package.

    Returns:
        Loaded build_ccpc500_training_data module.
    """
    package_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_dir))
    import build_ccpc500_training_data as generation  # noqa: PLC0415

    return generation


def ledger_paths(*directories: Path) -> list[Path]:
    """Return attempt ledger paths for one or more execution directories.

    Args:
        directories: Execution directories containing attempt ledgers.

    Returns:
        Existing ledger paths in argument order.
    """
    paths = [directory / "biased_generation_attempts.jsonl" for directory in directories]
    return [path for path in paths if path.exists()]


def reconcile_state(
    membership_path: Path,
    ledger_paths_in_order: list[Path],
    *,
    max_attempts: int,
) -> dict[str, Any]:
    """Reconcile accepted and unresolved IDs across one or more ledgers.

    Args:
        membership_path: Fixed population membership JSONL.
        ledger_paths_in_order: Attempt ledgers in provenance order.
        max_attempts: Retry budget from frozen config.

    Returns:
        Reconciliation report mapping.
    """
    population = sorted(
        int(row["pool_row_id"])
        for row in read_jsonl(membership_path)[0]
    )
    if len(population) != POPULATION_SIZE:
        raise RuntimeError(
            f"unexpected population size: {len(population)} != {POPULATION_SIZE}"
        )

    attempts: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    for ledger_path in ledger_paths_in_order:
        rows, bad = read_jsonl(ledger_path)
        attempts.extend(rows)
        malformed.extend(bad)

    by_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        by_id[int(row["pool_row_id"])].append(row)

    accepted_valid_ids = sorted(
        pid
        for pid, rows in by_id.items()
        if any(row.get("parse_valid") for row in rows)
    )
    duplicate_valid_ids = sorted(
        pid
        for pid, rows in by_id.items()
        if sum(1 for row in rows if row.get("parse_valid")) > 1
    )
    invalid_only_ids = sorted(
        pid
        for pid, rows in by_id.items()
        if pid not in accepted_valid_ids
        and all(not row.get("parse_valid") for row in rows)
        and max(row.get("attempt", 0) for row in rows) < max_attempts
    )
    exhausted_ids = sorted(
        pid
        for pid, rows in by_id.items()
        if pid not in accepted_valid_ids
        and max(row.get("attempt", 0) for row in rows) >= max_attempts
        and all(not row.get("parse_valid") for row in rows)
    )
    attempted_ids = sorted(by_id)
    never_attempted_ids = sorted(set(population) - set(attempted_ids))
    unresolved_ids = sorted(set(population) - set(accepted_valid_ids))

    if len(accepted_valid_ids) + len(unresolved_ids) != POPULATION_SIZE:
        raise RuntimeError("accepted_valid_ids + unresolved_ids != population")

    unresolved_hash = sha256_bytes(
        ("\n".join(str(pid) for pid in unresolved_ids) + "\n").encode("utf-8")
    )

    return {
        "population_size": POPULATION_SIZE,
        "total_attempt_records": len(attempts),
        "unique_ids_attempted": len(attempted_ids),
        "accepted_valid_ids": len(accepted_valid_ids),
        "unresolved_ids": len(unresolved_ids),
        "invalid_only_not_exhausted_ids": len(invalid_only_ids),
        "exhausted_retry_budget_ids": len(exhausted_ids),
        "never_attempted_ids": len(never_attempted_ids),
        "duplicate_valid_ids": len(duplicate_valid_ids),
        "malformed_ledger_records": len(malformed),
        "parse_valid_attempt_records": sum(1 for row in attempts if row.get("parse_valid")),
        "parse_invalid_attempt_records": sum(
            1 for row in attempts if not row.get("parse_valid")
        ),
        "reconciliation_identity": (
            f"{len(accepted_valid_ids)} + {len(unresolved_ids)} = {POPULATION_SIZE}"
        ),
        "unresolved_id_hash_sha256": unresolved_hash,
        "accepted_valid_id_list_sha256": sha256_bytes(
            ("\n".join(str(pid) for pid in accepted_valid_ids) + "\n").encode("utf-8")
        ),
        "duplicate_valid_id_list": duplicate_valid_ids,
        "malformed_records": malformed,
        "unresolved_id_list": unresolved_ids,
        "accepted_valid_id_list": accepted_valid_ids,
        "invalid_only_id_list": invalid_only_ids,
        "exhausted_id_list": exhausted_ids,
        "never_attempted_id_list": never_attempted_ids,
    }


def completed_and_terminal_ids(
    ledger_paths_in_order: list[Path], max_attempts: int
) -> tuple[set[int], set[int]]:
    """Return completed-valid and terminal-unresolved ID sets.

    Args:
        ledger_paths_in_order: Attempt ledgers in provenance order.
        max_attempts: Retry budget from frozen config.

    Returns:
        Tuple of completed-valid IDs and terminal-unresolved IDs.
    """
    attempts: list[dict[str, Any]] = []
    for ledger_path in ledger_paths_in_order:
        attempts.extend(read_jsonl(ledger_path)[0])

    completed = {int(row["pool_row_id"]) for row in attempts if row.get("parse_valid")}
    terminal = {
        int(row["pool_row_id"])
        for row in attempts
        if row.get("attempt") == max_attempts and not row.get("parse_valid")
    }
    return completed, terminal


def resume_generate_biased(
    output_dir: Path,
    upstream_ledgers: list[Path],
    *,
    workers: int,
) -> dict[str, Any]:
    """Generate only unresolved IDs into a continuation ledger.

    Args:
        output_dir: Writable continuation execution directory.
        upstream_ledgers: Read-only upstream attempt ledgers.
        workers: Thread pool size.

    Returns:
        Resume-run summary mapping.
    """
    generation = load_generation_module()
    generation.OUTPUT = output_dir

    config = generation.read_json(output_dir / "supervision_generator_config.json")
    if sha256_file(output_dir / "supervision_generator_config.json") != AMENDED_CONFIG_SHA256:
        raise RuntimeError("continuation config hash differs from frozen amendment")

    template = (output_dir / "supervision_generator_template.txt").read_text(
        encoding="utf-8"
    )
    rows = generation.read_jsonl(output_dir / "biased_membership.jsonl")
    max_attempts = int(config["retry"]["max_attempts"])

    upstream_completed, upstream_terminal = completed_and_terminal_ids(
        upstream_ledgers, max_attempts
    )
    continuation_path = output_dir / "biased_generation_attempts.jsonl"
    continuation_rows, _ = read_jsonl(continuation_path)
    continuation_completed = {
        int(row["pool_row_id"]) for row in continuation_rows if row.get("parse_valid")
    }
    continuation_terminal = {
        int(row["pool_row_id"])
        for row in continuation_rows
        if row.get("attempt") == max_attempts and not row.get("parse_valid")
    }

    skip_ids = (
        upstream_completed
        | upstream_terminal
        | continuation_completed
        | continuation_terminal
    )
    pending = [row for row in rows if int(row["pool_row_id"]) not in skip_ids]
    if any(int(row["pool_row_id"]) in upstream_completed for row in pending):
        raise RuntimeError("resume logic would regenerate a completed valid ID")

    with (
        continuation_path.open("a", encoding="utf-8") as handle,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        futures = {
            pool.submit(generation.teacher_attempt, row, config, template): row
            for row in pending
        }
        for index, future in enumerate(as_completed(futures), 1):
            attempts, _accepted = future.result()
            for item in attempts:
                item["resume_continuation"] = True
                item["upstream_ledgers"] = [str(path) for path in upstream_ledgers]
                handle.write(
                    json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                )
            handle.flush()
            if index % 25 == 0 or index == len(pending):
                print(
                    f"execution-4 resume completed {index}/{len(pending)}",
                    flush=True,
                )

    after = reconcile_state(
        output_dir / "biased_membership.jsonl",
        upstream_ledgers + [continuation_path],
        max_attempts=max_attempts,
    )
    return {
        "output_dir": str(output_dir),
        "upstream_ledgers": [str(path) for path in upstream_ledgers],
        "pending_at_launch": len(pending),
        "post_run_reconciliation": after,
    }


def prepare_resume_dir(resume_dir: Path, interrupted_dir: Path) -> None:
    """Create a continuation directory without mutating interrupted artifacts.

    Args:
        resume_dir: New continuation directory.
        interrupted_dir: Frozen interrupted execution directory.
    """
    if resume_dir.exists() and any(resume_dir.iterdir()):
        raise RuntimeError(f"refusing non-empty resume directory: {resume_dir}")
    resume_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "biased_membership.jsonl",
        "supervision_generator_config.json",
        "supervision_generator_template.txt",
    ):
        target = resume_dir / name
        if not target.exists():
            target.write_bytes((interrupted_dir / name).read_bytes())


def select_canary_ids(unresolved_ids: list[int], count: int) -> list[int]:
    """Deterministically select unresolved IDs for a resume-path canary.

    Args:
        unresolved_ids: Sorted unresolved population IDs.
        count: Number of IDs to select.

    Returns:
        Selected ID list.
    """
    if count > len(unresolved_ids):
        raise RuntimeError("canary count exceeds unresolved population")
    stride = max(1, len(unresolved_ids) // count)
    return [unresolved_ids[index * stride] for index in range(count)]


def run_resume_canary(
    resume_dir: Path,
    upstream_ledgers: list[Path],
    unresolved_ids: list[int],
    *,
    count: int,
) -> dict[str, Any]:
    """Run a tiny resume-path canary over unresolved IDs only.

    Args:
        resume_dir: Continuation directory.
        upstream_ledgers: Upstream ledgers used for skip logic.
        unresolved_ids: Current unresolved IDs.
        count: Canary size.

    Returns:
        Canary report mapping.
    """
    generation = load_generation_module()
    generation.OUTPUT = resume_dir
    config = generation.read_json(resume_dir / "supervision_generator_config.json")
    template = (resume_dir / "supervision_generator_template.txt").read_text(
        encoding="utf-8"
    )
    rows = {
        int(row["pool_row_id"]): row
        for row in generation.read_jsonl(resume_dir / "biased_membership.jsonl")
    }
    selected = select_canary_ids(unresolved_ids, count)
    ledger_path = resume_dir / "biased_generation_attempts.jsonl"
    records: list[dict[str, Any]] = []

    for pool_row_id in selected:
        row = rows[pool_row_id]
        attempts, accepted = generation.teacher_attempt(row, config, template)
        for item in attempts:
            item["resume_continuation"] = True
            item["resume_canary"] = True
            item["upstream_ledgers"] = [str(path) for path in upstream_ledgers]
            with ledger_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                )
        provider = None
        if accepted is not None:
            provider = accepted["provider"]
        response_model = provider.get("model") if provider else None
        finish_reason = None
        if provider and provider.get("choices"):
            finish_reason = provider["choices"][0].get("finish_reason")
        records.append(
            {
                "pool_row_id": pool_row_id,
                "parse_valid": bool(accepted),
                "response_model": response_model,
                "finish_reason": finish_reason,
                "max_completion_tokens": config["max_completion_tokens"],
            }
        )

    mechanical = all(
        row["parse_valid"]
        and row["response_model"] == EXPECTED_MODEL
        and row["finish_reason"] == "stop"
        and row["max_completion_tokens"] == 4096
        for row in records
    )
    return {
        "selected_ids": selected,
        "records": records,
        "mechanical_acceptance": "PASS" if mechanical else "BLOCK",
        "result": "PASS" if mechanical else "BLOCK",
    }


def write_runtime_status(path: Path, payload: dict[str, Any]) -> None:
    """Persist runtime status for a detached generation job.

    Args:
        path: Status artifact path.
        payload: Status mapping.
    """
    write_json(path, payload)


def launch_durable(
    *,
    resume_dir: Path,
    upstream_ledgers: list[Path],
    workers: int,
    log_path: Path,
    status_path: Path,
    pid_path: Path,
) -> int:
    """Launch a setsid-detached resume generation process.

    Args:
        resume_dir: Continuation directory.
        upstream_ledgers: Upstream ledgers for skip logic.
        workers: Worker count.
        log_path: Stdout/stderr log path.
        status_path: Runtime status artifact path.
        pid_path: PID file path.

    Returns:
        Child PID.
    """
    upstream_arg = ",".join(str(path) for path in upstream_ledgers)
    command = [
        "setsid",
        sys.executable,
        str(Path(__file__).resolve()),
        "generate",
        "--resume-dir",
        str(resume_dir),
        "--upstream-ledgers",
        upstream_arg,
        "--workers",
        str(workers),
        "--status-path",
        str(status_path),
    ]
    with log_path.open("ab") as log_handle:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    write_runtime_status(
        status_path,
        {
            "phase": "running",
            "started_at": now(),
            "pid": process.pid,
            "resume_dir": str(resume_dir),
            "upstream_ledgers": upstream_arg.split(","),
            "workers": workers,
            "log_path": str(log_path),
        },
    )
    return process.pid


def cmd_reconcile(args: Namespace) -> None:
    """Write a reconciliation report for execution-4 state."""
    interrupted = Path(args.interrupted_dir)
    resume_dirs = [Path(value) for value in args.resume_dirs]
    config = json.loads(
        (interrupted / "supervision_generator_config.json").read_text(encoding="utf-8")
    )
    report = reconcile_state(
        interrupted / "biased_membership.jsonl",
        ledger_paths(interrupted, *resume_dirs),
        max_attempts=int(config["retry"]["max_attempts"]),
    )
    report["timestamp"] = now()
    report["interrupted_dir"] = str(interrupted)
    report["resume_dirs"] = [str(path) for path in resume_dirs]
    write_json(Path(args.output), report)


def cmd_generate(args: Namespace) -> None:
    """Run continuation generation or finalize runtime status."""
    resume_dir = Path(args.resume_dir)
    upstream_ledgers = [Path(value) for value in args.upstream_ledgers.split(",") if value]
    status_path = Path(args.status_path) if args.status_path else None
    try:
        summary = resume_generate_biased(
            resume_dir,
            upstream_ledgers,
            workers=args.workers,
        )
        if status_path is not None:
            write_runtime_status(
                status_path,
                {
                    "phase": "completed",
                    "completed_at": now(),
                    "exit_code": 0,
                    "signal": None,
                    "summary": summary,
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
                    "exception_class": type(exc).__name__,
                    "message": str(exc),
                },
            )
        raise


def cmd_canary(args: Namespace) -> None:
    """Run a bounded resume-path canary."""
    interrupted = Path(args.interrupted_dir)
    resume_dir = Path(args.resume_dir)
    prepare_resume_dir(resume_dir, interrupted)
    config = json.loads(
        (interrupted / "supervision_generator_config.json").read_text(encoding="utf-8")
    )
    upstream = ledger_paths(interrupted)
    state = reconcile_state(
        interrupted / "biased_membership.jsonl",
        upstream,
        max_attempts=int(config["retry"]["max_attempts"]),
    )
    if state["duplicate_valid_ids"]:
        raise RuntimeError("duplicate valid IDs block resume canary")
    report = run_resume_canary(
        resume_dir,
        upstream,
        state["unresolved_id_list"],
        count=args.count,
    )
    report["timestamp"] = now()
    write_json(Path(args.output), report)
    if report["result"] != "PASS":
        raise RuntimeError("BLOCK — EXECUTION-4 RESUME-PATH CANARY FAILED")


def main() -> None:
    """CLI entrypoint."""
    import argparse

    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    reconcile = commands.add_parser("reconcile")
    reconcile.add_argument("--interrupted-dir", default=str(INTERRUPTED))
    reconcile.add_argument("--resume-dirs", nargs="*", default=[])
    reconcile.add_argument("--output", required=True)
    reconcile.set_defaults(func=cmd_reconcile)

    generate = commands.add_parser("generate")
    generate.add_argument("--resume-dir", required=True)
    generate.add_argument("--upstream-ledgers", required=True)
    generate.add_argument("--workers", type=int, default=8)
    generate.add_argument("--status-path", default="")
    generate.set_defaults(func=cmd_generate)

    canary = commands.add_parser("canary")
    canary.add_argument("--interrupted-dir", default=str(INTERRUPTED))
    canary.add_argument("--resume-dir", required=True)
    canary.add_argument("--count", type=int, default=3)
    canary.add_argument("--output", required=True)
    canary.set_defaults(func=cmd_canary)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
