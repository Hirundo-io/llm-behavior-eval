"""Synthetic fixture builders for the publication result exporter's tests.

Builds small, fabricated evaluator output directories (``responses.json``,
``metrics.csv``, ``run_config.json``) and a matching frozen CCPC dataset file,
shaped exactly like the real evaluator's schemas. The CCPC denominator is
hard-pinned to 500 by the contract, so CCPC fixtures always carry exactly 500
rows (fabricated instantly -- no model or judge is ever invoked); the refusal
fixtures use the frozen 250/200/1319/655 counts. Nothing here reads or depends
on any real result directory.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

CCPC_JUDGE_MODEL = "google/gemma-4-26B-A4B-it"
XSTEST_DATASET = "hirundo-io/XSTest"
OR_BENCH_DATASET = "hirundo-io/or-bench"

CCPC_ROW_COUNT = 500
RARE_TOPIC = "rare_topic"
RARE_REQUEST_FORM = "rare_form"


def benchmark_id(index: int) -> str:
    return f"ccpc500-{index:04d}"


def default_ccpc500_dataset_rows(n: int = CCPC_ROW_COUNT) -> list[tuple[str, str, str]]:
    """(benchmark_id, topic, request_form) triples for a frozen-CCPC-shaped file.

    The first 5 rows share ``RARE_TOPIC``/``RARE_REQUEST_FORM`` (a stratum with
    N=5, below ``MIN_STRATUM_N_FOR_CI``); the remainder round-robin across
    three larger topics/request-forms (each with N >= 10).
    """
    topics = ["topic_a", "topic_b", "topic_c"]
    forms = ["form_x", "form_y", "form_z"]
    rows: list[tuple[str, str, str]] = []
    for i in range(n):
        if i < 5:
            rows.append((benchmark_id(i), RARE_TOPIC, RARE_REQUEST_FORM))
        else:
            rows.append(
                (benchmark_id(i), topics[i % len(topics)], forms[i % len(forms)])
            )
    return rows


def write_ccpc_dataset(path: Path, rows: list[tuple[str, str, str]]) -> tuple[str, int]:
    """Write a frozen-CCPC-style JSONL file: (benchmark_id, topic, request_form) rows.

    Returns:
        ``(sha256, row_count)`` of the written file.
    """
    lines = [
        json.dumps(
            {
                "benchmark_id": row_id,
                "question": f"question for {row_id}",
                "topic": topic,
                "request_form": request_form,
            }
        )
        for row_id, topic, request_form in rows
    ]
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return sha256, len(rows)


def write_ccpc_run(
    output_dir: Path,
    model_repo_id: str,
    dataset_path: str,
    dataset_sha256: str,
    expected_rows: int,
    row_statuses: list[tuple[str, str, bool | None]],
) -> None:
    """Write one CCPC checkpoint's ``responses.json``/``metrics.csv``/``run_config.json``.

    Args:
        row_statuses: One ``(benchmark_id, status, judge_verdict)`` tuple per row,
            in the order the (fabricated) evaluator would have emitted them.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    responses = [
        {
            "benchmark_id": benchmark_id,
            "question": f"question for {benchmark_id}",
            "llm_answer": "synthetic answer",
            "finish_reason": "stop",
            "judge_finish_reason": "stop"
            if status in ("judged_true", "judged_false")
            else None,
            "judge_raw": "True" if verdict else ("False" if verdict is False else ""),
            "judge_verdict": verdict,
            "status": status,
        }
        for benchmark_id, status, verdict in row_statuses
    ]
    (output_dir / "responses.json").write_text(
        json.dumps(responses, indent=2), encoding="utf-8"
    )

    counts: dict[str, int] = {}
    for _, status, _ in row_statuses:
        counts[status] = counts.get(status, 0) + 1

    def _count(name: str) -> int:
        return counts.get(name, 0)

    metrics_row = {
        "Judged True": _count("judged_true"),
        "Judged False": _count("judged_false"),
        "Model Incomplete": _count("model_incomplete"),
        "Model Unknown": _count("model_unknown"),
        "Judge Incomplete": _count("judge_incomplete"),
        "Judge Unknown": _count("judge_unknown"),
        "Judge Unparseable": _count("judge_unparseable"),
        "Accounted Samples": len(row_statuses),
    }
    _write_single_row_csv(output_dir / "metrics.csv", metrics_row)

    run_config = {
        "evaluation_config": {
            "model_path_or_repo_id": model_repo_id,
            "judge_path_or_repo_id": CCPC_JUDGE_MODEL,
        },
        "dataset_config": {
            "dataset_id": "chinese_censorship",
        },
        "ccpc_benchmark": {
            "contract_version": "ccpc-bench-v1",
            "dataset_path": dataset_path,
            "dataset_sha256": dataset_sha256,
            "expected_rows": expected_rows,
            "identity_field": "benchmark_id",
        },
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )


def write_refusal_run(
    output_dir: Path,
    model_repo_id: str,
    dataset_id: str,
    rows: list[dict[str, Any]],
) -> None:
    """Write one XSTest/OR-Bench checkpoint's raw output directory.

    Args:
        rows: Each dict needs ``expected_label`` (``"safe"``/``"unsafe"``),
            ``grading_status``, and ``judge_label`` (``None`` unless judged).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    responses = [
        {
            "dataset": dataset_id,
            "question": f"question {i}",
            "expected_label": row["expected_label"],
            "llm_answer": "synthetic answer",
            "finish_reason": "stop"
            if row["grading_status"] != "model_incomplete"
            else "length",
            "judge_finish_reason": "stop"
            if row["grading_status"] == "judged"
            else None,
            "judge_raw": row.get("judge_label") or "",
            "judge_label": row.get("judge_label"),
            "grading_status": row["grading_status"],
            "heuristic_refusal": False,
            "judge_success": True,
        }
        for i, row in enumerate(rows)
    ]
    (output_dir / "responses.json").write_text(
        json.dumps(responses, indent=2), encoding="utf-8"
    )

    def _label_rows(label: str) -> list[dict[str, Any]]:
        return [r for r in rows if r["expected_label"] == label]

    def _known(label_rows: list[dict[str, Any]]) -> int:
        return sum(1 for r in label_rows if r["grading_status"] == "judged")

    safe_rows = _label_rows("safe")
    unsafe_rows = _label_rows("unsafe")
    metrics_row = {
        "Safe Samples": len(safe_rows),
        "Unsafe Samples": len(unsafe_rows),
        "Safe Known Samples": _known(safe_rows),
        "Unsafe Known Samples": _known(unsafe_rows),
        "Incomplete Responses": sum(
            1 for r in rows if r["grading_status"] == "model_incomplete"
        ),
        "Unknown Finish Reasons": sum(
            1 for r in rows if r["grading_status"] == "unknown_finish_reason"
        ),
        "Judge Unparseable": sum(
            1 for r in rows if r["grading_status"] == "judge_unparseable"
        ),
    }
    _write_single_row_csv(output_dir / "metrics.csv", metrics_row)

    run_config = {
        "evaluation_config": {
            "model_path_or_repo_id": model_repo_id,
        },
        "dataset_config": {
            "dataset_id": dataset_id,
        },
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )


def _write_single_row_csv(path: Path, row: dict[str, Any]) -> None:
    import pandas as pd

    pd.DataFrame([row]).to_csv(path, index=False)


def all_judged_true_refusal_rows(
    safe_count: int, unsafe_count: int, refuse: bool
) -> list[dict[str, Any]]:
    label = "refuse" if refuse else "comply"
    rows = [
        {"expected_label": "safe", "grading_status": "judged", "judge_label": label}
        for _ in range(safe_count)
    ]
    rows += [
        {"expected_label": "unsafe", "grading_status": "judged", "judge_label": label}
        for _ in range(unsafe_count)
    ]
    return rows


def refusal_rows_with_exclusions(
    safe_count: int,
    unsafe_count: int,
    safe_incomplete: int = 0,
    safe_unknown_finish: int = 0,
    safe_unparseable: int = 0,
    unsafe_incomplete: int = 0,
    unsafe_unknown_finish: int = 0,
    unsafe_unparseable: int = 0,
    refuse_fraction: float = 0.0,
) -> list[dict[str, Any]]:
    """Build a full refusal cohort with explicit per-label exclusions.

    Every row in the frozen safe/unsafe partition is present; exclusions are
    represented as non-judged grading statuses rather than dropped rows.
    """
    rows: list[dict[str, Any]] = []

    def _append_block(
        label: str,
        count: int,
        incomplete: int,
        unknown_finish: int,
        unparseable: int,
    ) -> None:
        judged_count = count - incomplete - unknown_finish - unparseable
        if judged_count < 0:
            raise ValueError("exclusion counts exceed partition size")
        refuse_count = int(round(refuse_fraction * judged_count))
        for _ in range(incomplete):
            rows.append(
                {
                    "expected_label": label,
                    "grading_status": "model_incomplete",
                    "judge_label": None,
                }
            )
        for _ in range(unknown_finish):
            rows.append(
                {
                    "expected_label": label,
                    "grading_status": "unknown_finish_reason",
                    "judge_label": None,
                }
            )
        for _ in range(unparseable):
            rows.append(
                {
                    "expected_label": label,
                    "grading_status": "judge_unparseable",
                    "judge_label": None,
                }
            )
        for index in range(judged_count):
            rows.append(
                {
                    "expected_label": label,
                    "grading_status": "judged",
                    "judge_label": "refuse" if index < refuse_count else "comply",
                }
            )

    _append_block(
        "safe",
        safe_count,
        safe_incomplete,
        safe_unknown_finish,
        safe_unparseable,
    )
    _append_block(
        "unsafe",
        unsafe_count,
        unsafe_incomplete,
        unsafe_unknown_finish,
        unsafe_unparseable,
    )
    return rows


def write_manifest(
    path: Path,
    run_id: str,
    entries: list[dict[str, Any]],
    qwen_family_pair: tuple[str, str] | None = None,
) -> None:
    payload: dict[str, Any] = {"run_id": run_id, "entries": entries}
    if qwen_family_pair is not None:
        payload["qwen_family_pair"] = list(qwen_family_pair)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_full_valid_run(
    root: Path,
    models: list[dict[str, Any]],
    qwen_family_pair: tuple[str, str] | None = None,
) -> Path:
    """Build a complete, all-valid multi-checkpoint manifest + result tree.

    Args:
        root: Directory to build the fixture tree under.
        models: One dict per checkpoint with keys ``model_key``, ``lineage``,
            ``checkpoint_label``, ``model_repo_id``, and ``censored_ids`` (a
            set of benchmark_id values judged censored; every other row is
            judged uncensored). All 500 CCPC rows are fully judged and all
            refusal cells are fully judged-compliant (0% refusal) unless the
            test overrides files afterward.

    Returns:
        The path to the written manifest JSON.
    """
    root.mkdir(parents=True, exist_ok=True)
    dataset_path = root / "ccpc500.jsonl"
    dataset_rows = default_ccpc500_dataset_rows()
    dataset_sha256, expected_rows = write_ccpc_dataset(dataset_path, dataset_rows)
    all_ids = [row_id for row_id, _, _ in dataset_rows]

    entries: list[dict[str, Any]] = []
    for model in models:
        model_key = model["model_key"]
        censored_ids = model.get("censored_ids", set())
        ccpc_dir = root / model_key / "ccpc"
        row_statuses = [
            (
                row_id,
                "judged_true" if row_id in censored_ids else "judged_false",
                row_id in censored_ids,
            )
            for row_id in all_ids
        ]
        write_ccpc_run(
            ccpc_dir,
            model["model_repo_id"],
            str(dataset_path),
            dataset_sha256,
            expected_rows,
            row_statuses,
        )

        xstest_dir = root / model_key / "xstest"
        write_refusal_run(
            xstest_dir,
            model["model_repo_id"],
            XSTEST_DATASET,
            all_judged_true_refusal_rows(250, 200, refuse=False),
        )

        orbench_dir = root / model_key / "orbench"
        write_refusal_run(
            orbench_dir,
            model["model_repo_id"],
            OR_BENCH_DATASET,
            all_judged_true_refusal_rows(1319, 655, refuse=False),
        )

        entries.append(
            {
                "model_key": model_key,
                "lineage": model["lineage"],
                "checkpoint_label": model["checkpoint_label"],
                "model_repo_id": model["model_repo_id"],
                "ccpc_dir": str(ccpc_dir),
                "xstest_dir": str(xstest_dir),
                "orbench_dir": str(orbench_dir),
            }
        )

    manifest_path = root / "manifest.json"
    write_manifest(manifest_path, "test-run", entries, qwen_family_pair)
    return manifest_path
