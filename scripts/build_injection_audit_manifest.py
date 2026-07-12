from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

from llm_behavior_eval.evaluation_utils.free_text_injection_evaluator import (
    total_refusal_heuristic,
)
from llm_behavior_eval.evaluation_utils.free_text_refusal_evaluator import (
    refusal_heuristic,
)

PROMPT_INJECTION_AUDIT_SPLITS = {
    "malicious": "bloom-prompt-injection-malicious-free-text",
    "conflicting-signals": "bloom-prompt-injection-conflicting-signals-free-text",
    "benign": "bloom-prompt-injection-benign-free-text",
    "purple-llama": "prompt-injection-purple-llama",
}

DEFAULT_AUDIT_SAMPLE_QUOTAS = {
    ("malicious", "judge_yes"): 50,
    ("malicious", "judge_no"): 50,
    ("conflicting-signals", "judge_yes"): 60,
    ("conflicting-signals", "judge_no"): 60,
    ("benign", "judge_no"): 999,
    ("benign", "over_refusal_detected"): 999,
    ("benign", "refusal_marker_present"): 20,
    ("benign", "judge_yes"): 40,
}

JsonObject = dict[str, object]
AuditRow = dict[str, object]


def stable_hash(value: str | None) -> str:
    """
    Return a stable short hash for audit row correlation.

    Args:
        value: Source value to hash.

    Returns:
        A short hexadecimal SHA-256 digest.
    """
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()[:16]


def normalize_space(value: str | None) -> str:
    """
    Collapse whitespace in a preview string.

    Args:
        value: Text to normalize.

    Returns:
        Text with contiguous whitespace collapsed to single spaces.
    """
    return re.sub(r"\s+", " ", value or "").strip()


def extract_user_text(input_text: str | None) -> str:
    """
    Extract the user-visible portion of a decoded chat prompt.

    Args:
        input_text: Decoded chat prompt.

    Returns:
        User-turn text when it can be isolated, otherwise the original prompt.
    """
    if not input_text:
        return ""
    if "user\n" not in input_text:
        return input_text
    user_part = input_text.split("user\n", 1)[1]
    return user_part.split("\nassistant", 1)[0]


def read_json_array(path: Path) -> list[JsonObject]:
    """
    Read a JSON file that must contain a list of objects.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON objects.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected {path} to contain a JSON array of objects.")
    return cast("list[JsonObject]", data)


def flatten_generations(path: Path) -> list[JsonObject]:
    """
    Flatten batched generation JSONL rows into per-sample rows.

    Args:
        path: Path to a generations JSONL file.

    Returns:
        One row per generated response.
    """
    rows: list[JsonObject] = []
    with path.open(encoding="utf-8") as file_handle:
        for line in file_handle:
            if not line.strip():
                continue
            batch = json.loads(line)
            batch_size = len(batch["answers"])
            for index in range(batch_size):
                rows.append(
                    {
                        "input_text": batch["input_texts"][index],
                        "judge_question": batch["judge_questions"][index],
                        "gt_answer": batch["gt_answers"][index],
                        "answer": batch["answers"][index],
                        "finish_reason": batch["finish_reasons"][index],
                        "label": batch.get("labels", [None] * batch_size)[index],
                        "technique": batch.get("techniques", [None] * batch_size)[
                            index
                        ],
                    }
                )
    return rows


def metric_buckets(split: str, response: JsonObject) -> list[str]:
    """
    Return audit bucket labels for a response row.

    Args:
        split: Prompt-injection split name.
        response: Evaluation response row.

    Returns:
        Metric bucket names that apply to the response.
    """
    judge = response.get("judge")
    answer = (
        response.get("llm_answer")
        if isinstance(response.get("llm_answer"), str)
        else None
    )
    buckets = []
    if judge == "Yes":
        buckets.append("judge_yes")
    elif judge == "No":
        buckets.append("judge_no")
    else:
        buckets.append("judge_unparseable")
    if split in {"benign", "conflicting-signals"} and total_refusal_heuristic(answer):
        buckets.append("over_refusal_detected")
    if refusal_heuristic(answer):
        buckets.append("refusal_marker_present")
    return buckets


def load_split_rows(
    results_root: Path, split: str, dataset_slug: str
) -> list[AuditRow]:
    """
    Load and align response and generation rows for one dataset split.

    Args:
        results_root: Model results directory containing dataset subdirectories.
        split: Human-readable split name to store in audit rows.
        dataset_slug: Dataset output directory name.

    Returns:
        Audit rows with aligned prompt, response, judge, and metric metadata.
    """
    split_dir = results_root / dataset_slug
    responses = read_json_array(split_dir / "responses.json")
    generations = flatten_generations(split_dir / "generations.jsonl")
    if len(responses) != len(generations):
        raise RuntimeError(
            f"{dataset_slug} response/generation count mismatch: "
            f"{len(responses)} != {len(generations)}"
        )

    rows = []
    for row_index, (response, generation) in enumerate(
        zip(responses, generations, strict=True)
    ):
        if response.get("llm_answer") != generation["answer"]:
            raise RuntimeError(
                f"{dataset_slug} answer mismatch at flattened row {row_index}"
            )

        input_text = cast("str", generation["input_text"])
        llm_answer = (
            response.get("llm_answer")
            if isinstance(response.get("llm_answer"), str)
            else ""
        )
        buckets = metric_buckets(split, response)
        label = response.get("label", generation.get("label"))
        technique = response.get("technique", generation.get("technique"))
        row = {
            "split": split,
            "dataset_slug": dataset_slug,
            "row_index": row_index,
            "label": label,
            "technique": technique,
            "judge": response.get("judge"),
            "judge_parse_success": response.get("judge_parse_success"),
            "judge_finish_reason": response.get("judge_finish_reason"),
            "finish_reason": generation.get("finish_reason"),
            "metric_buckets": "|".join(buckets),
            "over_refusal_detected": total_refusal_heuristic(llm_answer),
            "refusal_marker_present": refusal_heuristic(llm_answer),
            "input_hash": stable_hash(input_text),
            "response_hash": stable_hash(llm_answer),
            "input_preview": normalize_space(extract_user_text(input_text))[:240],
            "response_preview": normalize_space(llm_answer)[:240],
            "judge_question": generation.get("judge_question"),
            "gt_answer": generation.get("gt_answer"),
            "input_text": input_text,
            "llm_answer": llm_answer,
            "judge_raw": response.get("judge_raw"),
        }
        rows.append(row)
    return rows


def build_manifest(results_root: Path, splits: dict[str, str]) -> list[AuditRow]:
    """
    Build a combined audit manifest for selected prompt-injection splits.

    Args:
        results_root: Model results directory containing dataset subdirectories.
        splits: Mapping from split names to dataset output directory names.

    Returns:
        Combined audit rows from all requested splits.
    """
    rows = []
    for split, dataset_slug in splits.items():
        rows.extend(load_split_rows(results_root, split, dataset_slug))
    return rows


def write_manifest_csv(rows: Sequence[AuditRow], path: Path) -> None:
    """
    Write compact manifest fields to CSV.

    Args:
        rows: Audit rows to write.
        path: Destination CSV path.
    """
    fields = [
        "split",
        "dataset_slug",
        "row_index",
        "label",
        "technique",
        "judge",
        "judge_parse_success",
        "judge_finish_reason",
        "finish_reason",
        "metric_buckets",
        "over_refusal_detected",
        "refusal_marker_present",
        "input_hash",
        "response_hash",
        "input_preview",
        "response_preview",
    ]
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_jsonl(rows: Iterable[AuditRow], path: Path) -> None:
    """
    Write rows to JSONL.

    Args:
        rows: Audit rows to write.
        path: Destination JSONL path.
    """
    with path.open("w", encoding="utf-8") as file_handle:
        for row in rows:
            file_handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_rows(
    rows: Sequence[AuditRow],
    quotas: dict[tuple[str, str], int],
    seed: int,
) -> list[AuditRow]:
    """
    Sample audit rows by deterministic hash order within each split and metric bucket.

    Args:
        rows: Candidate audit rows.
        quotas: Maximum rows to select for each split and metric bucket.
        seed: Seed mixed into the hash order for deterministic sampling.

    Returns:
        Deduplicated sampled rows sorted by split and row index.
    """
    by_key: dict[tuple[str, str], list[AuditRow]] = defaultdict(list)
    for row in rows:
        for bucket in str(row["metric_buckets"]).split("|"):
            by_key[(str(row["split"]), bucket)].append(row)

    selected: dict[tuple[str, int], AuditRow] = {}
    for key, quota in quotas.items():
        pool = sorted(
            by_key.get(key, []),
            key=lambda row: stable_hash(
                f"{seed}:{key[0]}:{key[1]}:{row['split']}:{row['row_index']}"
            ),
        )
        for row in pool[:quota]:
            row_index = row["row_index"]
            if not isinstance(row_index, int):
                raise TypeError(f"Expected integer row_index, got {row_index!r}.")
            selected[(str(row["split"]), row_index)] = row

    return sorted(
        selected.values(), key=lambda row: (str(row["split"]), int(row["row_index"]))
    )


def write_sample_csv(rows: Sequence[AuditRow], path: Path) -> None:
    """
    Write sampled review rows to an annotation CSV.

    Args:
        rows: Sampled audit rows to write.
        path: Destination CSV path.
    """
    fields = [
        "split",
        "row_index",
        "label",
        "technique",
        "judge",
        "metric_buckets",
        "over_refusal_detected",
        "refusal_marker_present",
        "input_preview",
        "response_preview",
        "true_attack",
        "true_total_refusal",
        "task_completed",
        "partial_injected_compliance",
        "dataset_defect",
        "judge_error_type",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields},
                    "true_attack": "",
                    "true_total_refusal": "",
                    "task_completed": "",
                    "partial_injected_compliance": "",
                    "dataset_defect": "",
                    "judge_error_type": "",
                    "notes": "",
                }
            )


def print_summary(rows: Sequence[AuditRow], sample: Sequence[AuditRow]) -> None:
    """
    Log aggregate manifest and sample counts.

    Args:
        rows: Full audit manifest rows.
        sample: Sampled audit rows.
    """
    logging.info("manifest rows: %d", len(rows))
    logging.info("sample rows: %d", len(sample))
    logging.info("judge counts:")
    for key, count in sorted(
        Counter((row["split"], row["judge"]) for row in rows).items()
    ):
        logging.info("  %s: %d", key, count)
    logging.info("metric bucket counts:")
    bucket_counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        for bucket in str(row["metric_buckets"]).split("|"):
            bucket_counts[(str(row["split"]), bucket)] += 1
    for key, count in sorted(bucket_counts.items()):
        logging.info("  %s: %d", key, count)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(
        description="Build audit manifests and review samples for prompt-injection results."
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("audit_outputs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-purple-llama",
        action="store_true",
        help="Include Purple Llama in the manifest. It is not sampled by default quotas.",
    )
    return parser.parse_args()


def main() -> None:
    """Build audit manifest files from evaluation outputs."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    splits = dict(PROMPT_INJECTION_AUDIT_SPLITS)
    if not args.include_purple_llama:
        splits.pop("purple-llama")

    rows = build_manifest(args.results_root, splits)
    sample = sample_rows(rows, DEFAULT_AUDIT_SAMPLE_QUOTAS, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest_csv(rows, args.output_dir / "audit_manifest.csv")
    write_jsonl(rows, args.output_dir / "audit_manifest.jsonl")
    write_sample_csv(sample, args.output_dir / "audit_sample.csv")
    write_jsonl(sample, args.output_dir / "audit_sample.jsonl")
    print_summary(rows, sample)
    logging.info("wrote outputs to %s", args.output_dir)


if __name__ == "__main__":
    main()
