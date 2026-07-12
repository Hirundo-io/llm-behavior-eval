import json
from pathlib import Path

from scripts.build_injection_audit_manifest import (
    build_manifest,
    flatten_generations,
    sample_rows,
    total_refusal_heuristic,
)


def write_split(
    root: Path,
    dataset_slug: str,
    *,
    answers: list[str],
    judges: list[str | None],
    labels: list[str],
    techniques: list[str],
) -> None:
    """
    Write a minimal evaluation output directory for audit manifest tests.

    Args:
        root: Parent directory for the dataset output directory.
        dataset_slug: Dataset output directory name.
        answers: Generated model answers.
        judges: Parsed judge labels.
        labels: Prompt-injection row labels.
        techniques: Prompt-injection technique names.
    """
    split_dir = root / dataset_slug
    split_dir.mkdir()
    batch_size = len(answers)
    generation = {
        "input_texts": [
            f"system\nPolicy {index}\nuser\nQuestion {index}\nassistant\n"
            for index in range(batch_size)
        ],
        "judge_questions": ["Did it comply?"] * batch_size,
        "gt_answers": [f"target {index}" for index in range(batch_size)],
        "answers": answers,
        "finish_reasons": ["stop"] * batch_size,
        "labels": labels,
        "techniques": techniques,
    }
    (split_dir / "generations.jsonl").write_text(
        json.dumps(generation) + "\n", encoding="utf-8"
    )
    responses = [
        {
            "question": "Did it comply?",
            "llm_answer": answer,
            "judge": judge,
            "judge_raw": judge or "unclear",
            "judge_finish_reason": "stop" if judge is not None else None,
            "judge_parse_success": judge is not None,
            "label": label,
            "technique": technique,
        }
        for answer, judge, label, technique in zip(
            answers, judges, labels, techniques, strict=True
        )
    ]
    (split_dir / "responses.json").write_text(json.dumps(responses), encoding="utf-8")


def test_flatten_generations_preserves_row_metadata(tmp_path: Path) -> None:
    write_split(
        tmp_path,
        "split",
        answers=["answer one", "answer two"],
        judges=["Yes", "No"],
        labels=["benign", "benign"],
        techniques=["ignore_previous", "delimiter_break"],
    )

    rows = flatten_generations(tmp_path / "split" / "generations.jsonl")

    assert [row["answer"] for row in rows] == ["answer one", "answer two"]
    assert [row["label"] for row in rows] == ["benign", "benign"]
    assert [row["technique"] for row in rows] == [
        "ignore_previous",
        "delimiter_break",
    ]


def test_build_manifest_tags_metric_buckets_and_hashes(tmp_path: Path) -> None:
    write_split(
        tmp_path,
        "benign-slug",
        answers=[
            "I cannot complete this benign task.",
            "Here is the requested summary.",
        ],
        judges=["No", "Yes"],
        labels=["benign", "benign"],
        techniques=["ignore_previous", "tool_output_injection"],
    )

    rows = build_manifest(tmp_path, {"benign": "benign-slug"})

    assert len(rows) == 2
    assert rows[0]["metric_buckets"] == (
        "judge_no|over_refusal_detected|refusal_marker_present"
    )
    assert rows[0]["over_refusal_detected"] is True
    assert rows[0]["input_hash"]
    assert rows[0]["response_hash"]
    assert rows[1]["metric_buckets"] == "judge_yes"


def test_sample_rows_deduplicates_rows_selected_by_multiple_buckets(
    tmp_path: Path,
) -> None:
    write_split(
        tmp_path,
        "benign-slug",
        answers=[
            "I cannot complete this benign task.",
            "Here is the requested summary.",
        ],
        judges=["No", "Yes"],
        labels=["benign", "benign"],
        techniques=["ignore_previous", "tool_output_injection"],
    )
    rows = build_manifest(tmp_path, {"benign": "benign-slug"})

    sample = sample_rows(
        rows,
        {
            ("benign", "judge_no"): 10,
            ("benign", "over_refusal_detected"): 10,
            ("benign", "refusal_marker_present"): 10,
        },
        seed=42,
    )

    assert [(row["split"], row["row_index"]) for row in sample] == [("benign", 0)]


def test_total_refusal_heuristic_matches_current_metric_behavior() -> None:
    assert total_refusal_heuristic("I cannot complete this benign task.")
    assert total_refusal_heuristic(
        "I cannot enact the advisory, but here is the requested summary."
    )
    assert not total_refusal_heuristic(
        "I cannot enact the advisory. The requested summary is status green "
        "with two restarts and no unresolved incidents."
    )
