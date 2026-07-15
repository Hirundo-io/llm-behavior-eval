"""Detect and optionally filter ungrounded benign prompt-injection rows.

The detector is deliberately conservative: it only flags extraction/summarization
requests that refer to a missing document and whose reference answer contains
distinctive entities or values absent from the question and system prompt.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from datasets import Dataset, load_dataset

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


DOCUMENT_TASK_RE = re.compile(
    r"\b(?:summari[sz]e|extract|confirm whether|key points|differences)\b", re.I
)
MISSING_DOCUMENT_RE = re.compile(
    r"\b(?:attached|following|this|the|provided(?:\s+an?)?)\s+"
    r"(?:build guide|document|email|invoice(?: text)?|meeting minutes|memo|notice|"
    r"policy change notice|report|transcript)\b",
    re.I,
)
VALUE_RE = re.compile(
    r"(?:\$\s?\d[\d,.]*|\b\d{1,4}(?:[-/:]\d{1,4}){1,2}\b|"
    r"\b\d+(?:\.\d+)?\s?(?:%|kg|days?|weeks?|hours?|SKUs?)\b|"
    r"\b[A-Z]{2,}[A-Z0-9-]*\d[A-Z0-9-]*\b)",
    re.I,
)
CAPITALIZED_PHRASE_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:[- ][A-Za-z]+){1,3})\b")
INLINE_PAYLOAD_MARKERS = ("\n- ", "\n* ", "```", "{", "---", "excerpt:", "text:")


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).casefold()


def _novel_details(answer: str, visible_text: str) -> list[str]:
    visible = _normalized(visible_text)
    candidates = VALUE_RE.findall(answer) + CAPITALIZED_PHRASE_RE.findall(answer)
    return sorted(
        {
            detail.strip(" .,;:()[]")
            for detail in candidates
            if len(detail.strip()) >= 4
            and _normalized(detail.strip(" .,;:()[]")) not in visible
        },
        key=str.casefold,
    )


def detect_ungrounded_row(row: Mapping[str, Any]) -> list[str]:
    """Return deterministic evidence when a benign row lacks its source content."""
    if str(row.get("label", "")).casefold() != "benign":
        return []

    question = str(row.get("question", ""))
    system_prompt = str(row.get("system_prompt", ""))
    answer = str(row.get("answer", ""))
    if not DOCUMENT_TASK_RE.search(question) or not MISSING_DOCUMENT_RE.search(
        question
    ):
        return []

    lowered_question = question.casefold()
    # The generated benchmark puts supplied document bodies on following lines.
    # A single-line request that merely says a document was supplied is the defect.
    if "\n" in question.strip() or any(
        marker in lowered_question for marker in INLINE_PAYLOAD_MARKERS
    ):
        return []

    details = _novel_details(answer, f"{system_prompt}\n{question}")
    return details if details else []


def audit_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    findings = []
    for source_row_id, row in enumerate(rows):
        evidence = detect_ungrounded_row(row)
        if evidence:
            findings.append(
                {
                    "source_row_id": source_row_id,
                    "technique": row.get("technique"),
                    "novel_answer_details": evidence,
                }
            )
    return findings


def _load(source: str, split: str) -> Dataset:
    path = Path(source)
    if path.suffix == ".arrow":
        return Dataset.from_file(str(path))
    if path.suffix in {".json", ".jsonl"}:
        return load_dataset("json", data_files=str(path), split="train")
    if path.suffix == ".parquet":
        return load_dataset("parquet", data_files=str(path), split="train")
    return load_dataset(source, split=split)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", help="HF dataset id or local .arrow/.jsonl/.parquet file"
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--filtered-jsonl", type=Path)
    args = parser.parse_args()

    dataset = _load(args.source, args.split)
    findings = audit_rows(dataset)
    flagged_ids = {finding["source_row_id"] for finding in findings}
    report = {
        "source": args.source,
        "split": args.split,
        "input_rows": len(dataset),
        "flagged_count": len(findings),
        "flagged_source_row_ids": sorted(flagged_ids),
        "findings": findings,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(f"{rendered}\n", encoding="utf-8")
    if args.filtered_jsonl:
        args.filtered_jsonl.parent.mkdir(parents=True, exist_ok=True)
        kept_ids = [index for index in range(len(dataset)) if index not in flagged_ids]
        dataset.select(kept_ids).to_json(args.filtered_jsonl, force_ascii=False)


if __name__ == "__main__":
    main()
