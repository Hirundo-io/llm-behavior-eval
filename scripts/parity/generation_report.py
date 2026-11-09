"""Summarize real generation evidence without implying judge or benchmark parity."""

import argparse
import hashlib
import json
from pathlib import Path

from inspect_ai.log import read_eval_log


def generation_report(root: Path) -> dict:
    selection = json.loads((root / "selection.json").read_text())
    legacy_paths = list((root / "legacy").rglob("generations.jsonl"))
    native_paths = list((root / "native").glob("*.eval"))
    if len(legacy_paths) != 1 or len(native_paths) != 1:
        raise ValueError(
            "Expected exactly one legacy generation cache and native .eval"
        )
    old = []
    for line in legacy_paths[0].read_text().splitlines():
        batch = json.loads(line)
        old.extend(
            dict(answer=a, finish_reason=f, decoded_input=q, judge_question=j)
            for a, f, q, j in zip(
                batch["answers"],
                batch["finish_reasons"],
                batch["input_texts"],
                batch["judge_questions"],
                strict=True,
            )
        )
    with native_paths[0].open("rb") as stream:
        log = read_eval_log(stream)
    if log.status != "success" or not log.samples:
        raise ValueError("Native log is incomplete")
    native = {str(s.id): s for s in log.samples}
    ids = selection["ids"]
    if (
        len(old) != len(ids)
        or len(native) != len(log.samples)
        or set(ids) != set(native)
    ):
        raise ValueError("Missing/duplicate/misaligned generation identities")
    audit_path = root / "tokenizer-audit.json"
    audit_rows = json.loads(audit_path.read_text())["rows"]
    audit = {r["id"]: r for r in audit_rows}
    if len(audit) != len(audit_rows) or set(audit) != set(ids):
        raise ValueError("Tokenizer/source identity audit is missing or misaligned")
    rows = []
    for ident, legacy in zip(ids, old, strict=True):
        sample = native[ident]
        if (
            legacy["decoded_input"] != audit[ident]["legacy_decoded_input"]
            or legacy["judge_question"] != audit[ident]["legacy_judge_question"]
        ):
            raise ValueError(
                f"Legacy row does not match independent source/tokenizer audit: {ident}"
            )
        if sample.error:
            raise ValueError(f"Sample error: {ident}")
        finish = sample.output.choices[0].stop_reason if sample.output.choices else None
        rows.append(
            {
                "id": ident,
                "legacy_answer": legacy["answer"],
                "native_completion": sample.output.completion,
                "answer_equal": legacy["answer"] == sample.output.completion,
                "legacy_finish": legacy["finish_reason"],
                "native_finish": finish,
                "finish_equal": (
                    "max_tokens"
                    if legacy["finish_reason"] == "length"
                    else legacy["finish_reason"]
                )
                == finish,
                "legacy_judge_question": legacy["judge_question"],
                "native_judge_question": sample.metadata.get("judge_question"),
                "judge_question_equal": legacy["judge_question"]
                == sample.metadata.get("judge_question"),
                "native_content": [
                    c.message.model_dump(mode="json") for c in sample.output.choices
                ],
            }
        )
    return {
        "scope": "generation-only; cached GPT-OSS-20B; no judge executed; not benchmark acceptance",
        "count": len(rows),
        "exact_answer_matches": sum(r["answer_equal"] is True for r in rows),
        "finish_matches": sum(r["finish_equal"] is True for r in rows),
        "judge_question_matches": sum(r["judge_question_equal"] is True for r in rows),
        "rows": rows,
        "artifacts": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*legacy_paths, *native_paths, root / "selection.json", audit_path]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(generation_report(args.root), indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
