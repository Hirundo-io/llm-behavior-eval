"""Audit real cached tokenizer round trips; not a provider-boundary attestation."""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

from scripts.parity.historical import legacy_module


def main() -> None:
    from datasets import Dataset

    safe_apply_chat_template = legacy_module(
        "llm_behavior_eval.evaluation_utils.util_functions"
    ).safe_apply_chat_template
    from transformers import AutoTokenizer  # type: ignore[missing-import]

    from llm_behavior_eval.common.templates import FREE_TEXT_SYSTEM_MESSAGE
    from scripts.parity.tokenizer_probe import legacy_text_probe

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = cast(
        "Any",
        AutoTokenizer.from_pretrained(str(args.model), local_files_only=True),
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    dataset = Dataset.from_parquet(str(args.root / "dataset/train.parquet"))
    selection = json.loads((args.root / "selection.json").read_text())
    rows = []
    for ident in selection["ids"]:
        row = dataset[int(ident.rsplit(":", 1)[1])]
        messages = [
            {
                "role": "system",
                "content": row.get("system_prompt")
                or FREE_TEXT_SYSTEM_MESSAGE["content"],
            },
            {"role": "user", "content": row["question"] + "\n"},
        ]
        legacy = safe_apply_chat_template(
            tokenizer, deepcopy(messages), enable_thinking=False
        )
        native = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        judge_question = row.get("judge_question") or row["question"]
        probe = legacy_text_probe(
            tokenizer, legacy, row["answer"], judge_question=judge_question
        )
        old_ids = [
            i
            for i, mask in zip(
                probe["prompt"]["input_ids"],
                probe["prompt"]["attention_mask"],
                strict=True,
            )
            if mask
        ]
        native_ids = tokenizer.encode(cast("str", native), add_special_tokens=False)
        rows.append(
            {
                "id": ident,
                "rendered_template_equal": legacy == native,
                "legacy_input_tokens": len(old_ids),
                "legacy_decoded_input": probe["prompt"]["decoded"],
                "native_template_tokens": len(native_ids),
                "input_ids_equal": old_ids == native_ids,
                "legacy_judge_question": probe["judge_question"],
                "native_judge_question": judge_question,
                "judge_question_equal": probe["judge_question"] == judge_question,
                "target_equal": probe["target"]["decoded"] == row["answer"],
            }
        )
    paths = [
        args.model / name
        for name in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        )
    ]
    report = {
        "scope": "real tokenizer audit; expected native template tokens, not server capture",
        "rows": rows,
        "model_files_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "dataset_parquet_sha256": hashlib.sha256(
            (args.root / "dataset/train.parquet").read_bytes()
        ).hexdigest(),
    }
    (args.root / "tokenizer-audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
