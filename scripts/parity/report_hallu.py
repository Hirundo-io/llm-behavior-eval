"""Build strict evidence for qualify_hallu without inventing legacy boundaries.

Capture both actual legacy API inputs and native server inputs. Raw generation
logs are preserved; a hashed derived copy attaches observed legacy boundary rows.
Missing observations fail closed rather than substituting a tokenizer oracle.
"""

import argparse
import json
from pathlib import Path
from typing import cast

from scripts.parity.compare import compare_case, digest
from scripts.parity.provision_models import MODELS


def main() -> None:
    from inspect_ai.log import read_eval_log, write_eval_log
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--provision", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source = json.loads(args.source.read_text())
    provision = json.loads(args.provision.read_text())
    legacy = root / "legacy" / MODELS["google/gemma-3-4b-it"] / "halueval"
    responses = legacy / "responses.json"
    generations = legacy / "generations.jsonl"
    captured = json.loads((root / "legacy-boundary.json").read_text())
    batches = [json.loads(line) for line in generations.read_text().splitlines()]
    cursor = 0
    for batch in batches:
        count = len(batch["answers"])
        observed = captured[cursor : cursor + count]
        if len(observed) != count:
            raise ValueError(
                "Missing actual legacy input observations; rerun without cached generations"
            )
        for key in ("rendered_prompt", "input_token_ids"):
            batch[key] = [row[key] for row in observed]
        cursor += count
    if cursor != len(captured):
        raise ValueError("Legacy boundary observation count mismatch")
    raw_generations = generations
    generations = root / "generations-boundary.jsonl"
    generations.write_text("".join(json.dumps(batch) + "\n" for batch in batches))
    identity = {
        "responses_sha256": digest(responses),
        "raw_generations_sha256": digest(raw_generations),
        "legacy_boundary_sha256": digest(root / "legacy-boundary.json"),
        "generations_sha256": digest(generations),
        "ids": source["ids"],
    }
    (root / "identity.json").write_text(json.dumps(identity, indent=2))
    log = read_eval_log(str(root / "scored.eval"))
    tokenizer = AutoTokenizer.from_pretrained(
        provision["models"]["google/gemma-3-4b-it"]["path"]
    )
    if tokenizer is None:
        raise ValueError("Answer tokenizer failed to load")
    boundary = []
    for sample in log.samples or []:
        # Score adds judge model events after the original answer event.
        events = [event for event in sample.events if event.event == "model"]
        if not events or events[0].call is None:
            raise ValueError(f"{sample.id}: missing answer model call")
        response = events[0].call.response
        if response is None:
            raise ValueError(f"{sample.id}: missing model response")
        ids = response.get("prompt_token_ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{sample.id}: actual prompt token IDs absent")
        if any(type(token) is not int for token in ids):
            raise ValueError(f"{sample.id}: invalid token ID type")
        ids = cast("list[int]", ids)
        probe = sample.metadata["legacy_oracle"]["prompt"]
        expected = [
            token
            for token, mask in zip(
                probe["input_ids"], probe["attention_mask"], strict=True
            )
            if mask
        ]
        sample.metadata["actual_input_token_ids"] = ids
        sample.metadata["actual_decoded_prompt"] = tokenizer.decode(
            ids, skip_special_tokens=True
        )
        sample.metadata["actual_rendered_prompt"] = tokenizer.decode(
            ids, skip_special_tokens=False
        )
        boundary.append(
            {
                "id": sample.id,
                "native_prompt_token_ids": ids,
                "expected_legacy_prompt_token_ids": expected,
                "matches_oracle": ids == expected,
                "native_prompt_text": response.get("prompt_text"),
                "legacy_ids_source": "tokenizer oracle, not captured legacy generation",
            }
        )
    write_eval_log(log, str(root / "boundary.eval"))
    config = {
        "seed": 0,
        "dataset_seed": 42,
        "cap": len(source["ids"]),
        "dtype": "bfloat16",
        "batch_size": 1,
        "judge_batch_size": 1,
        "max_num_seqs": 1,
        "device": "NVIDIA H200",
        "quantization": None,
        "temperature": 0,
        "answer_budget": 128,
        "judge_budget": 32,
        "thinking": False,
        "template_kwargs": {"enable_thinking": False},
        "tokenizer_metadata_hashes": provision["models"]["google/gemma-3-4b-it"][
            "metadata_sha256"
        ],
        "judge_tokenizer_metadata_hashes": provision["models"]["google/gemma-3-12b-it"][
            "metadata_sha256"
        ],
        "max_model_len": 8192,
        "enforce_eager": True,
        "gpu_memory_utilization": 0.5,
    }
    run = {
        "model_revision": MODELS["google/gemma-3-4b-it"],
        "judge_revision": MODELS["google/gemma-3-12b-it"],
        "tokenizer_revision": MODELS["google/gemma-3-4b-it"],
        "dataset_revision": source["revision"],
        "backend": "vllm",
        "packages": provision["packages"],
        "config": config,
    }
    fields = {
        "verdict": ("grade", "scores.hallucination_scorer.metadata.grade"),
        "answer": ("llm_answer", "output.completion"),
        "judge_question": ("question", "scores.hallucination_scorer.metadata.question"),
        "finish_reason": (
            "generation.finish_reasons",
            "scores.hallucination_scorer.metadata.stop_reason",
        ),
        "decoded_prompt": ("generation.input_texts", "metadata.actual_decoded_prompt"),
        "input_token_ids": (
            "generation.input_token_ids",
            "metadata.actual_input_token_ids",
        ),
        "rendered_prompt": (
            "generation.rendered_prompt",
            "metadata.actual_rendered_prompt",
        ),
    }
    case = {
        "name": "halueval:small-slice",
        "family": "halueval",
        "expected_samples": len(source["ids"]),
        "responses": str(responses),
        "generations": str(generations),
        "generation_fields": [
            "input_texts",
            "answers",
            "gt_answers",
            "finish_reasons",
            "rendered_prompt",
            "input_token_ids",
        ],
        "identity": "identity.json",
        "eval": "boundary.eval",
        "metrics_csv": str(legacy / "metrics.csv"),
        "runs": {
            "legacy": {**run, "transport": "in-process retained API"},
            "native": {**run, "transport": "Inspect-owned server"},
        },
        "fields": {k: {"legacy": a, "native": b} for k, (a, b) in fields.items()},
        "metrics": [
            {
                "name": "accuracy",
                "unit": "percent",
                "legacy_column": "Accuracy (%) ⬆️",
                "scale": 1,
                "native_scorer": "hallucination_scorer",
                "native_metric": "accuracy",
            }
        ],
    }
    manifest = {"expected_cases": [case["name"]], "cases": [case]}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (root / "strict-report.json").write_text(
        json.dumps(compare_case(root, case), indent=2)
    )
    (root / "boundary.json").write_text(json.dumps(boundary, indent=2))


if __name__ == "__main__":
    main()
