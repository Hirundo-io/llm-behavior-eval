"""Aggregate a pinned prefix of actual retained GPU evidence without inference.

This is an evidence replay, not a model implementation. Existing retained graders
parse the recorded raw judgments and aggregate the selected generation records.
Every replayed judge prompt must match its original observed token IDs exactly.

The retained graders are the legacy engine's, which llm-behavior-eval 0.2.0
removed: run this from the frozen pre-removal checkout
(``scripts.parity.historical.FROZEN_COMMIT``; see scripts/parity/README.md).
"""

import argparse
import csv
import json
import shutil
from dataclasses import fields
from pathlib import Path
from typing import Any

from scripts.parity.compare import digest
from scripts.parity.historical import legacy_module, require_frozen_checkout
from scripts.parity.recorded_judgments import RecordedJudgments, judge_tokenizer


def reuse_legacy(primary: Path, output: Path) -> dict[str, Any]:
    """Reuse a deterministic prefix and recompute its metrics via retained graders."""
    source = json.loads((primary / "source.json").read_text())
    selected = json.loads((output / "source.json").read_text())
    identity = json.loads((output / "identity-config.json").read_text())
    n = len(selected["ids"])
    if not n or selected["ids"] != source["ids"][:n]:
        raise ValueError("Reuse requires an exact prefix of the original selected IDs")
    for key in ("repository", "revision", "parquet_sha256", "rows", "dataset_seed"):
        if source[key] != selected[key]:
            raise ValueError(f"Frozen source changed: {key}")
    original_identity = json.loads((primary / "identity-config.json").read_text())
    if {k: v for k, v in identity.items() if k != "samples"} != {
        k: v for k, v in original_identity.items() if k != "samples"
    }:
        raise ValueError("Generation identity changed beyond the selected cap")
    response_paths = list((primary / "legacy").rglob("responses.json"))
    config_paths = list((primary / "legacy").rglob("run_config.json"))
    if len(response_paths) != 1 or len(config_paths) != 1:
        raise ValueError("Missing or ambiguous completed retained evidence")
    original = json.loads(response_paths[0].read_text())
    boundaries = json.loads((primary / "legacy-boundary.json").read_text())
    if not len(original) == len(boundaries) == len(source["ids"]):
        raise ValueError("Original retained evidence is incomplete")
    config = json.loads(config_paths[0].read_text())
    legacy = "llm_behavior_eval.evaluation_utils"
    EvaluateFactory = legacy_module(f"{legacy}.evaluate_factory").EvaluateFactory
    EvaluationConfig = legacy_module(f"{legacy}.eval_config").EvaluationConfig
    DatasetConfig = legacy_module(f"{legacy}.dataset_config").DatasetConfig
    calculate_censorship_metrics = legacy_module(
        f"{legacy}.censorship_utils"
    ).calculate_censorship_metrics
    bias = legacy_module(f"{legacy}.free_text_bias_evaluator")
    hallu = legacy_module(f"{legacy}.free_text_hallu_evaluator")
    injection = legacy_module(f"{legacy}.free_text_injection_evaluator")
    refusal_module = legacy_module(f"{legacy}.free_text_refusal_evaluator")
    censorship_module = legacy_module(f"{legacy}.free_text_censorship_evaluator")
    family = EvaluateFactory.get_evaluator_family(
        config["dataset_config"]["dataset_id"]
    )
    classes: dict[str, Any] = {
        "bias": (bias.FreeTextBiasEvaluator, bias._BiasGenerationRecord),
        "hallucination": (hallu.FreeTextHaluEvaluator, hallu._HalluGenerationRecord),
        "prompt-injection": (
            injection.FreeTextPromptInjectionEvaluator,
            injection._InjectionGenerationRecord,
        ),
        "refusal": (
            refusal_module.FreeTextRefusalEvaluator,
            refusal_module._RefusalGenerationRecord,
        ),
        "censorship": (
            censorship_module.FreeTextCensorshipEvaluator,
            censorship_module._CensorshipGenerationRecord,
        ),
    }
    evaluator_class, record_class = classes[family]
    generation_paths = list((primary / "legacy").rglob("generations.jsonl"))
    selected_lines = []
    if family == "censorship":
        records = [
            record_class(
                answers=[r["llm_answer"]],
                questions=[r["question"]],
                row_ids=[r["benchmark_id"]],
                finish_reasons=[r["finish_reason"]],
            )
            for r in original[:n]
        ]
    else:
        if len(generation_paths) != 1:
            raise ValueError("Missing or ambiguous actual generation cache")
        lines = generation_paths[0].read_bytes().splitlines(keepends=True)
        if len(lines) != len(original):
            raise ValueError("Reuse requires the observed batch size of one")
        selected_lines = lines[:n]
        raw = [json.loads(line) for line in selected_lines]
        if any(len(row["answers"]) != 1 for row in raw):
            raise ValueError("Reuse requires single-row generation records")
        records = [
            record_class(**{f.name: row[f.name] for f in fields(record_class)})
            for row in raw
        ]
    # The original EvaluateFactory constructor loads a model. Only its existing
    # parsing/aggregation methods are needed here; never construct a new engine.
    evaluator = evaluator_class.__new__(evaluator_class)
    evaluator.eval_config = EvaluationConfig.model_validate(config["evaluation_config"])
    evaluator.dataset_config = DatasetConfig.model_validate(config["dataset_config"])
    evaluator.num_samples = n
    evaluator.has_stereotype = (
        family == "bias" and records[0].stereotyped_answers is not None
    )
    evaluator.judge_tokenizer = judge_tokenizer(identity["judge"])
    observations = json.loads((primary / "legacy-judge-boundary.json").read_text())
    replay = RecordedJudgments(evaluator.judge_tokenizer, observations)
    evaluator.run_judge_with_backoff = replay
    captured: dict[str, Any] = {}

    def shared(
        responses,
        accuracy,
        stereotyped_bias,
        empty_responses,
        incomplete_response_rate=None,
    ):
        captured.update(
            responses=responses,
            metrics=dict(
                accuracy=accuracy,
                error=1 - accuracy,
                attack_success_rate=100 * (1 - accuracy),
                stereotype_bias=stereotyped_bias,
                empty_responses=empty_responses,
                incomplete_response_rate=incomplete_response_rate,
            ),
        )

    def refusal(responses, metrics):
        captured.update(responses=responses, metrics=metrics)

    def censorship(responses):
        rate, stderr, judged = calculate_censorship_metrics(
            [r["judge_verdict"] for r in responses]
        )
        captured.update(
            responses=responses,
            metrics=dict(censorship_rate=rate, stderr=stderr, judged_fraction=judged),
        )

    evaluator.save_results = shared
    evaluator._save_refusal_results = refusal
    evaluator._save_results = censorship
    evaluator._grade_impl(records, judge_engine=object())
    if captured["responses"] != original[:n]:
        raise ValueError("Retained replay changed original per-row responses/verdicts")
    evidence = output / "legacy" / "reused-evidence"
    evidence.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_paths[0], evidence / "run_config.json")
    (evidence / "responses.json").write_text(
        json.dumps(captured["responses"], indent=2)
    )
    if selected_lines:
        (evidence / "generations.jsonl").write_bytes(b"".join(selected_lines))
    with (evidence / "metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(captured["metrics"]))
        writer.writeheader()
        writer.writerow(captured["metrics"])
    (output / "legacy-metrics-unrounded.json").write_text(
        json.dumps(captured["metrics"], indent=2)
    )
    (output / "legacy-boundary.json").write_text(json.dumps(boundaries[:n]))
    (output / "legacy-judge-boundary.json").write_text(
        json.dumps(observations[: replay.consumed])
    )
    paths = [
        primary / "source.json",
        primary / "identity-config.json",
        primary / "legacy-boundary.json",
        primary / "legacy-judge-boundary.json",
        response_paths[0],
        config_paths[0],
        *generation_paths,
    ]
    provenance = dict(
        kind="Actual immutable GPU evidence prefix; CPU aggregation replay only",
        original_root=str(primary.resolve()),
        original_generation_cap=original_identity["samples"],
        selected_rows=n,
        observed_judge_requests=replay.consumed,
        selection="First selected source IDs in the existing frozen order; no output-based row selection",
        metrics="Recomputed on selected actual rows via retained grader methods; no full-run aggregate reused",
        prompt_check="Every replayed judge prompt matched original observed input token IDs",
        original_hashes={str(path): digest(path) for path in paths},
        retained_original_config=True,
    )
    (output / "legacy-reuse-provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    args = parser.parse_args()
    require_frozen_checkout()
    for case in args.cases:
        name = case.replace(":", "-")
        reuse_legacy(args.primary_root / name, args.root / name)


if __name__ == "__main__":
    main()
