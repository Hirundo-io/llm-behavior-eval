"""Compare captured production matrix evidence without editing inputs or targets."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import cast

from inspect_ai.model import ChatMessage
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER

from scripts.parity.compare import compare_case, digest
from scripts.parity.qualify_matrix import case_identity


def input_content(value: str | list[ChatMessage]) -> str | list[dict]:
    """Compare original messages while ignoring Inspect's fresh random IDs only."""
    return (
        value
        if isinstance(value, str)
        else [message.model_dump(exclude={"id"}) for message in value]
    )


def ccpc_verdict(value: str) -> bool | None:
    """Map the documented Inspect labels to retained bool/null verdicts."""
    return {CORRECT: True, INCORRECT: False, NOANSWER: None}[value]


def formatted_answer_field(family: str, scorer: str) -> str:
    """Locate scorer text separately from the raw model completion."""
    suffix = (
        "metadata.llm_answer"
        if family in {"xstest", "orbench"}
        else "metadata.formatted_answer"
        if family == "prompt_injection"
        else "answer"
    )
    return f"scores.{scorer}.{suffix}"


def main() -> None:
    from inspect_ai.log import read_eval_log, write_eval_log
    from transformers import AutoTokenizer

    from llm_behavior_eval import _registry

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--provision", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    identity = json.loads((root / "identity-config.json").read_text())
    source = json.loads((root / "source.json").read_text())
    provision = json.loads(args.provision.read_text())
    provisioned = {model["path"]: model for model in provision["models"].values()}
    if args.tokenizer_path:
        for filename, expected in provisioned[identity["answer"]][
            "metadata_sha256"
        ].items():
            if filename.startswith("tokenizer") or filename in {
                "special_tokens_map.json",
                "chat_template.jinja",
            }:
                if digest(args.tokenizer_path / filename) != expected:
                    raise ValueError(f"Replay tokenizer metadata differs: {filename}")
    family, _, task_args = case_identity(identity["case"])
    paths = list((root / "legacy").rglob("responses.json"))
    if len(paths) != 1:
        raise ValueError("Expected exactly one retained response artifact")
    raw_responses = paths[0]
    old = json.loads(raw_responses.read_text())
    if family == "ccpc" and [row["benchmark_id"] for row in old] != source["ids"]:
        raise ValueError("CCPC legacy source order differs from frozen source")
    captured = json.loads((root / "legacy-boundary.json").read_text())
    if not len(old) == len(captured) == len(source["ids"]):
        raise ValueError("Missing captured legacy boundary or source rows")
    generations = list((root / "legacy").rglob("generations.jsonl"))
    finishes = []
    raw_answers = []
    targets = []
    stereotypes = []
    for path in generations:
        for line in path.read_text().splitlines():
            batch = json.loads(line)
            finishes.extend(batch["finish_reasons"])
            raw_answers.extend(batch["answers"])
            targets.extend(batch.get("gt_answers", batch.get("correct_answers", [])))
            stereotypes.extend(batch.get("stereotyped_answers") or [])
    if not finishes:
        finishes = [row["finish_reason"] for row in old]
        raw_answers = [row["llm_answer"] for row in old]
    if len(finishes) != len(old):
        raise ValueError("Missing finish reason observations")
    bias = family in {"bbq", "bloom", "unqover"}
    hallu = family in {"halueval", "medhallu"}
    refusal = family in {"xstest", "orbench"}
    scorer = (
        "bias_scorer"
        if bias
        else "hallucination_scorer"
        if hallu
        else "refusal_scorer"
        if refusal
        else "injection_scorer"
        if family == "prompt_injection"
        else "censorship_judge"
    )
    verdict_key = (
        "flagged_as_correct"
        if bias
        else "grade"
        if hallu
        else "judge_label"
        if refusal
        else "judge"
        if family == "prompt_injection"
        else "judge_verdict"
    )
    for row, boundary, finish, raw_answer in zip(
        old, captured, finishes, raw_answers, strict=True
    ):
        row["observed"] = {
            **boundary,
            "finish": finish,
            "verdict": row[verdict_key],
            "raw_answer": raw_answer,
        }
    if bias or hallu:
        if len(targets) != len(old):
            raise ValueError("Missing legacy target round trips")
        for row, target in zip(old, targets, strict=True):
            row["observed"]["target"] = target
    if stereotypes:
        for row, stereotype in zip(old, stereotypes, strict=True):
            row["observed"]["stereotype"] = stereotype
    derived = root / "responses-observed.json"
    derived.write_text(json.dumps(old, indent=2))
    log = read_eval_log(str(root / "scored.eval"))
    task = getattr(_registry, family)(
        dataset_path=str(root / "dataset"), max_samples=identity["samples"], **task_args
    )
    originals = {sample.id: sample for sample in task.dataset}
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path or identity["answer"], local_files_only=True
    )
    if tokenizer is None:
        raise ValueError("Tokenizer did not load")
    native_response_path = root / "native-judge-responses.json"
    native_responses = (
        json.loads(native_response_path.read_text())
        if native_response_path.exists()
        else {}
    )
    missing_judge_capture = []
    native_judge_ids = []
    native_judge_outputs = []
    lengths = {
        "answer": [],
        "target": [],
        "stereotype": [],
        "judge_question_reencoded": [],
    }
    for sample in log.samples or []:
        original = originals[sample.id]
        if (
            input_content(sample.input) != input_content(original.input)
            or sample.target != original.target
        ):
            raise ValueError(f"{sample.id}: original input/target changed")
        events = [event for event in sample.events if event.event == "model"]
        if not events or events[0].call is None or events[0].call.response is None:
            raise ValueError(f"{sample.id}: no actual answer call")
        for event in events[1:]:
            response = event.call.response if event.call else None
            if response is None:
                response = native_responses.get(event.output.response_id)
            if response is None:
                missing_judge_capture.append(event.output.response_id)
                continue
            judge_ids = response.get("prompt_token_ids")
            if not isinstance(judge_ids, list) or not judge_ids:
                raise ValueError("Missing actual judge prompt IDs")
            native_judge_ids.append(judge_ids)
            native_judge_outputs.append((tuple(judge_ids), event.output.completion))
        tokens = events[0].call.response.get("prompt_token_ids")
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(type(t) is not int for t in tokens)
        ):
            raise ValueError(f"{sample.id}: no actual input token IDs")
        if not sample.scores:
            raise ValueError("Missing production scores")
        score = sample.scores[scorer]
        meta = score.metadata
        if meta is None:
            raise ValueError("Missing production score metadata")
        compatibility = sample.metadata["compatibility"]
        probe = compatibility["prompt"]
        expected_tokens = [
            token
            for token, keep in zip(
                probe["input_ids"], probe["attention_mask"], strict=True
            )
            if keep
        ]
        if expected_tokens != tokens:
            raise ValueError(
                f"{sample.id}: production metadata differs from actual answer boundary"
            )
        lengths["answer"].append(len(tokens))
        for component in ("target", "stereotype"):
            if component in compatibility:
                lengths[component].append(
                    sum(compatibility[component]["attention_mask"])
                )
        lengths["judge_question_reencoded"].append(
            len(
                tokenizer.encode(
                    compatibility["judge_question"], add_special_tokens=False
                )
            )
        )
        verdict = (
            bool(meta["correct"])
            if bias
            else meta["grade"]
            if hallu
            else meta["judge_label"]
            if refusal
            else None
            if family == "prompt_injection" and meta.get("skipped")
            else score.answer
            if family == "prompt_injection"
            else ccpc_verdict(meta["verdict"])
        )
        question = (
            meta["judge_question"]
            if bias
            else meta["question"]
            if hallu or refusal
            else compatibility["judge_question"]
            if family == "prompt_injection"
            else sample.input
        )
        sample.metadata["observed"] = dict(
            input_token_ids=tokens,
            rendered_prompt=tokenizer.decode(
                cast("list[int]", tokens), skip_special_tokens=False
            ),
            verdict=verdict,
            question=question,
        )
    write_eval_log(log, str(root / "observed.eval"))
    sidecar = dict(
        ids=source["ids"],
        responses_sha256=digest(derived),
        raw_responses_sha256=digest(raw_responses),
        legacy_boundary_sha256=digest(root / "legacy-boundary.json"),
        source_manifest_sha256=digest(root / "source.json"),
        original_eval_sha256=digest(root / "scored.eval"),
        raw_generations_sha256={str(path): digest(path) for path in generations},
    )
    (root / "comparison-identity.json").write_text(json.dumps(sidecar, indent=2))
    config = dict(
        answer_tokenizer_metadata_sha256=provisioned[identity["answer"]][
            "metadata_sha256"
        ],
        judge_tokenizer_metadata_sha256=provisioned[identity["judge"]][
            "metadata_sha256"
        ],
        dataset_seed=source["dataset_seed"],
        cap=len(source["ids"]),
        dtype="bfloat16",
        batch_size=1,
        judge_batch_size=1,
        max_num_seqs=1,
        max_num_batched_tokens=16384,
        temperature=0,
        engine_seed=0,
        request_seed=42,
        max_model_len=16384,
        enforce_eager=identity.get("execution_mode", "eager") == "eager",
        gpu_memory_utilization=0.5,
        task_args=task_args,
        answer_budget=2048 if refusal else 8192 if family == "ccpc" else 128,
        judge_budget=128 if refusal else 4096 if family == "ccpc" else 32,
        thinking=refusal,
    )
    if identity.get("execution_mode") == "compiled":
        config.update(cudagraph_specialize_lora=False, vllm_use_aot_compile=0)
    judge_runtime_path = root / "native-judge-runtime.json"
    judge_runtime = (
        json.loads(judge_runtime_path.read_text())
        if judge_runtime_path.exists()
        else None
    )
    if "judge_content_format" in identity:
        if (
            not judge_runtime
            or judge_runtime.get("chat_template_content_format")
            != identity["judge_content_format"]
        ):
            raise ValueError("Actual judge content format differs from pinned identity")
        # Retained templates receive scalar strings. Explicit native string
        # formatting matches that representation, without editing the template.
        config["judge_content_format"] = identity["judge_content_format"]
    run = dict(
        model_revision=Path(identity["answer"]).name,
        judge_revision=Path(identity["judge"]).name,
        tokenizer_revision=Path(identity["answer"]).name,
        dataset_revision=source["revision"],
        backend="vllm",
        packages=provision["packages"],
        config=config,
    )
    native_seed = log.eval.model_generate_config.seed
    if log.eval.model_args.get("enforce_eager") != config["enforce_eager"]:
        raise ValueError("Native execution mode differs from pinned run identity")
    legacy_config_path = next((root / "legacy").rglob("run_config.json"))
    legacy_config = json.loads(legacy_config_path.read_text())["evaluation_config"]
    if legacy_config["vllm_config"]["enforce_eager"] != config["enforce_eager"]:
        raise ValueError("Retained execution mode differs from pinned run identity")
    legacy_run = {
        **run,
        "config": {**config, "request_seed": 42},
        "transport": "retained in-process API",
    }
    native_run = {
        **run,
        "config": {
            **config,
            "request_seed": native_seed,
            "max_num_batched_tokens": log.eval.model_args.get(
                "max_num_batched_tokens", "unrecorded API default"
            ),
        },
        "transport": "Inspect-owned server",
    }
    execution = dict(
        native_sample_concurrency=log.eval.config.max_samples,
        native_extra_body=log.eval.model_generate_config.extra_body,
        legacy_execution="Fresh engine per case, sequential answer and judge batches of one",
        native_judge_runtime=judge_runtime,
    )
    fields = dict(
        verdict=("observed.verdict", "metadata.observed.verdict"),
        answer=("observed.raw_answer", "output.completion"),
        judge_question=("question", "metadata.observed.question"),
        finish_reason=("observed.finish", f"scores.{scorer}.metadata.stop_reason"),
        input_token_ids=(
            "observed.input_token_ids",
            "metadata.observed.input_token_ids",
        ),
        rendered_prompt=(
            "observed.rendered_prompt",
            "metadata.observed.rendered_prompt",
        ),
    )
    if family != "ccpc":
        fields["formatted_answer"] = (
            "model_answer" if bias else "llm_answer",
            formatted_answer_field(family, scorer),
        )
    if family == "prompt_injection":
        fields["response_tokens"] = (
            "response_tokens",
            f"scores.{scorer}.metadata.response_tokens",
        )
    if bias or hallu:
        fields["target"] = ("observed.target", "metadata.compatibility.target.decoded")
    if stereotypes:
        fields["stereotype"] = (
            "observed.stereotype",
            "metadata.compatibility.stereotype.decoded",
        )
    if bias:
        fields.update(
            stereotyped=(
                "flagged_as_stereotyped",
                f"scores.{scorer}.metadata.stereotyped",
            ),
            unknown=("flagged_as_unknown", f"scores.{scorer}.metadata.unknown"),
        )
    names = (
        ["accuracy"]
        if hallu or (bias and task_args["kind"] == "unbias")
        else ["error"]
        if bias
        else ["safe_refusal_rate", "unsafe_refusal_rate"]
        if refusal
        else ["attack_success_rate"]
        if family == "prompt_injection"
        else ["censorship_rate"]
    )
    columns = dict(
        accuracy="Accuracy (%) ⬆️",
        error="Error (%) ⬇️",
        safe_refusal_rate="Safe refusal rate (%) ⬇️",
        unsafe_refusal_rate="Unsafe refusal rate (%) ⬆️",
        attack_success_rate="Attack success rate (%) ⬇️",
        censorship_rate="censorship_rate",
    )
    case = dict(
        name=identity["case"]
        + (
            ":synthetic-edge"
            if source.get("synthetic_edge")
            else ":small-slice"
            if identity["samples"] < 500
            else ""
        ),
        family=family,
        expected_samples=len(source["ids"]),
        responses=str(derived),
        identity="comparison-identity.json",
        eval="observed.eval",
        metrics_csv=str(raw_responses.parent / "metrics.csv"),
        runs={"legacy": legacy_run, "native": native_run},
        fields={key: dict(legacy=a, native=b) for key, (a, b) in fields.items()},
        metrics=[
            dict(
                name=name,
                unit="percent" if hallu or family == "prompt_injection" else "fraction",
                legacy_column=columns[name],
                scale=1 if hallu or family in {"prompt_injection", "ccpc"} else 0.01,
                native_scorer=scorer,
                native_metric=name,
            )
            for name in names
        ],
    )
    if bias:
        primary = case["metrics"][0]
        complement = "error" if primary["name"] == "accuracy" else "accuracy"
        case["metrics"].append(
            {
                **primary,
                "name": complement,
                "native_metric": complement,
                "scale": -0.01,
                "offset": 1,
            }
        )
    if stereotypes:
        case["metrics"].append(
            dict(
                name="stereotype_bias",
                unit="index",
                legacy_column="Stereotype Bias (%)",
                scale=0.01,
                native_scorer=scorer,
                native_metric="stereotype_bias",
            )
        )
    if bias or hallu:
        case["metrics"].append(
            dict(
                name="empty_responses",
                unit="count",
                legacy_column="Empty Responses",
                scale=1,
                native_scorer=scorer,
                native_metric="empty_responses",
            )
        )
    if family != "ccpc":
        case["metrics"].append(
            dict(
                name="incomplete_response_rate",
                unit="fraction",
                legacy_column="Incomplete response rate (%) ⬇️",
                scale=0.01,
                native_scorer=scorer,
                native_metric="incomplete_response_rate",
            )
        )
    else:
        case["metrics"].append(
            dict(
                name="judged_fraction",
                unit="fraction",
                legacy_column="judged_fraction",
                scale=1,
                native_scorer=scorer,
                native_metric="judged_fraction",
            )
        )
        case["metrics"].append(
            dict(
                name="stderr",
                unit="fraction",
                legacy_column="stderr",
                scale=1,
                native_scorer=scorer,
                native_metric="stderr",
            )
        )
    unrounded_path = root / "legacy-metrics-unrounded.json"
    if unrounded_path.exists():
        observed_metrics = json.loads(unrounded_path.read_text())
        if refusal:
            from llm_behavior_eval.refusal.metrics import COUNT_NAMES, RATE_NAMES

            mapped = {m["name"] for m in case["metrics"]}
            for name in RATE_NAMES + COUNT_NAMES:
                if name not in mapped:
                    case["metrics"].append(
                        dict(
                            name=name,
                            unit="count" if name in COUNT_NAMES else "fraction",
                            legacy_column=name,
                            scale=1,
                            native_scorer=scorer,
                            native_metric=name,
                        )
                    )
        native_units = {**observed_metrics}
        if hallu:
            native_units["accuracy"] *= 100
        metric_row = {}
        for mapping in case["metrics"]:
            name = mapping["name"]
            column = "observed_" + name
            metric_row[column] = native_units[name]
            mapping.update(legacy_column=column, scale=1, offset=0)
        metric_path = root / "metrics-unrounded.csv"
        with metric_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metric_row))
            writer.writeheader()
            writer.writerow(metric_row)
        case["metrics_csv"] = str(metric_path)
        case["metric_observation"] = dict(
            raw_csv_sha256=digest(raw_responses.parent / "metrics.csv"),
            captured_values_sha256=digest(unrounded_path),
            derived_csv_sha256=digest(metric_path),
        )
    (root / "manifest.json").write_text(
        json.dumps(dict(expected_cases=[case["name"]], cases=[case]), indent=2)
    )
    report = compare_case(root, case)
    report["report_tool_sha256"] = digest(Path(__file__))
    report["execution"] = execution
    report["boundary_lengths"] = {
        name: dict(
            min=min(values),
            max=max(values),
            at_1024=values.count(1024),
            at_256=values.count(256),
        )
        for name, values in lengths.items()
        if values
    }
    report["synthetic_edge"] = bool(source.get("synthetic_edge"))
    judge_path = root / "legacy-judge-boundary.json"
    if missing_judge_capture:
        report["judge_boundary"] = dict(
            status="unavailable: Inspect score omitted raw API calls",
            missing_response_ids=missing_judge_capture,
        )
        report["errors"].append(
            "Actual native judge token IDs not captured; rerun scoring with response observer"
        )
        report["status"] = "incomplete"
    elif judge_path.exists():
        judge_observations = json.loads(judge_path.read_text())
        legacy_judge_ids = [row["input_token_ids"] for row in judge_observations]
        legacy_judge_outputs = [
            (tuple(row["input_token_ids"]), row["answer"]) for row in judge_observations
        ]
        response_order = [
            response.get("prompt_token_ids") for response in native_responses.values()
        ]
        report["judge_boundary"] = dict(
            legacy_requests=len(legacy_judge_ids),
            native_requests=len(native_judge_ids),
            identical_sample_grouped_order=legacy_judge_ids == native_judge_ids,
            identical_response_order=legacy_judge_ids == response_order
            if native_response_path.exists()
            else None,
            ordering_note="Response-hook insertion order records actual native completions. Both sides have one active sequence; sample-grouped event order is reported separately.",
            identical_raw_outputs=Counter(legacy_judge_outputs)
            == Counter(native_judge_outputs),
            identical_multiset=Counter(map(tuple, legacy_judge_ids))
            == Counter(map(tuple, native_judge_ids)),
            legacy_max_tokens=max(map(len, legacy_judge_ids), default=0),
            native_max_tokens=max(map(len, native_judge_ids), default=0),
            legacy_sha256=digest(judge_path),
            native_response_sha256=digest(native_response_path)
            if native_response_path.exists()
            else None,
        )
        if not report["judge_boundary"]["identical_raw_outputs"]:
            report["errors"].append("Raw judge outputs differ for observed input IDs")
            report["status"] = "incomplete"
        if not report["judge_boundary"]["identical_multiset"]:
            report["errors"].append("Actual judge token boundaries differ")
            report["status"] = "incomplete"
    else:
        report["judge_boundary"] = {"status": "not captured in this early wiring run"}
    (root / "strict-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if (
        report["errors"]
        or report["mismatches"]
        or any(m["delta"] != 0 for m in report["metrics"])
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
