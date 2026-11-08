"""Pinned 31-case qualification orchestration using retained and Inspect APIs.

Each phase is a separate process. Outputs are never reused under changed identity.
No successful phase alone constitutes a parity result; compare captured evidence.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.parity.capture import capture_judge_responses
from scripts.parity.ordered import ordered_samples


def case_identity(name: str) -> tuple[str, str, dict[str, Any]]:
    """Resolve required cases against the production preset mapping."""
    from llm_behavior_eval.presets import build_bias_dataset_id

    required = json.loads(Path(__file__).with_name("pending.json").read_text())[
        "expected_cases"
    ]
    if name not in required:
        raise ValueError(f"Not a required matrix case: {name}")
    parts = name.split(":")
    family = parts[0]
    if len(parts) == 3:
        _, bias_type, kind = parts
        variant = "ambiguous-bias" if kind == "ambiguous" else kind
        return (
            family,
            build_bias_dataset_id(family, bias_type, variant),
            dict(bias_type=bias_type, kind=kind),
        )
    return (
        family,
        {
            "halueval": "hirundo-io/halueval",
            "medhallu": "hirundo-io/medhallu",
            "prompt_injection": "hirundo-io/prompt-injection-purple-llama",
            "ccpc": "hirundo-io/ccpc-500",
            "xstest": "hirundo-io/XSTest",
            "orbench": "hirundo-io/or-bench",
        }[family],
        {},
    )


def immutable_json(path: Path, value: Any) -> None:
    """Reject changed run identities instead of overwriting prior evidence."""
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Identity changed: {path}")
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "legacy", "answer", "judge"])
    parser.add_argument("--case", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--answer", required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("Use an explicit positive sample cap")
    family, repository, task_args = case_identity(args.case)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    dataset_path = root / "dataset"
    identity = dict(
        case=args.case, answer=args.answer, judge=args.judge, samples=args.samples
    )
    immutable_json(root / "identity-config.json", identity)
    if args.phase == "prepare":
        from datasets import Dataset, DatasetDict, load_dataset
        from huggingface_hub import HfApi

        from llm_behavior_eval import _registry
        from llm_behavior_eval.ccpc.contract import CCPC_DATASET_REVISION

        revision = (
            CCPC_DATASET_REVISION
            if family == "ccpc"
            else "1516057d20afbd37202ce4d3f8c761c36c4d481c"
            if family == "halueval"
            else HfApi().dataset_info(repository).sha
        )
        splits = load_dataset(repository, revision=revision)
        if not isinstance(splits, DatasetDict):
            raise TypeError("Expected named source splits")
        split = (
            "test"
            if family == "ccpc"
            else "train"
            if "train" in splits
            else next(iter(splits))
        )
        if family != "ccpc" and "train" not in splits and len(splits) != 1:
            raise ValueError("Ambiguous source split")
        source = splits[split]
        if not isinstance(source, Dataset):
            raise TypeError("Expected a single source split")
        dataset_path.mkdir(exist_ok=True)
        parquet = dataset_path / (
            "test.parquet" if family == "ccpc" else "train.parquet"
        )
        source.to_parquet(str(parquet))
        task = getattr(_registry, family)(
            dataset_path=str(dataset_path), max_samples=args.samples, **task_args
        )
        selected = source.add_column(
            "qualification_source_index", list(range(len(source)))
        )
        if family != "ccpc":
            selected = selected.shuffle(seed=42)
        if not isinstance(selected, Dataset):
            raise TypeError("Expected shuffled Dataset")
        selected = selected.select(range(min(args.samples, len(selected))))
        ids = (
            list(selected["benchmark_id"])
            if family == "ccpc"
            else [
                f"{repository}:{index}"
                for index in selected["qualification_source_index"]
            ]
        )
        if ids != [sample.id for sample in task.dataset]:
            raise ValueError(
                "Independent source selection disagrees with native selection"
            )
        immutable_json(
            root / "source.json",
            dict(
                repository=repository,
                revision=revision,
                rows=len(source),
                split=split,
                ids=ids,
                parquet_sha256=hashlib.sha256(parquet.read_bytes()).hexdigest(),
                dataset_seed=None if family == "ccpc" else 42,
            ),
        )
        return
    source = json.loads((root / "source.json").read_text())
    parquet = next(dataset_path.glob("*.parquet"))
    if hashlib.sha256(parquet.read_bytes()).hexdigest() != source["parquet_sha256"]:
        raise ValueError("Pinned dataset bytes changed")
    if args.phase == "legacy":
        from llm_behavior_eval.evaluation_utils.base_evaluator import (
            FreeTextSharedEvaluator,
        )
        from llm_behavior_eval.evaluation_utils.dataset_config import DatasetConfig
        from llm_behavior_eval.evaluation_utils.enums import DatasetType
        from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig
        from llm_behavior_eval.evaluation_utils.evaluate_factory import EvaluateFactory
        from llm_behavior_eval.evaluation_utils.vllm_config import VllmConfig

        evaluator = EvaluateFactory.create_evaluator(
            EvaluationConfig(
                model_path_or_repo_id=args.answer,
                judge_path_or_repo_id=args.judge,
                inference_engine="vllm",
                use_4bit=False,
                use_4bit_judge=False,
                sample=False,
                sample_judge=False,
                max_samples=args.samples,
                batch_size=1,
                judge_batch_size=1,
                results_dir=root / "legacy",
                vllm_config=VllmConfig(
                    max_model_len=16384,
                    gpu_memory_utilization=0.5,
                    language_model_only=True,
                    enforce_eager=True,
                ),
            ),
            DatasetConfig(
                file_path=str(dataset_path),
                dataset_id="chinese_censorship" if family == "ccpc" else repository,
                dataset_type=DatasetType.UNBIAS
                if task_args.get("kind") == "unbias"
                else DatasetType.BIAS,
                seed=42,
            ),
        )
        if not isinstance(evaluator, FreeTextSharedEvaluator):
            raise TypeError("Expected retained free-text evaluator")
        save_results = evaluator.save_results

        def capture_metrics(
            responses: list[dict],
            accuracy: float,
            stereotyped_bias: float | None,
            empty_responses: int,
            incomplete_response_rate: float | None = None,
        ) -> None:
            (root / "legacy-metrics-unrounded.json").write_text(
                json.dumps(
                    dict(
                        accuracy=accuracy,
                        error=1 - accuracy,
                        attack_success_rate=100 * (1 - accuracy),
                        stereotype_bias=stereotyped_bias,
                        empty_responses=empty_responses,
                        incomplete_response_rate=incomplete_response_rate,
                    )
                )
            )
            save_results(
                responses,
                accuracy,
                stereotyped_bias,
                empty_responses,
                incomplete_response_rate=incomplete_response_rate,
            )

        evaluator.save_results = capture_metrics
        if family in {"xstest", "orbench"}:
            from llm_behavior_eval.evaluation_utils.free_text_refusal_evaluator import (
                FreeTextRefusalEvaluator,
            )

            if not isinstance(evaluator, FreeTextRefusalEvaluator):
                raise TypeError("Expected refusal evaluator")
            save_refusal = evaluator._save_refusal_results

            def capture_refusal(responses: Any, metrics: Any) -> None:
                (root / "legacy-metrics-unrounded.json").write_text(json.dumps(metrics))
                save_refusal(responses, metrics)

            evaluator._save_refusal_results = capture_refusal
        original = evaluator.generate_answers
        captured = []

        def capture(input_ids: Any, attention_mask: Any, do_sample: bool | None = None):
            for ids, mask in zip(
                input_ids.tolist(), attention_mask.tolist(), strict=True
            ):
                tokens = [token for token, keep in zip(ids, mask, strict=True) if keep]
                captured.append(
                    dict(
                        input_token_ids=tokens,
                        rendered_prompt=evaluator.tokenizer.decode(
                            tokens, skip_special_tokens=False
                        ),
                    )
                )
            return original(input_ids, attention_mask, do_sample=do_sample)

        judge_boundary = []
        original_judge = evaluator.run_judge_with_backoff

        def capture_judge(
            judge_engine: Any, prompts: Any, repetition_penalty: float = 1.0
        ):
            engine = judge_engine
            generate = engine.generate_answers

            def observe(input_ids: Any, attention_mask: Any, **generation_kwargs: Any):
                ids = input_ids.tolist()
                masks = attention_mask.tolist()
                outputs = generate(input_ids, attention_mask, **generation_kwargs)
                for row, mask, answer, finish in zip(ids, masks, *outputs, strict=True):
                    tokens = [
                        token for token, keep in zip(row, mask, strict=True) if keep
                    ]
                    judge_boundary.append(
                        dict(
                            input_token_ids=tokens,
                            answer=answer,
                            finish_reason=finish,
                            generation_config=str(generation_kwargs),
                        )
                    )
                return outputs

            engine.generate_answers = observe
            try:
                return original_judge(
                    engine, prompts, repetition_penalty=repetition_penalty
                )
            finally:
                engine.generate_answers = generate

        evaluator.run_judge_with_backoff = capture_judge
        evaluator.generate_answers = capture
        try:
            evaluator.evaluate()
        finally:
            (root / "legacy-judge-boundary.json").write_text(json.dumps(judge_boundary))
            (root / "legacy-boundary.json").write_text(json.dumps(captured))
        return
    from inspect_ai import eval as inspect_eval
    from inspect_ai import score_async
    from inspect_ai.log import read_eval_log, write_eval_log
    from inspect_ai.model import GenerateConfig, get_model

    from llm_behavior_eval import _registry

    def model(path: str):
        return get_model(
            "vllm/" + path,
            config=GenerateConfig(
                temperature=0,
                seed=42,
                max_connections=1,
                extra_body={"return_token_ids": True, "add_special_tokens": True},
            ),
            max_model_len=16384,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
            language_model_only=True,
            dtype="bfloat16",
            max_num_seqs=1,
            max_num_batched_tokens=16384,
            seed=0,
            host="127.0.0.1",
        )

    task = getattr(_registry, family)(
        dataset_path=str(dataset_path),
        max_samples=args.samples,
        compatibility_tokenizer=args.answer,
        **task_args,
    )
    if [sample.id for sample in task.dataset] != source["ids"]:
        raise ValueError("Native selection changed from pinned source manifest")
    grader = model(args.judge)
    if args.phase == "answer":
        logs = inspect_eval(
            task,
            model=model(args.answer),
            model_roles={"grader": grader},
            score=False,
            max_connections=1,
            max_samples=1,
            log_dir=str(root / "native"),
            display="none",
        )
        if logs[0].status != "success":
            raise RuntimeError(str(logs[0].error))
        (root / "answer-log.txt").write_text(logs[0].location)
    else:
        log = read_eval_log((root / "answer-log.txt").read_text().strip())
        if not task.scorer:
            raise ValueError("Missing production scorer")
        with capture_judge_responses(grader, root / "native-judge-responses.json"):
            scored = asyncio.run(
                score_async(
                    log,
                    task.scorer,
                    metrics=task.metrics,
                    model_roles={"grader": grader},
                    display="none",
                    samples=ordered_samples(log, source["ids"]),
                    copy=False,
                )
            )
        write_eval_log(scored, str(root / "scored.eval"))


if __name__ == "__main__":
    main()
