"""Bounded generation-only diagnostic using retained legacy and native Inspect APIs.

This is not benchmark qualification: no judge is run and no score is invented.
Model weights and dataset Arrow file must already exist locally. Run prepare,
legacy, and native in separate processes to release GPU resources between engines.
"""

import argparse
import json
from pathlib import Path

from scripts.parity.historical import legacy_module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "legacy", "native"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arrow", type=Path)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    if args.samples < 1 or args.samples > 32:
        raise ValueError("diagnostic sample cap must be 1..32")
    source = args.root / "dataset"
    if args.mode == "prepare":
        from datasets import Dataset

        if args.arrow is None:
            raise ValueError("--arrow is required for prepare")
        dataset = Dataset.from_file(str(args.arrow))
        source.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(str(source / "train.parquet"))
        selected = dataset.add_column("_source_id", list(range(len(dataset)))).shuffle(
            seed=42
        )
        if not isinstance(selected, Dataset):
            raise TypeError("Expected a single shuffled Arrow dataset")
        selected = selected.select(range(min(args.samples, len(selected))))
        (args.root / "selection.json").write_text(
            json.dumps(
                {
                    "source_arrow": str(args.arrow),
                    "rows": len(dataset),
                    "seed": 42,
                    "ids": [
                        f"hirundo-io/prompt-injection-purple-llama:{i}"
                        for i in selected["_source_id"]
                    ],
                },
                indent=2,
            )
        )
    elif args.mode == "legacy":
        DatasetConfig = legacy_module(
            "llm_behavior_eval.evaluation_utils.dataset_config"
        ).DatasetConfig
        DatasetType = legacy_module(
            "llm_behavior_eval.evaluation_utils.enums"
        ).DatasetType
        EvaluationConfig = legacy_module(
            "llm_behavior_eval.evaluation_utils.eval_config"
        ).EvaluationConfig
        EvaluateFactory = legacy_module(
            "llm_behavior_eval.evaluation_utils.evaluate_factory"
        ).EvaluateFactory
        VllmConfig = legacy_module(
            "llm_behavior_eval.evaluation_utils.vllm_config"
        ).VllmConfig

        evaluator = EvaluateFactory.create_evaluator(
            EvaluationConfig(
                model_path_or_repo_id=args.model,
                inference_engine="vllm",
                max_samples=args.samples,
                batch_size=1,
                sample=False,
                enable_thinking=False,
                max_answer_tokens=128,
                results_dir=args.root / "legacy",
                vllm_config=VllmConfig(
                    max_model_len=8192, gpu_memory_utilization=0.5, enforce_eager=True
                ),
            ),
            DatasetConfig(
                file_path=str(source),
                dataset_id="hirundo-io/prompt-injection-purple-llama",
                dataset_type=DatasetType.BIAS,
                seed=42,
            ),
        )
        try:
            evaluator.generate()
        finally:
            evaluator.free_test_model()
    else:
        from inspect_ai import eval as inspect_eval
        from inspect_ai.model import get_model

        from llm_behavior_eval.prompt_injection.task import prompt_injection

        task = prompt_injection(dataset_path=str(source), max_samples=args.samples)
        task.scorer = []
        task.metadata = {
            **(task.metadata or {}),
            "qualification": "generation-only; no judge; unapproved diagnostic model",
        }
        model = get_model(
            "vllm/" + args.model,
            max_model_len=8192,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
            max_num_seqs=1,
            seed=0,  # Match retained vLLM engine default; dataset shuffle remains 42.
            host="127.0.0.1",
        )
        log = inspect_eval(
            task,
            model=model,
            model_roles={"grader": "mockllm/unused-no-scoring"},
            max_connections=1,
            log_dir=str(args.root / "native"),
            display="none",
        )[0]
        if log.status != "success":
            raise RuntimeError(f"Native run failed: {log.error}")
        print(log.location)


if __name__ == "__main__":
    main()
