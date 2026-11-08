"""Run a pinned HaluEval slice through retained legacy and native Inspect APIs.

Run legacy, answer, judge in separate processes, in that order. The native
compatibility experiment preserves decoded question/target via the tokenizer
oracle; only captured server token IDs can establish answer-boundary parity.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "legacy", "answer", "judge"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--answer", required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args()
    args.answer = str(Path(args.answer).resolve())
    args.judge = str(Path(args.judge).resolve())
    args.root.mkdir(parents=True, exist_ok=True)
    if args.phase == "prepare":
        from datasets import Dataset, load_dataset

        revision = "1516057d20afbd37202ce4d3f8c761c36c4d481c"
        source = load_dataset("hirundo-io/halueval", revision=revision, split="train")
        if not isinstance(source, Dataset):
            raise TypeError("Expected a single HaluEval split")
        path = Path(args.dataset)
        path.mkdir(parents=True, exist_ok=True)
        parquet = path / "train.parquet"
        source.to_parquet(str(parquet))
        selected = source.add_column("source_index", list(range(len(source)))).shuffle(
            seed=42
        )
        if not isinstance(selected, Dataset):
            raise TypeError("Expected a shuffled HaluEval split")
        selected = selected.select(range(min(args.samples, len(selected))))
        (args.root / "source.json").write_text(
            json.dumps(
                {
                    "revision": revision,
                    "parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
                    "rows": len(source),
                    "seed": 42,
                    "ids": [
                        f"hirundo-io/halueval:{i}" for i in selected["source_index"]
                    ],
                },
                indent=2,
            )
        )
        return
    if args.phase == "legacy":
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
                enable_thinking=False,
                max_samples=args.samples,
                batch_size=1,
                judge_batch_size=1,
                max_answer_tokens=128,
                max_judge_tokens=32,
                results_dir=args.root / "legacy",
                vllm_config=VllmConfig(
                    max_model_len=8192,
                    gpu_memory_utilization=0.5,
                    language_model_only=True,
                    enforce_eager=True,
                ),
            ),
            DatasetConfig(
                file_path=args.dataset,
                dataset_id="hirundo-io/halueval",
                dataset_type=DatasetType.BIAS,
                seed=42,
            ),
        )
        original_generate = evaluator.generate_answers
        boundary = []

        def capture(input_ids: Any, attention_mask: Any, do_sample: bool | None = None):
            # Observe the retained API input; generation and lifecycle stay in it.
            for ids, mask in zip(
                input_ids.tolist(), attention_mask.tolist(), strict=True
            ):
                tokens = [token for token, keep in zip(ids, mask, strict=True) if keep]
                boundary.append(
                    {
                        "input_token_ids": tokens,
                        "rendered_prompt": evaluator.tokenizer.decode(
                            tokens, skip_special_tokens=False
                        ),
                    }
                )
            return original_generate(input_ids, attention_mask, do_sample=do_sample)

        evaluator.generate_answers = capture
        try:
            evaluator.evaluate()
        finally:
            (args.root / "legacy-boundary.json").write_text(
                json.dumps(boundary, indent=2)
            )
        return

    from inspect_ai import eval as inspect_eval
    from inspect_ai import score
    from inspect_ai.log import read_eval_log, write_eval_log
    from inspect_ai.model import GenerateConfig, get_model
    from inspect_ai.solver import Generate, TaskState, solver
    from transformers import AutoTokenizer, PreTrainedTokenizerBase

    from llm_behavior_eval.evaluation_utils.util_functions import (
        safe_apply_chat_template,
    )
    from llm_behavior_eval.hallucination.scorer import hallucination_scorer
    from llm_behavior_eval.halueval.task import halueval
    from scripts.parity.tokenizer_probe import legacy_text_probe

    def model(path: str, *, answer: bool = False):
        extra: dict[str, Any] = {"return_token_ids": True, "add_special_tokens": True}
        if answer:
            extra.update(truncate_prompt_tokens=1024, truncation_side="right")
        return get_model(
            "vllm/" + path,
            config=GenerateConfig(temperature=0, seed=0, extra_body=extra),
            max_model_len=8192,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
            language_model_only=True,
            dtype="bfloat16",
            max_num_seqs=1,
            seed=0,
            host="127.0.0.1",
        )

    grader = model(args.judge)
    if args.phase == "judge":
        log = read_eval_log((args.root / "answer-log.txt").read_text().strip())
        for sample in log.samples or []:
            probe = sample.metadata["legacy_oracle"]
            sample.input = probe["judge_question"]
            sample.target = probe["target"]["decoded"]
        result = score(
            log,
            hallucination_scorer(),
            model_roles={"grader": grader},
            display="none",
        )
        write_eval_log(result, str(args.root / "scored.eval"))
        return

    task = halueval(max_samples=args.samples, dataset_path=args.dataset)
    tokenizer = cast(
        "PreTrainedTokenizerBase", AutoTokenizer.from_pretrained(args.answer)
    )
    original = task.solver

    @solver
    def compatibility():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            # Reference oracle is deliberately retained in this experimental runner,
            # never imported by the production native benchmark.
            from llm_behavior_eval.halueval.prompts import SYSTEM_MESSAGE

            rendered = safe_apply_chat_template(
                tokenizer,
                [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": state.input_text + "\n"},
                ],
            )
            probe = legacy_text_probe(tokenizer, rendered, state.target.text)
            state = await original(state, generate)
            state.metadata["legacy_oracle"] = probe
            state.metadata["original_question"] = state.input_text
            return state

        return solve

    task.solver = compatibility()
    logs = inspect_eval(
        task,
        model=model(args.answer, answer=True),
        model_roles={"grader": grader},
        score=False,
        max_connections=1,
        log_dir=str(args.root / "native"),
        display="none",
    )
    if logs[0].status != "success":
        raise RuntimeError(str(logs[0].error))
    (args.root / "answer-log.txt").write_text(logs[0].location)


if __name__ == "__main__":
    main()
