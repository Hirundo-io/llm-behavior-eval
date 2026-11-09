"""Reuse Inspect-owned model servers across production matrix tasks.

All roots must already be prepared by qualify_matrix and share model identities.
Run answer and judge in separate processes; CCPC judge gets a separate group.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from inspect_ai import eval_async, score_async
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import GenerateConfig, get_model

from llm_behavior_eval import _registry
from scripts.parity.capture import capture_judge_responses
from scripts.parity.compare import digest
from scripts.parity.ordered import ordered_samples
from scripts.parity.qualify_matrix import case_identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["answer", "judge"])
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--isolate-cache", action="store_true")
    parser.add_argument("--ordered", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("VLLM_USE_AOT_COMPILE", "0")
    identities = [
        json.loads((root / "identity-config.json").read_text()) for root in args.roots
    ]
    for key in ("answer", "judge"):
        if len({identity[key] for identity in identities}) != 1:
            raise ValueError(f"Batch requires identical {key} identity")
    modes = {identity.get("execution_mode", "eager") for identity in identities}
    if len(modes) != 1 or not modes <= {"eager", "compiled"}:
        raise ValueError("Batch requires one supported execution mode")

    def model(path: str, case: str):
        extra_body: dict[str, Any] = {
            "return_token_ids": True,
            "add_special_tokens": True,
        }
        if args.isolate_cache:
            extra_body["cache_salt"] = "llm169:" + case
        return get_model(
            "vllm/" + path,
            config=GenerateConfig(
                temperature=0,
                seed=42,
                max_connections=1,
                extra_body=extra_body,
            ),
            max_model_len=16384,
            gpu_memory_utilization=0.5,
            enforce_eager=modes == {"eager"},
            compilation_config='{"cudagraph_specialize_lora":false}',
            language_model_only=True,
            dtype="bfloat16",
            max_num_seqs=1,
            max_num_batched_tokens=16384,
            seed=0,
            host="127.0.0.1",
        )

    tasks = []
    for root, identity in zip(args.roots, identities, strict=True):
        output = root / ("answer-log.txt" if args.phase == "answer" else "scored.eval")
        if output.exists():
            raise ValueError(f"Refusing to overwrite completed phase: {output}")
        source = json.loads((root / "source.json").read_text())
        if (
            digest(next((root / "dataset").glob("*.parquet")))
            != source["parquet_sha256"]
        ):
            raise ValueError("Pinned source bytes changed")
        family, _, task_args = case_identity(identity["case"])
        task = getattr(_registry, family)(
            dataset_path=str((root / "dataset").resolve()),
            max_samples=identity["samples"],
            compatibility_tokenizer=identity["answer"],
            **task_args,
        )
        if [sample.id for sample in task.dataset] != json.loads(
            (root / "source.json").read_text()
        )["ids"]:
            raise ValueError("Source sample order differs")
        tasks.append(task)
    # Keep wrappers alive while the provider shares its server by model identity.
    graders = [model(i["judge"], i["case"]) for i in identities]
    if args.phase == "answer":
        answers = [model(i["answer"], i["case"]) for i in identities]

        async def answer_all():
            failed = []
            for root, task, answer, grader in zip(
                args.roots, tasks, answers, graders, strict=True
            ):
                logs = await eval_async(
                    task,
                    model=answer,
                    model_roles={"grader": grader},
                    score=False,
                    max_connections=1,
                    max_tasks=1,
                    max_samples=1 if args.ordered else None,
                    log_dir=str(args.log_dir),
                )
                if len(logs) != 1:
                    raise ValueError("Missing or ambiguous task log")
                log = logs[0]
                if log.status != "success":
                    failed.append(str(log.error))
                    continue
                saved = read_eval_log(log.location)
                expected = json.loads((root / "source.json").read_text())["ids"]
                samples = saved.samples or []
                if len(samples) != len(expected) or {s.id for s in samples} != set(
                    expected
                ):
                    raise ValueError("Batch log does not match source identity")
                # Checkpoint each completed task before another expensive task
                # starts. A later failure must not hide successful answer logs.
                (root / "answer-log.txt").write_text(log.location)
            if failed:
                raise RuntimeError("Failed tasks: " + "; ".join(failed))

        asyncio.run(answer_all())
    else:

        async def judge_all() -> None:
            for root, task, grader in zip(args.roots, tasks, graders, strict=True):
                log = read_eval_log((root / "answer-log.txt").read_text().strip())
                if not task.scorer:
                    raise ValueError("Missing scorer")
                with capture_judge_responses(
                    grader, root / "native-judge-responses.json"
                ):
                    result = await score_async(
                        log,
                        task.scorer,
                        metrics=task.metrics,
                        model_roles={"grader": grader},
                        samples=ordered_samples(
                            log, json.loads((root / "source.json").read_text())["ids"]
                        )
                        if args.ordered
                        else None,
                        copy=not args.ordered,
                    )
                write_eval_log(result, str(root / "scored.eval"))

        asyncio.run(judge_all())


if __name__ == "__main__":
    main()
