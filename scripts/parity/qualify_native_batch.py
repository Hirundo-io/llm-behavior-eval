"""Reuse Inspect-owned model servers across production matrix tasks.

All roots must already be prepared by qualify_matrix and share model identities.
Run answer and judge in separate processes; CCPC judge gets a separate group.
"""

import argparse
import asyncio
import json
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai import score_async
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import GenerateConfig, get_model

from llm_behavior_eval import _registry
from scripts.parity.compare import digest
from scripts.parity.qualify_matrix import case_identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["answer", "judge"])
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    identities = [
        json.loads((root / "identity-config.json").read_text()) for root in args.roots
    ]
    for key in ("answer", "judge"):
        if len({identity[key] for identity in identities}) != 1:
            raise ValueError(f"Batch requires identical {key} identity")

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
    grader = model(identities[0]["judge"])
    if args.phase == "answer":
        logs = inspect_eval(
            tasks,
            model=model(identities[0]["answer"]),
            model_roles={"grader": grader},
            score=False,
            max_connections=1,
            max_tasks=1,
            log_dir=str(args.log_dir),
            display="none",
        )
        if len(logs) != len(tasks):
            raise ValueError("Missing task logs")
        failed = []
        for root, log in zip(args.roots, logs, strict=True):
            if log.status != "success":
                failed.append(str(log.error))
                continue
            saved = read_eval_log(log.location)
            expected = json.loads((root / "source.json").read_text())["ids"]
            if {s.id for s in saved.samples or []} != set(expected):
                raise ValueError("Batch log does not match source identity")
            (root / "answer-log.txt").write_text(log.location)
        if failed:
            raise RuntimeError("Failed tasks: " + "; ".join(failed))
    else:

        async def judge_all() -> None:
            for root, task in zip(args.roots, tasks, strict=True):
                log = read_eval_log((root / "answer-log.txt").read_text().strip())
                if not task.scorer:
                    raise ValueError("Missing scorer")
                result = await score_async(
                    log,
                    task.scorer,
                    metrics=task.metrics,
                    model_roles={"grader": grader},
                )
                write_eval_log(result, str(root / "scored.eval"))

        asyncio.run(judge_all())


if __name__ == "__main__":
    main()
