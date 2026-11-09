"""Native answer/score wrapper for all nine production tasks; no alternate evaluator.

Supply local dataset and compatibility tokenizer via --task-args JSON. Provider
model arguments (including server resource limits) are JSON objects per role.
Run answer then judge in separate processes to release GPU resources between roles.
"""

import argparse
import hashlib
import json
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai import score
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import GenerateConfig, get_model

from llm_behavior_eval import _registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["answer", "judge"])
    parser.add_argument(
        "--family",
        required=True,
        choices=[n for n in _registry.__all__ if n != "inspect_template"],
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task-args", type=Path, required=True)
    parser.add_argument("--answer", required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--answer-args", type=Path)
    parser.add_argument("--judge-args", type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    task_args = json.loads(args.task_args.read_text())
    task = getattr(_registry, args.family)(**task_args)
    model_args = {
        role: json.loads(path.read_text()) if path else {}
        for role, path in [("answer", args.answer_args), ("judge", args.judge_args)]
    }
    identity = dict(
        family=args.family,
        task_args=task_args,
        answer=args.answer,
        judge=args.judge,
        model_args=model_args,
        task_args_sha256=hashlib.sha256(args.task_args.read_bytes()).hexdigest(),
    )
    identity_path = args.root / "native-identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Run identity differs from existing output directory")
    identity_path.write_text(json.dumps(identity, indent=2))
    grader = get_model(args.judge, config=GenerateConfig(seed=0), **model_args["judge"])
    if args.phase == "answer":
        logs = inspect_eval(
            task,
            model=get_model(
                args.answer, config=GenerateConfig(seed=0), **model_args["answer"]
            ),
            model_roles={"grader": grader},
            score=False,
            max_connections=1,
            log_dir=str(args.root / "native"),
            display="none",
        )
        if logs[0].status != "success":
            raise RuntimeError(str(logs[0].error))
        (args.root / "answer-log.txt").write_text(logs[0].location)
    else:
        log = read_eval_log((args.root / "answer-log.txt").read_text().strip())
        if not task.scorer:
            raise ValueError("Task has no scorer")
        scored = score(
            log,
            task.scorer,
            metrics=task.metrics,
            model_roles={"grader": grader},
            display="none",
        )
        write_eval_log(scored, str(args.root / "scored.eval"))


if __name__ == "__main__":
    main()
