"""Bounded upstream-parser/Inspect/scorer integration; no per-model parser code.

Parser and effort are explicit operator inputs. This is not a benchmark matrix
or a qualification of strict thinking-off tasks. Run answer and judge in separate
processes to release GPU memory between roles.
"""

import argparse
import asyncio
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from inspect_ai import Task, eval_async, score_async
from inspect_ai.dataset import MemoryDataset
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.solver import generate

from llm_behavior_eval.xstest.scorer import scorer
from llm_behavior_eval.xstest.task import xstest
from scripts.parity.capture import capture_judge_responses
from scripts.parity.compare import digest


def is_unsupported_effort(rejection: dict[str, Any]) -> bool:
    """Recognize an actual server rejection, preserving its full error evidence."""
    error = rejection.get("error", "")
    return (
        rejection.get("rejected") is True
        and "400" in error
        and "reasoning_effort" in error
        and any(text in error for text in ("not supported", "does not support"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["answer", "judge", "report"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--answer", required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--reasoning-parser", required=True)
    parser.add_argument("--reasoning-effort", required=True)
    parser.add_argument("--rejected-effort", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("VLLM_USE_AOT_COMPILE", "0")
    os.environ.setdefault("VLLM_SYSTEM_START_DATE", "2026-11-09")
    settings = dict(
        answer=args.answer,
        judge=args.judge,
        reasoning_parser=args.reasoning_parser,
        reasoning_effort=args.reasoning_effort,
        rejected_effort=args.rejected_effort,
        system_start_date=os.environ["VLLM_SYSTEM_START_DATE"],
        scope="Three actual generations plus one rejected-effort request; provider/scorer integration only",
        parser_selection="Explicit user-supplied upstream parser via Inspect model args; no package dispatch",
        dataset_sha256={p.name: digest(p) for p in args.dataset.glob("*.parquet")},
        answer_metadata_sha256={
            name: digest(Path(args.answer) / name)
            for name in ("config.json", "chat_template.jinja", "tokenizer_config.json")
            if (Path(args.answer) / name).exists()
        },
        answer_quantization=json.loads(
            (Path(args.answer) / "config.json").read_text()
        ).get("quantization_config"),
        activation_dtype="bfloat16",
    )
    settings_path = root / "settings.json"
    if args.phase == "answer":
        with settings_path.open("x") as stream:
            json.dump(settings, stream, indent=2)
    elif json.loads(settings_path.read_text()) != settings:
        raise ValueError("Probe configuration changed between phases")

    def model(path: str, *, answer: bool = False):
        return get_model(
            "vllm/" + path,
            config=GenerateConfig(
                temperature=0,
                seed=42,
                max_connections=1,
                extra_body=dict(
                    return_token_ids=True,
                    add_special_tokens=True,
                    **({"reasoning_effort": args.reasoning_effort} if answer else {}),
                ),
            ),
            **({"reasoning_parser": args.reasoning_parser} if answer else {}),
            max_model_len=16384,
            gpu_memory_utilization=0.5,
            enforce_eager=False,
            compilation_config='{"cudagraph_specialize_lora":false}',
            language_model_only=True,
            dtype="bfloat16",
            max_num_seqs=1,
            max_num_batched_tokens=16384,
            seed=0,
            host="127.0.0.1",
        )

    if args.phase == "answer":

        async def answers() -> None:
            answer = model(args.answer, answer=True)
            grader = model(args.judge)
            task = xstest(
                dataset_path=str(args.dataset),
                max_samples=100,
                compatibility_tokenizer=args.answer,
            )
            selected = [
                next(
                    s
                    for s in task.dataset
                    if s.metadata and s.metadata["label"] == label
                )
                for label in ("safe", "unsafe")
            ]
            task.dataset = MemoryDataset(selected)
            # The third case deliberately changes the budget, and is labelled a
            # diagnostic task rather than an unchanged production benchmark.
            truncated = Task(
                name="reasoning_truncation",
                dataset=MemoryDataset([deepcopy(selected[0])]),
                solver=generate(
                    max_tokens=8,
                    temperature=0,
                    extra_body=dict(
                        return_token_ids=True,
                        add_special_tokens=True,
                        reasoning_effort=args.reasoning_effort,
                        chat_template_kwargs={"enable_thinking": False},
                    ),
                ),
                scorer=scorer(),
            )
            for name, current in [("production", task), ("truncation", truncated)]:
                logs = await eval_async(
                    current,
                    model=answer,
                    model_roles={"grader": grader},
                    score=False,
                    max_samples=1,
                    max_connections=1,
                    log_dir=str(root / "logs"),
                )
                if len(logs) != 1 or logs[0].status != "success":
                    raise ValueError(f"{name}: native generation failed: {logs}")
                (root / f"{name}-answer.txt").write_text(logs[0].location)
            try:
                result = await answer.generate(
                    "Reply with OK.",
                    config=GenerateConfig(
                        max_tokens=1,
                        extra_body={"reasoning_effort": args.rejected_effort},
                    ),
                )
            except Exception as exc:
                rejection = dict(
                    rejected=True, error_type=type(exc).__name__, error=str(exc)
                )
            else:
                rejection = dict(rejected=False, output=result.model_dump(mode="json"))
            (root / "unsupported-effort.json").write_text(
                json.dumps(rejection, indent=2)
            )

        asyncio.run(answers())
    elif args.phase == "judge":

        async def judges() -> None:
            grader = model(args.judge)
            for name in ("production", "truncation"):
                log = read_eval_log((root / f"{name}-answer.txt").read_text().strip())
                with capture_judge_responses(
                    grader, root / f"{name}-judge-responses.json"
                ):
                    result = await score_async(
                        log, scorer(), model_roles={"grader": grader}
                    )
                write_eval_log(result, str(root / f"{name}-scored.eval"))

        asyncio.run(judges())
    else:
        rows = []
        for name in ("production", "truncation"):
            path = root / f"{name}-scored.eval"
            log = read_eval_log(str(path), resolve_attachments=True)
            for sample in log.samples or []:
                events = [e for e in sample.events if e.event == "model"]
                event = events[0]
                if not event.call or not event.call.response:
                    raise ValueError("Missing actual provider response")
                raw: Any = event.call.response
                raw_message = raw["choices"][0]["message"]
                raw_final = raw_message.get("content") or ""
                raw_reasoning = (
                    raw_message.get("reasoning", raw_message.get("reasoning_content"))
                    or ""
                )
                score = sample.scores["refusal_scorer"] if sample.scores else None
                if score is None or score.metadata is None:
                    raise ValueError("Missing production scorer observation")
                meta = score.metadata
                judge_events = events[1:]
                if len(judge_events) != 1:
                    raise ValueError("Expected one actual production grader call")
                actual_judge_input = judge_events[0].input[-1].text
                content = sample.output.choices[0].message.content
                blocks = (
                    [c.model_dump(mode="json") for c in content]
                    if isinstance(content, list)
                    else []
                )
                row: dict[str, Any] = dict(
                    case=name,
                    id=sample.id,
                    original_input=sample.input,
                    actual_response=raw,
                    output=sample.output.model_dump(mode="json"),
                    content_blocks=blocks,
                    scorer_answer=meta["llm_answer"],
                    judge_prompt=meta.get("judge_prompt"),
                    actual_judge_input=actual_judge_input,
                    actual_judge_input_equals_scorer_prompt=actual_judge_input
                    == meta.get("judge_prompt"),
                    judge_label=meta["judge_label"],
                    score_metadata=meta,
                    scored_log_sha256=digest(path),
                    final_text_reaches_scorer=meta["llm_answer"]
                    == sample.output.completion,
                    stop_reason=sample.output.choices[0].stop_reason,
                    requested_budget=event.config.max_tokens,
                    requested_enable_thinking=(event.config.extra_body or {})
                    .get("chat_template_kwargs", {})
                    .get("enable_thinking"),
                    upstream_final_equals_completion=raw_final
                    == sample.output.completion,
                    upstream_reasoning_captured=raw_reasoning
                    == "".join(
                        block["reasoning"]
                        for block in blocks
                        if block["type"] == "reasoning"
                    ),
                    has_final=bool(raw_final),
                    has_reasoning=bool(raw_reasoning),
                )
                probe = sample.metadata.get("compatibility", {}).get("prompt")
                if probe:
                    expected_ids = [
                        token
                        for token, keep in zip(
                            probe["input_ids"], probe["attention_mask"], strict=True
                        )
                        if keep
                    ]
                    row["production_metadata_matches_actual_prompt_ids"] = (
                        expected_ids == raw.get("prompt_token_ids")
                    )
                rows.append(row)
        rejection = json.loads((root / "unsupported-effort.json").read_text())
        result = dict(
            settings=settings,
            rows=rows,
            unsupported_effort=rejection,
            limitations="No old/new generation equality claim; strict thinking-off is not certified; effort low is not off",
            integration_checks=dict(
                actual_samples=len(rows) == 3,
                final_text_reaches_scorer=all(
                    r["final_text_reaches_scorer"] for r in rows
                ),
                actual_judge_input_matches_scorer_prompt=all(
                    r["actual_judge_input_equals_scorer_prompt"] for r in rows
                ),
                upstream_representation_preserved=all(
                    r["upstream_final_equals_completion"]
                    and r["upstream_reasoning_captured"]
                    for r in rows
                ),
                normal_final_observed=any(
                    r["case"] == "production"
                    and r["has_final"]
                    and r["stop_reason"] == "stop"
                    for r in rows
                ),
                reasoning_only_truncation_observed=any(
                    r["case"] == "truncation"
                    and r["has_reasoning"]
                    and not r["has_final"]
                    and r["stop_reason"] == "max_tokens"
                    and r["requested_budget"] == 8
                    for r in rows
                ),
                thinking_disable_ignored_observed=any(
                    r["case"] == "truncation"
                    and r["requested_enable_thinking"] is False
                    and r["has_reasoning"]
                    for r in rows
                ),
                unsupported_effort_rejected=is_unsupported_effort(rejection),
            ),
        )
        (root / "report.json").write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
