"""Validate recorded production settings without inferring them from scores."""

import argparse
import json
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log
from inspect_ai.model import GenerateConfig

from scripts.parity.compare import digest


def audit_settings(
    family: str,
    legacy: dict[str, Any],
    native: list[tuple[str, GenerateConfig]],
) -> dict[str, Any]:
    """Check retained configuration and every recorded native model request."""
    if family not in {
        "bbq",
        "bloom",
        "unqover",
        "halueval",
        "medhallu",
        "prompt_injection",
        "ccpc",
        "xstest",
        "orbench",
    }:
        raise ValueError("Unknown production family")
    refusal = family in {"xstest", "orbench"}
    answer_budget = 2048 if refusal else 8192 if family == "ccpc" else 128
    judge_budget = 128 if refusal else 4096 if family == "ccpc" else 32
    expected_legacy = dict(
        max_answer_tokens=answer_budget,
        max_judge_tokens=judge_budget,
        batch_size=1,
        judge_batch_size=1,
        sample=False,
        sample_judge=False,
        use_4bit=False,
        use_4bit_judge=False,
        enable_thinking=refusal,
        inference_engine="vllm",
    )
    for key, expected in expected_legacy.items():
        if legacy.get(key) != expected:
            raise ValueError(f"Retained production setting differs: {key}")
    if legacy.get("sampling_config", {}).get("seed") != 42:
        raise ValueError("Retained request seed differs")
    counts = dict(answer=0, judge=0)
    for role, config in native:
        if role not in counts:
            raise ValueError("Unknown model-event role")
        expected = dict(
            max_tokens=answer_budget if role == "answer" else judge_budget,
            temperature=0,
            seed=42,
            max_connections=1,
        )
        for key, value in expected.items():
            if getattr(config, key) != value:
                raise ValueError(f"Native {role} setting differs: {key}")
        body = config.extra_body or {}
        thinking = body.get("chat_template_kwargs", {}).get("enable_thinking")
        if thinking is not (refusal if role == "answer" else False):
            raise ValueError(f"Native {role} thinking setting differs")
        if body.get("return_token_ids") is not True:
            raise ValueError(f"Native {role} does not request observed token IDs")
        if body.get("add_special_tokens") is not True:
            raise ValueError(f"Native {role} special-token setting differs")
        if role == "answer" and body.get("truncate_prompt_tokens") != 1024:
            raise ValueError("Native answer prompt cap differs")
        if role == "judge" and "truncate_prompt_tokens" in body:
            raise ValueError("Native judge must not carry an answer prompt cap")
        counts[role] += 1
    if not counts["answer"]:
        raise ValueError("No recorded answer requests")
    return dict(
        legacy=expected_legacy,
        request_seed=42,
        native_requests_checked=counts,
        answer_prompt_cap=1024,
        judge_prompt_cap=None,
        judge_thinking=False,
        scope="Retained run configuration and actual native model-event configs",
    )


def audit_case(root: Path) -> dict[str, Any]:
    """Bind the settings audit to immutable evidence-file hashes."""
    identity = json.loads((root / "identity-config.json").read_text())
    paths = list((root / "legacy").rglob("run_config.json"))
    if len(paths) != 1:
        raise ValueError("Missing or ambiguous retained run configuration")
    legacy = json.loads(paths[0].read_text())["evaluation_config"]
    log = read_eval_log(str(root / "scored.eval"))
    configs = []
    for sample in log.samples or []:
        events = [event for event in sample.events if event.event == "model"]
        if not events:
            raise ValueError(f"Missing actual model events: {sample.id}")
        configs.extend(
            ("answer" if index == 0 else "judge", event.config)
            for index, event in enumerate(events)
        )
    return dict(
        case=identity["case"],
        evidence_root=str(root.resolve()),
        scored_log_sha256=digest(root / "scored.eval"),
        retained_config_sha256=digest(paths[0]),
        source_manifest_sha256=digest(root / "source.json"),
        settings=audit_settings(identity["case"].split(":")[0], legacy, configs),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for root in args.roots:
        try:
            rows.append({**audit_case(root), "verified": True})
        except (OSError, ValueError, KeyError) as error:
            rows.append(dict(evidence_root=str(root), verified=False, error=str(error)))
    with args.output.open("x") as output:
        json.dump(
            dict(
                scope="Settings audit only; paired outputs and metrics require strict reports",
                cases=rows,
            ),
            output,
            indent=2,
        )
    if not all(row["verified"] for row in rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
