"""Validate recorded production settings without inferring them from scores.

Expected settings are read from each benchmark's parity.yaml, the only place
parity inference settings live.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from inspect_ai.log import read_eval_log
from inspect_ai.model import GenerateConfig

from scripts.parity.compare import digest
from scripts.parity.parity_models import expected_request_settings

FAMILIES = {
    "bbq",
    "bloom",
    "unqover",
    "halueval",
    "medhallu",
    "prompt_injection",
    "ccpc",
    "xstest",
    "orbench",
}


def audit_settings(
    family: str,
    legacy: dict[str, Any],
    native: list[tuple[str, GenerateConfig]],
) -> dict[str, Any]:
    """Check retained configuration and every recorded native model request.

    Expected values come from the benchmark's parity.yaml: the retained legacy
    run must have used its budgets and thinking setting, and every native
    request must carry exactly its per-role settings plus the capture fields.
    """
    if family not in FAMILIES:
        raise ValueError("Unknown production family")
    expected_native = expected_request_settings(family)
    answer = expected_native["answer"]
    judge = expected_native["judge"]
    thinking = answer["extra_body"]["chat_template_kwargs"]["enable_thinking"]
    expected_legacy = dict(
        max_answer_tokens=answer["max_tokens"],
        max_judge_tokens=judge["max_tokens"],
        batch_size=1,
        judge_batch_size=1,
        sample=False,
        sample_judge=False,
        use_4bit=False,
        use_4bit_judge=False,
        enable_thinking=thinking,
        inference_engine="vllm",
    )
    for key, expected in expected_legacy.items():
        if legacy.get(key) != expected:
            raise ValueError(f"Retained production setting differs: {key}")
    if legacy.get("sampling_config", {}).get("seed") != answer["seed"]:
        raise ValueError("Retained request seed differs")
    counts = dict(answer=0, judge=0)
    for role, config in native:
        if role not in counts:
            raise ValueError("Unknown model-event role")
        if config.max_connections != 1:
            raise ValueError(f"Native {role} setting differs: max_connections")
        body = dict(config.extra_body or {})
        if body.pop("return_token_ids", None) is not True:
            raise ValueError(f"Native {role} does not request observed token IDs")
        body.pop("cache_salt", None)
        if role == "judge" and "truncate_prompt_tokens" in body:
            raise ValueError("Native judge must not carry an answer prompt cap")
        expected = expected_native[role]
        kwargs = body.get("chat_template_kwargs", {})
        if kwargs != expected["extra_body"]["chat_template_kwargs"]:
            raise ValueError(f"Native {role} thinking setting differs")
        actual = config.model_dump(exclude_none=True)
        actual.pop("max_connections", None)
        actual["extra_body"] = body
        for key in sorted(set(actual) | set(expected)):
            if key == "extra_body":
                for name in sorted(set(body) | set(expected["extra_body"])):
                    if body.get(name) != expected["extra_body"].get(name):
                        raise ValueError(f"Native {role} setting differs: {name}")
            elif actual.get(key) != expected.get(key):
                raise ValueError(f"Native {role} setting differs: {key}")
        counts[role] += 1
    if not counts["answer"]:
        raise ValueError("No recorded answer requests")
    return dict(
        legacy=expected_legacy,
        request_seed=answer["seed"],
        native_requests_checked=counts,
        answer_prompt_cap=answer["extra_body"]["truncate_prompt_tokens"],
        judge_prompt_cap=None,
        judge_thinking=False,
        parity_config=f"llm_behavior_eval/{family}/parity.yaml",
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
