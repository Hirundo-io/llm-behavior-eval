#!/usr/bin/env python3
"""Zero-GPU (and optional live-GPU) identity/runtime preflight for the T4U
prompt-injection Purple Llama OFF/ON study.

This is study-level infrastructure, not a change to llm-behavior-eval: it
enforces the frozen RSCH-76 identity (model/tokenizer/judge/adapter/dataset
revisions, thinking-template support, and -- only when explicitly asked for,
since it requires vLLM + a GPU -- a live max_model_len readback) *before*
`run_smoke.sh`/`run_full.sh` invoke the evaluator CLI.

Every check hard-fails (raises SystemExit(1) with a clear message) rather
than warning and continuing. Run with the same virtualenv as the
llm-behavior-eval checkout (it needs `huggingface_hub`/`transformers`, and,
only for --check live-max-model-len, `vllm`).

Usage:
    python preflight.py static                 # everything that needs no GPU
    python preflight.py resolve-model <repo_id> <revision>
    python preflight.py live-max-model-len ...  # requires vllm + GPU; NOT run by this freeze
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

FROZEN = {
    "dataset_id": "hirundo-io/prompt-injection-purple-llama",
    "dataset_revision": "403abe13df3913940c065e5af6ca471c4fb7daf6",
    "base_model_id": "Qwen/Qwen3.5-4B",
    "base_model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "judge_model_id": "google/gemma-4-12B-it",
    "judge_model_revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    # RESOLVED: ab5c9beb854884db6c9c44675a2ec1c5a15c8a6e1cd2c173faac2647b6e6c74c
    # is the canonical v2c adapter SHA-256, confirmed by directly hashing the
    # real artifact at /Users/ilana/repos/artifacts/rsch-76/v2c-adapter-654d5acdd2eb_0/
    # and corroborated independently by two unrelated reports (T3E_REPORT.md,
    # T4O_REPORT.md). The previously-cited 9b3158331c... (Notion) is a stale
    # transcription error that never matched a real file -- see
    # UNCENSORED_MODEL_ADAPTER_IDENTITY.md. `verify-adapter` still requires
    # --expected-sha256 explicitly (no silent default at the CLI layer) as
    # good hygiene, but `run_common.sh` now defaults UNCENSORED_LORA_SHA256
    # to this resolved value.
    "adapter_sha256": "ab5c9beb854884db6c9c44675a2ec1c5a15c8a6e1cd2c173faac2647b6e6c74c",
    "adapter_rank": 16,
    "adapter_lora_alpha": 16,  # corrected from an earlier, wrong "32"
    "max_model_len": 16384,
    "max_answer_tokens": 8192,
    "max_judge_tokens": 32,
}


class PreflightError(RuntimeError):
    """Raised on any identity/runtime mismatch. Callers must hard-fail."""


@dataclass
class PreflightRecord:
    """Everything persisted into run_config for a verified preflight run."""

    checks: dict = field(default_factory=dict)

    def add(self, name: str, value: object) -> None:
        self.checks[name] = value

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.checks, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_snapshot(
    repo_id: str, revision: str, allow_patterns: list[str] | None = None
) -> Path:
    """Resolve `repo_id`@`revision` to an immutable local snapshot path.

    Hard-fails if the resolved snapshot's own directory name (the resolved
    commit) does not equal the requested revision -- i.e. this proves the
    *exact* pinned commit was fetched, not just "some snapshot of repo_id".
    """
    from huggingface_hub import snapshot_download

    local_path = Path(
        snapshot_download(
            repo_id=repo_id, revision=revision, allow_patterns=allow_patterns
        )
    )
    resolved_commit = local_path.name
    if resolved_commit != revision:
        raise PreflightError(
            f"{repo_id}: resolved snapshot commit {resolved_commit!r} != "
            f"frozen revision {revision!r}"
        )
    return local_path


def verify_lora_adapter(
    adapter_dir: Path, expected_sha256: str, expected_rank: int
) -> dict:
    """Hard-fail unless the adapter's own weight file and rank match the freeze.

    Never trusts a value recorded in Markdown -- always re-reads the
    canonical local artifact directly.
    """
    weight_file = adapter_dir / "adapter_model.safetensors"
    config_file = adapter_dir / "adapter_config.json"
    if not weight_file.is_file():
        raise PreflightError(f"missing {weight_file}")
    if not config_file.is_file():
        raise PreflightError(f"missing {config_file}")

    actual_sha256 = sha256_file(weight_file)
    if actual_sha256 != expected_sha256:
        raise PreflightError(
            f"adapter SHA-256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256} (from {weight_file})"
        )

    config = json.loads(config_file.read_text())
    actual_rank = config.get("r")
    if actual_rank != expected_rank:
        raise PreflightError(
            f"adapter rank mismatch: expected {expected_rank}, "
            f"got {actual_rank!r} (from {config_file})"
        )

    return {
        "adapter_dir": str(adapter_dir),
        "adapter_sha256": actual_sha256,
        "adapter_rank": actual_rank,
        "lora_alpha": config.get("lora_alpha"),
        "lora_dropout": config.get("lora_dropout"),
    }


def verify_thinking_template(
    tokenizer,
    *,
    enable_thinking_arg_name: str = "enable_thinking",
    thinking_start_token: str = "<think>",  # noqa: S107 -- a template token, not a secret
    thinking_end_token: str = "</think>",  # noqa: S107 -- a template token, not a secret
    probe_message: str = "Hello",
) -> dict:
    """Prove `enable_thinking` actually reaches `apply_chat_template` and
    changes the rendered prompt in the expected direction, rather than
    trusting the string-sniffing heuristic in `util_functions.py` as proof.

    Hard-fails if the tokenizer's chat template does not literally contain
    the configured kwarg name, or if toggling it produces an identical
    rendered prompt (a silent no-op).
    """
    template = tokenizer.chat_template or ""
    if enable_thinking_arg_name not in template:
        raise PreflightError(
            f"pinned chat template does not contain {enable_thinking_arg_name!r}; "
            "enable_thinking would silently no-op for this tokenizer"
        )

    conv = [{"role": "user", "content": probe_message}]
    rendered_on = tokenizer.apply_chat_template(
        conv,
        tokenize=False,
        add_generation_prompt=True,
        **{enable_thinking_arg_name: True},
    )
    rendered_off = tokenizer.apply_chat_template(
        conv,
        tokenize=False,
        add_generation_prompt=True,
        **{enable_thinking_arg_name: False},
    )

    if rendered_on == rendered_off:
        raise PreflightError(
            "enable_thinking=True and enable_thinking=False rendered an "
            "identical prompt -- the control is a silent no-op for this "
            "tokenizer/template"
        )

    # OFF must render a *closed* thinking block with nothing to reason about
    # (the target-side incomplete-thinking rule in base_evaluator.py assumes
    # this: a genuinely-open, unclosed thinking block signals incompleteness).
    off_has_closed_block = (
        thinking_start_token in rendered_off and thinking_end_token in rendered_off
    )

    return {
        "enable_thinking_arg_name": enable_thinking_arg_name,
        "template_contains_kwarg": True,
        "rendered_on_tail": rendered_on[-120:],
        "rendered_off_tail": rendered_off[-120:],
        "on_differs_from_off": True,
        "off_renders_closed_thinking_block": off_has_closed_block,
    }


def verify_max_model_len_live(model_path: str, max_model_len: int, **llm_kwargs) -> int:
    """Construct a throwaway vLLM engine with the frozen config and read back
    the engine's own resolved `max_model_len`, hard-failing on any mismatch.

    REQUIRES vllm + a GPU. Not run by this freeze (see the "live" subcommand
    guard in main()). This is deliberately a *study-level* preflight -- a
    disposable engine built with the exact same arguments the real run will
    use -- rather than a change to llm-behavior-eval, since the real run's
    engine object is never exposed outside the library's own process.
    """
    from vllm import LLM

    llm = LLM(model=model_path, max_model_len=max_model_len, **llm_kwargs)
    try:
        # vLLM has moved this attribute across versions; try the known paths
        # rather than assuming one.
        model_config = getattr(llm, "llm_engine", llm).model_config
        actual = getattr(model_config, "max_model_len", None)
        if actual is None:
            raise PreflightError(
                "could not read back max_model_len from the constructed "
                "vLLM engine (no llm.llm_engine.model_config.max_model_len "
                "attribute) -- treat this as a hard failure, not a pass"
            )
        if actual != max_model_len:
            raise PreflightError(
                f"vLLM engine resolved max_model_len={actual}, "
                f"requested {max_model_len}"
            )
        return actual
    finally:
        del llm


def run_static_checks(output_path: Path) -> PreflightRecord:
    from transformers import AutoTokenizer

    from llm_behavior_eval.evaluation_utils.free_text_injection_evaluator import (
        FreeTextPromptInjectionEvaluator,
    )
    from llm_behavior_eval.evaluation_utils.util_functions import (
        safe_apply_chat_template,
    )

    record = PreflightRecord()

    # 1. Dataset revision + row-level identity (re-derives DATASET_IDENTITY.md,
    #    does not just trust it).
    from datasets import load_dataset

    ds = load_dataset(FROZEN["dataset_id"], revision=FROZEN["dataset_revision"])[
        "train"
    ]
    if len(ds) != 251:
        raise PreflightError(f"expected 251 rows, got {len(ds)}")
    record.add(
        "dataset",
        {
            "dataset_id": FROZEN["dataset_id"],
            "revision": FROZEN["dataset_revision"],
            "row_count": len(ds),
        },
    )

    # 2. Base model + tokenizer immutable snapshot.
    base_path = resolve_snapshot(FROZEN["base_model_id"], FROZEN["base_model_revision"])
    base_tokenizer = AutoTokenizer.from_pretrained(
        FROZEN["base_model_id"], revision=FROZEN["base_model_revision"]
    )
    record.add(
        "base_model",
        {
            "repo_id": FROZEN["base_model_id"],
            "revision": FROZEN["base_model_revision"],
            "local_snapshot_path": str(base_path),
        },
    )

    # 3. Thinking-template preflight on the pinned base tokenizer.
    thinking_probe = verify_thinking_template(base_tokenizer)
    record.add("base_model_thinking_preflight", thinking_probe)

    # 4. Judge model + tokenizer immutable snapshot (tokenizer/config only --
    #    full weights are resolved at actual launch time via the same pinned
    #    revision; see JUDGE_IDENTITY.md).
    judge_path = resolve_snapshot(
        FROZEN["judge_model_id"],
        FROZEN["judge_model_revision"],
        allow_patterns=["tokenizer*", "*.json"],
    )
    judge_tokenizer = AutoTokenizer.from_pretrained(
        FROZEN["judge_model_id"], revision=FROZEN["judge_model_revision"]
    )
    record.add(
        "judge_model",
        {
            "repo_id": FROZEN["judge_model_id"],
            "revision": FROZEN["judge_model_revision"],
            "local_snapshot_path": str(judge_path),
            "note": "tokenizer/config only in this preflight; full weights "
            "resolved at launch time under the same pinned revision",
        },
    )

    # 5. Judge prompt hash + parser identity (re-derives, does not trust).
    template = FreeTextPromptInjectionEvaluator.PROMPT_JUDGE_RESPONSE
    template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
    rendered_judge_prompt = safe_apply_chat_template(
        judge_tokenizer,
        [
            {
                "role": "user",
                "content": template.format(
                    llm_response="I can't answer",
                    question="Does this response contain the word 'Piano'?",
                ),
            }
        ],
    )
    parser_probe = FreeTextPromptInjectionEvaluator._map_judge_outputs_yes_no(
        [
            [{"generated_text": "Yes", "finish_reason": "stop"}],
            [{"generated_text": "No.", "finish_reason": "stop"}],
            [{"generated_text": "I refuse.", "finish_reason": "stop"}],
        ]
    )
    labels, _raw, _finish = parser_probe
    if labels != ["Yes", "No", None]:
        raise PreflightError(f"judge parser identity check failed: {labels}")
    record.add(
        "judge_prompt",
        {
            "template_sha256": template_hash,
            "rendered_probe_tail": rendered_judge_prompt[-200:],
            "max_judge_tokens": FROZEN["max_judge_tokens"],
            "parser_identity_check": "pass",
        },
    )

    # 6. Judge template's own thought-channel behavior, under the exact
    #    invocation `_judge_batch` actually uses (no thinking kwarg at all).
    #    Disclosed finding, not currently gated: see STUDY_CONTRACT.md.
    default_render = safe_apply_chat_template(
        judge_tokenizer, [{"role": "user", "content": "probe"}]
    )
    record.add(
        "judge_template_thought_channel_note",
        {
            "rendered_tail_under_current_no_kwarg_invocation": default_render[-80:],
            "observation": (
                "gemma-4-12B-it's chat template always primes assistant "
                "turns with a channel marker; under this evaluator's actual "
                "invocation (no thinking kwarg passed) it renders as a "
                "closed, empty '<|channel>thought\\n<channel|>' block -- "
                "i.e. the model is not primed to produce reasoning content "
                "before its answer. Not independently verified against a "
                "live model completion (requires GPU); flagged as a smoke "
                "verification item in SMOKE_PLAN.md."
            ),
        },
    )

    record.add("frozen_parameters", FROZEN)
    record.write(output_path)
    return record


def cmd_static(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    try:
        record = run_static_checks(output_path)
    except PreflightError as exc:
        print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"All static preflight checks passed. Record written to {output_path}")
    print(json.dumps(record.checks, indent=2, sort_keys=True))


# RESOLVED, per UNCENSORED_MODEL_ADAPTER_IDENTITY.md: the real artifact at
# /Users/ilana/repos/artifacts/rsch-76/v2c-adapter-654d5acdd2eb_0/ hashes to
# ab5c9beb..., corroborated by T3E_REPORT.md and T4O_REPORT.md (both outside
# this repo). 9b3158331c... (Notion) never matched any real file and is a
# stale transcription error -- kept here only so its provenance/rejection is
# not lost.
_REJECTED_STALE_ADAPTER_SHA = {
    "9b3158331c001e94046469059a8e8c59d4f2f2095f882cb528f87fb3e8c3e9a2": (
        "Notion Appendix A/B -- claimed source (the CCPC-500 v2c continuity "
        "run) actually recorded ab5c9beb... in its own T3E_REPORT.md; never "
        "matched any real adapter_model.safetensors found on this machine."
    ),
}


def cmd_verify_adapter(args: argparse.Namespace) -> None:
    if args.expected_sha256 is None:
        print(
            "PREFLIGHT FAILED: --expected-sha256 is required and has no "
            "default at the CLI layer (good hygiene, even though "
            f"FROZEN['adapter_sha256'] = {FROZEN['adapter_sha256']!r} is "
            "resolved -- see UNCENSORED_MODEL_ADAPTER_IDENTITY.md). Pass "
            "that resolved value explicitly, after confirming it against "
            "whatever local artifact you're actually verifying.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if args.expected_sha256 in _REJECTED_STALE_ADAPTER_SHA:
        print(
            f"PREFLIGHT FAILED: {args.expected_sha256} is a known-stale "
            f"value: {_REJECTED_STALE_ADAPTER_SHA[args.expected_sha256]}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    try:
        result = verify_lora_adapter(
            Path(args.adapter_dir), args.expected_sha256, FROZEN["adapter_rank"]
        )
    except PreflightError as exc:
        print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print("Adapter verified:")
    print(json.dumps(result, indent=2, sort_keys=True))


def cmd_resolve_model(args: argparse.Namespace) -> None:
    """Resolve repo_id@revision to an immutable local snapshot path and print
    it (and nothing else) to stdout, for `local_path=$(preflight.py
    resolve-model ...)` capture from the wrapper scripts. Hard-fails (exit 1,
    message on stderr) if the resolved commit doesn't match."""
    try:
        path = resolve_snapshot(args.repo_id, args.revision)
    except PreflightError as exc:
        print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(str(path))


def cmd_live_max_model_len(args: argparse.Namespace) -> None:
    try:
        actual = verify_max_model_len_live(args.model_path, FROZEN["max_model_len"])
    except PreflightError as exc:
        print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"runtime_verified_max_model_len = {actual}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_static = sub.add_parser("static", help="Run every check that needs no GPU/vllm.")
    p_static.add_argument(
        "--output", default="preflight_record.json", help="Where to write the record."
    )
    p_static.set_defaults(func=cmd_static)

    p_adapter = sub.add_parser(
        "verify-adapter", help="Verify a materialized LoRA adapter directory."
    )
    p_adapter.add_argument("adapter_dir")
    p_adapter.add_argument(
        "--expected-sha256",
        default=None,
        help=(
            "Required, no default. See UNCENSORED_MODEL_ADAPTER_IDENTITY.md: "
            "two conflicting candidates exist and must be resolved against "
            "the real artifact before one is passed here."
        ),
    )
    p_adapter.set_defaults(func=cmd_verify_adapter)

    p_resolve = sub.add_parser(
        "resolve-model",
        help="Resolve repo_id@revision to a local snapshot path (stdout only).",
    )
    p_resolve.add_argument("repo_id")
    p_resolve.add_argument("revision")
    p_resolve.set_defaults(func=cmd_resolve_model)

    p_live = sub.add_parser(
        "live-max-model-len",
        help="REQUIRES vllm + GPU. Not run by this freeze.",
    )
    p_live.add_argument("model_path")
    p_live.set_defaults(func=cmd_live_max_model_len)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
