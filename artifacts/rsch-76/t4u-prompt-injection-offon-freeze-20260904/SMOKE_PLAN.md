# Smoke plan

Script: `run_smoke.sh` (sources `run_common.sh`). **Not run by this freeze.**
All four arms are preflight-verified and unblocked — the Uncensored Model
adapter identity conflict is resolved (see
`UNCENSORED_MODEL_ADAPTER_IDENTITY.md`); `run_common.sh`'s
`verify_uncensored_adapter` still re-hashes the real artifact against the
resolved SHA-256 before every Uncensored Model arm.

## Gates that now run before any target inference (fail-closed)

Sourcing `run_common.sh` (which both `run_smoke.sh` and `run_full.sh` do
first) unconditionally runs `preflight.py static`: dataset revision + row
count, base model snapshot resolution, judge tokenizer/config snapshot
resolution, judge prompt hash + parser identity, and the thinking-template
proof. Any failure aborts the whole script (`set -euo pipefail` plus
`preflight.py`'s `SystemExit(1)`). `run_arm` additionally calls a live
`max_model_len` readback (`preflight.py live-max-model-len`, GPU-only, code
complete but not executed by this freeze) before every arm, and
`verify_uncensored_adapter` (also GPU-independent) before every Uncensored
Model arm.

## What it exercises

| Requirement | How |
| --- | --- |
| Base Model | `Qwen/Qwen3.5-4B` @ pinned local snapshot, no LoRA, 2 arms (OFF/ON) |
| Uncensored Model / LoRA | Same base + LoRA adapter at `$UNCENSORED_LORA_PATH`, 2 arms (OFF/ON); `verify_uncensored_adapter` hard-fails before either arm if the materialized adapter's SHA-256 doesn't match `ab5c9beb854884db6c9c44675a2ec1c5a15c8a6e1cd2c173faac2647b6e6c74c` (resolved — see `UNCENSORED_MODEL_ADAPTER_IDENTITY.md`) |
| OFF / ON | `--thinking-off` vs `--thinking-on --enable-thinking-arg-name enable_thinking --thinking-start-token '<think>' --thinking-end-token '</think>' --exclude-thinking-trace-for-judge` |
| Target incomplete handling | Inspect `responses.json` for any `status in {model_incomplete, model_unknown}` rows -- confirms the fail-open fix from `83a814a` is active, not just present in source |
| Final-answer extraction | For thinking-ON arms, confirm judge input (loggable via `judge_raw`'s context, or by re-deriving from `llm_answer` + `_format_answers` logic) never contains a `<think>` tag |
| Judge path | Confirms `--judge-engine vllm --judge-model google/gemma-4-12B-it` actually loads and returns verdicts (not a `judge_unknown`-only run, which would indicate the judge engine failed to start) |
| Judge thought-channel risk (new) | The judge's own chat template always primes a `<|channel>thought<channel|>` marker (see `STUDY_CONTRACT.md` §5b). Confirm empirically that real judge completions stay within `max_judge_tokens=32` and land in `judged_yes`/`judged_no` at a normal rate, not predominantly `judge_incomplete` (which would indicate the model is reopening a reasoning channel the 32-token budget can't afford) |
| Unparseable handling | If any `judge_unparseable` row appears, confirms `judge_raw` is non-empty and `judge_verdict` is `null` -- proves the fix persists evidence rather than defaulting to `judged_no` |
| Artifact persistence | `responses.json` and `metrics.csv` exist under each arm's output dir with every row carrying `llm_answer`, `finish_reason`, `judge_finish_reason`, `judge_raw`, `judge_verdict`, `status` |
| Clean teardown | Process exits cleanly after all four arms; no leaked vLLM worker processes (`ps` check) |

## Explicit constraint

Smoke outcomes cannot change any frozen scientific parameter (dataset
revision, model identity, judge identity, `max_answer_tokens`,
`max_model_len`, sampling settings, validity contract). If the smoke reveals
that a frozen parameter is actually wrong (e.g. `enable_thinking` turns out
to be a silent no-op for this tokenizer -- see `PARAMETER_PLUMBING.md` gap
#5), that is a **blocking finding to resolve and re-freeze**, not a value to
quietly patch and continue.

## Expected GPU work units

| | Target generations | Judge calls (upper bound; fewer if any row is target-incomplete) |
| --- | --- | --- |
| Smoke (4 arms x 8 rows) | 32 | 32 |
| Full (4 arms x 251 rows) | 1,004 | 1,004 |

Both target and judge generation are single-pass (no self-consistency/majority
voting, no retries beyond `run_judge_with_backoff`'s batch-size backoff on
OOM) — one target completion and, for target-valid rows, one judge call per
row per arm.
