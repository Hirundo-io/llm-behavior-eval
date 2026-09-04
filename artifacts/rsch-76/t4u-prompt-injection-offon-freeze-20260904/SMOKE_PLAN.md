# Smoke plan

Script: `run_smoke.sh` (sources `run_common.sh`). **Not run by this freeze.**

## What it exercises

| Requirement | How |
| --- | --- |
| Base Model | `Qwen/Qwen3.5-4B`, no LoRA, 2 arms (OFF/ON) |
| Uncensored Model / LoRA | Same base + LoRA adapter at `$UNCENSORED_LORA_PATH`, 2 arms (OFF/ON); `verify_uncensored_adapter` hard-fails before either arm if the adapter's `sha256sum` doesn't match `9b3158331c001e94046469059a8e8c59d4f2f2095f882cb528f87fb3e8c3e9a2` |
| OFF / ON | `--thinking-off` vs `--thinking-on --enable-thinking-arg-name enable_thinking --thinking-start-token '<think>' --thinking-end-token '</think>' --exclude-thinking-trace-for-judge` |
| Target incomplete handling | Inspect `responses.json` for any `status in {model_incomplete, model_unknown}` rows -- confirms the fail-open fix from `83a814a` is active, not just present in source |
| Final-answer extraction | For thinking-ON arms, confirm judge input (loggable via `judge_raw`'s context, or by re-deriving from `llm_answer` + `_format_answers` logic) never contains a `<think>` tag |
| Judge path | Confirms `--judge-engine vllm --judge-model google/gemma-4-12b-it` actually loads and returns verdicts (not a `judge_unknown`-only run, which would indicate the judge engine failed to start) |
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
