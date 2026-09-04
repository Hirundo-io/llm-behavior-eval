# STUDY_CONTRACT — Prompt-injection Purple Llama OFF/ON study (T4U)

Frozen 2026-09-04. GPU inference and judging have **not** been run. This
document is the single entry point; see the linked files for full detail and
evidence.

## 1. Population

**Primary, executable now:** `hirundo-io/prompt-injection-purple-llama`
@ `403abe13df3913940c065e5af6ca471c4fb7daf6`, 251 rows (`prompt_id` 0-250,
all unique). Full identity/hash manifest: `DATASET_IDENTITY.md`.

**Secondary, NOT executable from this freeze:** Bloom malicious/benign/
conflicting-signals (owned by PR #159). See `DATASET_IDENTITY.md` for the
integration order. Does not block or extend the primary study.

## 2. Four matched arms

1. Base Model — thinking OFF
2. Uncensored Model — thinking OFF
3. Base Model — thinking ON
4. Uncensored Model — thinking ON

"Base Model" = `Qwen/Qwen3.5-4B` (stock). "Uncensored Model" = same base +
the `v2c` LoRA adapter (rank 16 / alpha 32 / dropout 0.1, snapshot
`654d5acdd2eb_0`, safetensors SHA-256
`9b3158331c001e94046469059a8e8c59d4f2f2095f882cb528f87fb3e8c3e9a2`, MLflow
run `654d5acdd2eb4b1ba613cb980abdada8`) — the same two identities already
frozen elsewhere in RSCH-76 (Appendix A). Exact commands: `run_full.sh` /
`run_common.sh`.

## 3. Prior-run recovery

The commonly-repeated "~40% incomplete at 128 tokens" figure is **not
backed by any recoverable primary artifact** — full recovery report in
`PRIOR_RUN_RECOVERY.md`. The new cap below is chosen from real, locatable
precedent instead (also in that file), not from the unrecoverable figure and
not from guesswork.

## 4. Frozen generation cap

| Parameter | Frozen value |
| --- | --- |
| `max_answer_tokens` (all 4 arms) | **8192** |
| `max_model_len` (target and judge) | **16384** |

**Rationale for 8192, not simply copied from another benchmark:**

- `max_answer_tokens=128` (the current library default for the
  `"prompt-injection"` family) is rejected independent of the unrecoverable
  40% claim: it is a known-inadequate value for this exact model family on
  structurally similar free-text tasks (RSCH-71 moved off a much larger 512
  before this study existed).
- **8192 is independently, empirically converged-on twice** for this exact
  base model (Qwen3.5) on structurally similar free-text-generation-then-
  judge tasks: RSCH-71's probe (512 → 8192), and the CCPC-500 continuity run
  (8192 answer tokens inside a 16384 context, on the same `Qwen/Qwen3.5-4B`
  base, achieving **zero** incomplete/unknown/unparseable rows at 500/500).
  This is not "copying CCPC's number" — it is the same base model converging
  on the same figure from two independent evaluation efforts.
- A **known counter-signal exists and is disclosed, not hidden**: a Purple
  Llama-specific reproduction script in `hirundo-research`
  (`run_purple_llama_db_reproduction.sh`) runs Qwen3.5 thinking-ON prompt
  injection with `MAX_ANSWER_TOKENS=16384` and no `--max-model-len` cap (i.e.
  under a much larger native context). This means some thinking-ON
  completions on this exact dataset *can* need more than 8192 tokens of
  reasoning + answer. Doubling the answer budget to 16384 would require
  raising `max_model_len` well past 16384 to leave any room for the prompt —
  which contradicts the frozen `max_model_len=16384` contract "used
  successfully elsewhere" (CCPC) that this study also adopts.
- Given that conflict, **8192 is chosen and a non-zero
  `model_incomplete` rate under thinking-ON is accepted and predeclared as a
  possible, disclosed outcome** — not chased away by inflating the cap. This
  is the specific trade-off the brief asks to make explicit rather than
  silently resolve by picking whichever cap guarantees 100% completion.
- **This cap must not be raised after seeing full-run results.** If the
  smoke or full run shows `model_incomplete` under thinking-ON exceeding what
  is scientifically tolerable, that is a **finding to report**, and grounds
  to re-freeze a new study with a new cap and a fresh smoke — not grounds to
  edit this document post hoc.

`max_model_len=16384` is requested at the vLLM constructor for both the
target and judge engines (`PARAMETER_PLUMBING.md` #6/#10), but **no live
readback/hard-fail guard verifying vLLM actually honored it exists anywhere
in this codebase** — confirmed absent by direct audit, not merely
undocumented. This is a disclosed gap, not a silently-accepted one; see
`PARAMETER_PLUMBING.md`.

## 5. Judge identity

Primary judge: **`google/gemma-4-12b-it` via the vLLM backend** (PR #157's
already-merged text-only vLLM judge loading). Two divergent prior choices
were found and reconciled explicitly — full rationale in `JUDGE_IDENTITY.md`.
Judge prompt unchanged from the current evaluator; **not** hardened against
response-to-judge prompt injection in this freeze (disclosed gap, out of the
5-item patch scope).

## 6. Validity contract

Full rules in `VALIDITY_CONTRACT.md`. Summary: every row lands in exactly
one of `judged_yes`, `judged_no`, `model_incomplete`, `model_unknown`,
`judge_incomplete`, `judge_unknown`, `judge_unparseable`. No incomplete or
unparseable row is ever silently scored as a resistant/safe verdict. A fully
invalid cohort raises rather than reporting a fabricated accuracy.

## 7. Analysis contract

Full rules in `ANALYSIS_CONTRACT.md`. Summary: fixed-N accounting per arm,
a primary conditional attack-success metric explicitly labeled as
conditional, fixed-N bounds reported as an interval (never collapsed to a
point estimate), and target completeness treated as a first-class,
separately-reported behavior metric (Base vs Uncensored x OFF vs ON).

## 8. Evaluator basis

`origin/main` @ `5d96a95` + local commits `361842c` (dataset revision pin)
+ `83a814a` (fail-open/thinking/judge-validity fixes), on local branch
`rsch76-prompt-injection-offon-freeze`. Not pushed, no PR opened. Full
contract audit against PRs #145/#149/#157/#159/#161 in `EVALUATOR_IDENTITY.md`.

## 9. Parameter plumbing

Every parameter above traced from config to the actual `LLM(...)` /
`SamplingParams(...)` call site, with disclosed gaps (no model-revision pin,
no `max_model_len` guard, `repetition_penalty` hard-pinned to `1.0`, no LoRA
rank/identity assertion in-repo, `enable_thinking` is a soft best-effort) in
`PARAMETER_PLUMBING.md`.

## 10. Smoke

`run_smoke.sh` (not run). Exercises all four arms on 8 rows each, LoRA
loading + checksum verification, `max_model_len` request, final-answer
extraction, `model_incomplete`/`judge_unparseable` handling, raw-evidence
persistence, and clean teardown. Full plan and expected GPU work units in
`SMOKE_PLAN.md`. Smoke results cannot change any value in this document.

## 11. Do-not-change-after-outcomes list

- `max_answer_tokens` (8192), `max_model_len` (16384)
- Judge identity (`google/gemma-4-12b-it`, vLLM backend)
- Dataset revision (`403abe13df3913940c065e5af6ca471c4fb7daf6`)
- Sampling settings (`seed=42`, `do_sample=False`/greedy, `top_p=1.0`,
  `top_k=0`, `repetition_penalty=1.0`)
- The validity contract's status vocabulary and judged-only accuracy
  denominator
- The evaluator basis commits (`361842c`, `83a814a`)

Any change to the above after GPU outputs exist requires a new freeze
document, not an edit to this one.
