# STUDY_CONTRACT — Prompt-injection Purple Llama OFF/ON study (T4U)

Frozen 2026-09-04. GPU inference and judging have **not** been run. This
document is the single entry point; see the linked files for full detail and
evidence.

## STATUS

**Base Model arms (OFF/ON): preflight-verified, ready for GPU smoke.**
**Uncensored Model arms (OFF/ON): BLOCKED.** Two conflicting, unverifiable
SHA-256 values were found for the v2c LoRA adapter (`9b3158331c...` in
Notion vs `ab5c9beb...` in an untracked local script), and no real
`adapter_model.safetensors` file exists anywhere on this machine to resolve
which is correct. See `UNCENSORED_MODEL_ADAPTER_IDENTITY.md`. The wrappers
(`run_common.sh`) hard-block on this by design — `UNCENSORED_LORA_SHA256` has
no default and must be set explicitly, only after independent verification
against the real artifact. **Do not run any Uncensored Model arm until this
is resolved.**

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

"Base Model" = `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
(stock, resolved to an immutable local snapshot path -- see
`PARAMETER_PLUMBING.md`). "Uncensored Model" = same base + the `v2c` LoRA
adapter (rank 16 / alpha 32 / dropout 0.1, snapshot `654d5acdd2eb_0`, MLflow
run `654d5acdd2eb4b1ba613cb980abdada8`) — **the exact safetensors SHA-256 is
currently ambiguous between two conflicting candidates; see the STATUS
banner above and `UNCENSORED_MODEL_ADAPTER_IDENTITY.md`.** Exact commands:
`run_full.sh` / `run_common.sh`.

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
target and judge engines (`PARAMETER_PLUMBING.md` #6/#10). **llm-behavior-eval
itself has no live readback/hard-fail guard** verifying vLLM actually
honored it — confirmed absent by direct audit, not merely undocumented, and
not fixed in the library for this freeze (per the "prefer a thin wrapper"
principle). Instead, `run_common.sh` calls a **study-level preflight**
(`preflight.py live-max-model-len`) before every arm and before the judge is
used: it constructs its own disposable vLLM engine with the identical
`(model, max_model_len=16384)` pair, reads back
`llm.llm_engine.model_config.max_model_len`, and hard-fails if it isn't
exactly 16384, persisting `runtime_verified_max_model_len = 16384` on
success. This requires vLLM + a GPU and has **not** been run in this
freeze (no GPU here) — but its assertion logic was verified against a mock
engine object (both "honored" and "silently clamped" cases) and behaves
correctly in both. See `PARAMETER_PLUMBING.md` for the enforcement-mechanism
classification.

## 4b. Sampling contract (explicit, all four arms)

| Parameter | Frozen value | Why |
| --- | --- | --- |
| `seed` | `42` | Library default; `DatasetConfig.seed` and `SamplingConfig.seed` both resolve to it via the same CLI `--seed`, so the latent `dataset_config.seed or sampling_config.seed` falsy-zero bug in `base_evaluator.py` is out of scope for this study (it only bites an explicit `seed=0`, which this study never sets) -- not fixed here, deliberately not broadened. |
| `do_sample` / `temperature` | `do_sample=False` (`--no-sample`); greedy | Implemented as `temperature=0.0`, confirmed at `vllm_eval_engine.py:184-192,198`. Matches the CCPC-500 continuity run's `T=0` convention on the same base model. |
| `top_p` / `top_k` | `1.0` / `0` | Library defaults, frozen explicitly rather than left implicit. |
| `min_p` | **Not set -- does not exist as a pipeline concept.** `SamplingConfig(min_p=...)` raises `pydantic.ValidationError` (fails loudly, not silently). This study does not need a non-default `min_p`, so no plumbing was added for it; if a future study needs one, the evaluator must be patched first, not worked around. |
| `repetition_penalty` | **`1.0`, explicitly recorded as the frozen choice.** Not inherited from ChinaBench's `1.10` -- no independent prompt-injection-specific evidence was found requiring a non-default value (`PRIOR_RUN_RECOVERY.md` found no recoverable prior-run evidence at all). `BaseEvaluator.generate_answers` (the path this evaluator uses) never forwards a non-default value to `SamplingParams`, so `1.0` is also the *only* value currently reachable for this evaluator family -- verified at `vllm_eval_engine.py:201`, not merely assumed. No new CLI plumbing was added, since `1.0` is the desired frozen value. |

## 5. Judge identity

Primary judge: **`google/gemma-4-12B-it` via the vLLM backend** (PR #157's
already-merged text-only vLLM judge loading). Two divergent prior choices
were found and reconciled explicitly — full rationale in `JUDGE_IDENTITY.md`.
Judge prompt unchanged from the current evaluator; **not** hardened against
response-to-judge prompt injection in this freeze (disclosed gap, out of the
5-item patch scope).

## 5b. Thinking-mode preflight (zero-GPU, actually run)

`preflight.py static` renders the pinned Qwen3.5 tokenizer's chat template
with `enable_thinking=True` and `enable_thinking=False` and asserts they
differ -- this was **actually executed** (not just asserted) on
2026-09-04, with no GPU needed:

- ON tail: `...<|im_start|>assistant\n<think>\n` (open thinking block)
- OFF tail: `...<|im_start|>assistant\n<think>\n\n</think>\n\n` (closed,
  empty thinking block)

This proves `enable_thinking` is not a silent no-op for the pinned
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` snapshot -- the heuristic
string-sniffing in `util_functions.py` is not relied on as proof; the actual
rendered template is. Full record: `preflight_record.json`.

A secondary finding surfaced by the same preflight: the judge model's own
chat template (`google/gemma-4-12B-it`) always primes assistant turns with a
`<|channel>thought\n<channel|>` marker, regardless of any thinking kwarg.
Under this evaluator's actual invocation (no thinking kwarg passed to the
judge at all), it renders as a **closed, empty** thought channel -- i.e. the
judge is not primed to produce reasoning before its answer. This was **not**
independently verified against a live judge completion (requires GPU) and
is flagged as a specific smoke-verification item in `SMOKE_PLAN.md`: confirm
the judge reliably returns a parseable Yes/No within the 32-token budget
rather than reopening a reasoning channel.

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

## 9. Parameter plumbing and enforcement

Every parameter traced from config to the actual `LLM(...)` /
`SamplingParams(...)` call site, now additionally classified by **how** it's
enforced (library-native / study-wrapper preflight / live runtime readback /
merely documented) in `PARAMETER_PLUMBING.md`. Remaining disclosed gaps:
`repetition_penalty` hard-pinned to `1.0` by the library (frozen value,
recorded above), and the judge repo id itself (not yet a pinned local
snapshot at the CLI layer -- `run_common.sh` resolves it for the static
preflight but the actual `--judge-model` argument passed to
`llm-behavior-eval` remains the mutable repo id, since the library has no
`--judge-revision` flag to receive a local path distinctly from the model
positional argument).

Mechanically enforced by the new `preflight.py` (study-level, not a library
change): dataset revision + row count, base model snapshot resolution
(hard-fails on commit mismatch), judge tokenizer/config snapshot resolution,
judge prompt hash + parser identity, thinking-template proof, LoRA
SHA-256/rank verification (**currently blocked pending adapter identity
resolution**, see below), and a live `max_model_len` readback via a
disposable vLLM engine (code complete, GPU-only, not run in this freeze).

## 10. Smoke

`run_smoke.sh` (not run; and additionally blocked for the Uncensored Model
arms -- see STATUS). `run_common.sh` now fails closed: sourcing it runs
`preflight.py static` unconditionally, resolves the base model to its pinned
local snapshot, and `run_arm` runs a live `max_model_len` check before every
`llm-behavior-eval` invocation. `verify_uncensored_adapter` refuses to run
at all while `UNCENSORED_LORA_SHA256` is unset. Full plan and expected GPU
work units in `SMOKE_PLAN.md`. Smoke results cannot change any value in this
document.

## 11. Do-not-change-after-outcomes list

- `max_answer_tokens` (8192), `max_model_len` (16384)
- Judge identity (`google/gemma-4-12B-it`, vLLM backend)
- Dataset revision (`403abe13df3913940c065e5af6ca471c4fb7daf6`)
- Sampling settings (`seed=42`, `do_sample=False`/greedy, `top_p=1.0`,
  `top_k=0`, `repetition_penalty=1.0`)
- The validity contract's status vocabulary and judged-only accuracy
  denominator
- The evaluator basis commits (`361842c`, `83a814a`)

Any change to the above after GPU outputs exist requires a new freeze
document, not an edit to this one. The Uncensored Model adapter SHA-256 is
explicitly **not yet frozen** (see STATUS) — resolving it is not covered by
this rule; it must be resolved *before* any Uncensored Model GPU output
exists, not changed after.
