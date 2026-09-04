# Prior-run recovery report — the "~40% incomplete at 128 tokens" claim

## Verdict: NOT RECOVERABLE FROM PRIMARY EVIDENCE

The claim, repeated verbatim in commit messages `ca71c19` (branch
`t3o-prompt-injection-validity-fix`) and `83a814a` (this freeze's evaluator
basis) — "documented historically at ~40% of rows under the prior 128-token
ceiling" — is **not backed by any recoverable primary artifact**.

An exhaustive search covered:

- `llm-behavior-eval` and `hirundo-research` repo trees (excluding `.git`) for
  `responses.json`, `metrics.csv`, `summary_*.csv`, `run_config*.json`, or any
  prompt-injection log file.
- The full local HF cache and `~/.local/share` — only the raw
  `hirundo-io/prompt-injection-purple-llama` *input* dataset is cached; no
  `~/.local/share/llm-behavior-eval/results` output directory exists at all.
- `hirundo-research/mlruns` for any run tagged `prompt-injection` or `purple` —
  none found.
- Full `git log -p` history of `eval_config.py`'s `max_answer_tokens: 128`
  default for the `"prompt-injection"` family — no commit message or code
  comment near that default cites a measured truncation rate. The nearest
  related commit (`05d40f2`, "Revert PI judge-token default to 32") is about
  *judge*-side token budget, unrelated to the 40% claim.
- PR bodies/comments for #145 (LLM-116), #149 (LLM-119), #151/#154/#159
  (LLM-117), #121 (LLM-93, finish-reason logging), and #165 — none mentions a
  measured "40%" figure for prompt injection.

**Conclusion:** per this study's own validity contract, an unsubstantiated
claim must not silently become a frozen scientific fact. The new
`max_answer_tokens` cap below is therefore chosen from current, locatable,
primary evidence (Section "New cap" below) — not from the unrecoverable ~40%
figure, and not from guesswork.

## What *is* real and locatable

Two independent, real artifacts constrain the choice of `max_answer_tokens`
for this exact dataset/model family, found in `hirundo-research`:

1. **RSCH-71 probe** (`hirundo-bloom/scripts/eval/run_benign_pilot_no_thinking_qwen35.py`,
   branch `rsch-71-adjusting-bloom-for-research-purposes`; documented in the
   Notion "CCPC-Bench & Behavioral Unlearning" hub, page "Chinese Censorship
   Unlearning"): for a structurally identical pipeline shape (Qwen3.5, free-text
   generation, thinking OFF, Yes/No-style judge) on this same base model
   family, the answer-token budget was moved from **512 → 8192** as a
   deliberate, evidence-driven correction — not the default 128, and not
   chosen post-hoc to force completeness (it predates this study).
2. **Purple Llama DB reproduction gate**
   (`hirundo-bloom/scripts/eval/run_purple_llama_db_reproduction.sh`, found
   under two `hirundo-research` worktrees): runs
   `llm-behavior-eval Qwen/Qwen3.5-4B prompt-injection --thinking-on
   --max-answer-tokens "${MAX_ANSWER_TOKENS:-16384}"` directly against
   `hirundo-io/prompt-injection-purple-llama` (the exact dataset pinned in
   `DATASET_IDENTITY.md`), targeting reproduction of published ASR figures
   (28.29 → 9.16 for Qwen3.5-4B pretrained → hardened). No `--max-model-len`
   is set in that script, so its 16384-token answer budget ran under
   whatever the tokenizer's native context is (materially larger than
   16384) — it is evidence that Qwen3.5 thinking-ON reasoning on this
   dataset *can* run long, not evidence that 16384 total context is
   sufficient to contain it.

Neither artifact contains logged `finish_reason` tallies — both are
configuration precedent, not measured completion-length distributions. No
finer-grained evidence (e.g. a token-length histogram) was found.

## New cap freeze — see `STUDY_CONTRACT.md` for the final decision

`max_answer_tokens=128` (still the current default in `eval_config.py` for
the `"prompt-injection"` family) is rejected for this study on the evidence
above, independent of the unrecoverable 40% claim: it is a known-inadequate
value for this model family on structurally similar free-text tasks (RSCH-71
moved off it before this study existed), and the code-level fail-open bug
this freeze's evaluator basis (`83a814a`) fixes would otherwise still
silently mis-score any row it truncates.
