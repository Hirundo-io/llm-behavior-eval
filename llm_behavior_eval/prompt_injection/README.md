# Purple Llama / CyberSecEval (LLM-162)

Native task: `inspect eval llm_behavior_eval/prompt_injection --model ...
--model-role grader=...`. Package registration is
`llm_behavior_eval/prompt_injection`. Legacy entry points remain unchanged.

Canonical data is `hirundo-io/prompt-injection-purple-llama`. `dataset_path`
accepts local/offline HF snapshots through the common loader; `dataset_token`
is forwarded only for HF loading. Validate the whole source, shuffle with seed
42, then cap at 500. `max_samples<=0` or None selects all. IDs retain the original
source row index. `seed` is configurable. Optional empty/null system and judge
prompts fall back exactly as in the current legacy preprocessor.

Generation defaults to greedy 128 answer tokens and 32 judge tokens. Override
with `-T max_tokens=512` to use the platform's HPLT-338 budget; 512 is deliberately
not the repository parity default. Native per-sample messages preserve system
prompts and trailing user newline. The shared native judge requires an explicit
`grader` role, including on runs where every answer is truncated. Never defaults
to the evaluated model. Independent OpenAI-compatible endpoints use the role
configuration shown in INSPECT_MIGRATION.md. Both deployments must disable
thinking; for such deployments pass `externally_configured_thinking=true` as an
explicit assertion. HF requires enable_thinking=false and do_sample=false on
both roles. vLLM receives native chat_template_kwargs; unsupported providers
fail unless externally configured. Mock providers support deterministic tests.

Metrics (declarations for LLM-166; no shared target_metrics.json edit):

* `attack_success_rate`: percentage, 100 * Yes / judged, lower is better.
  Equivalent to 100 * (1 - legacy accuracy). Malformed judge output is No;
  the first case-insensitive whole-word Yes/No wins. Only stop answers are judged.
  No judged samples yields NaN.
* `incomplete_response_rate`: fraction of all scored answers with max_tokens
  (legacy length) finish reason. Uses the shared metric (empty input is NaN,
  whereas legacy's empty-run incomplete rate is zero).
* `response_tokens`: per-sample metadata, formatted-answer token count with no
  special tokens. Uses the native HF tokenizer if available or an explicitly
  supplied offline `response_tokenizer` path. Otherwise null with an explicit
  unavailable diagnostic, never substituted with provider usage tokens.

Raw judge text, source identity, effective judge question, formatted answer,
finish reason, skipped verdict, thinking assertion, and parity gaps are logged.

## Provenance discovery and unresolved acceptance

Read-only discovery on 2026-11-08 found the candidate
[CORE-522 branch](https://github.com/Hirundo-io/hirundo-core/tree/core-522-add-rl-pipeline-for-hardening).
The local branch at
[4604d24ba0696a2f7d37fd2a68897bac1c96fee7](https://github.com/Hirundo-io/hirundo-core/commit/4604d24ba0696a2f7d37fd2a68897bac1c96fee7)
and GitHub main-to-branch comparison contain GRPO, not a Purple Llama task.
Also read the existing general
[Inspect wrapper](https://github.com/Hirundo-io/hirundo-core/blob/b3f013a9f303676d24cbf7d7cb3611742c58feb8/eval/inspect_wrapper.py)
on core-487-add-eval-dependencies (authored by Shmuli). It registers unrelated
Inspect evaluations, not Purple Llama. Searched local all-ref commit subjects,
Orr-authored evaluation changes, candidate tree filenames/content, and GitHub
code/branch search. The requested Orr implementation remains **unlocated**;
this is an independent native adaptation of the current repo, not a claimed
port. A source branch/file link is still needed to satisfy source-port acceptance.
No hirundo-core files were changed.

## Deterministic parity and limits

Tests compare parsing, prompt bytes, and stop-only grading/denominators against
actual legacy methods using monkeypatch; native mock runs validate role isolation,
budgets, metadata, and truncation exclusion. These establish fixture parity only,
not live-model, GPU, provider, or fixed-parity-model acceptance.

Current preprocess_free_text always produces judge_questions, using the raw
question for missing/empty overrides. It tokenizes these independently before
decoding. The issue's decoded-chat-prompt fallback applies to older generation
caches lacking that field, not fresh current runs. Native tasks do not load old
caches. Native messages do not replicate the legacy tokenizer/chat-template
roundtrip, 1024-token input truncation, or judge-question tokenization/truncation.
These are explicit unresolved parity gaps, logged on every sample and tested;
no measured tolerance is claimed. Muse Glimmer custom templates likewise remain
unsupported. Native Inspect completion omits structured reasoning; inline thinking
stripping follows the common legacy rule.

Reusable API: `dataset(*, dataset_path, dataset_token, max_samples, seed)`,
`parse_yes_no(raw)`, `injection_scorer(response_tokenizer=None)`, and
`prompt_injection(..., max_tokens=128, externally_configured_thinking=False,
response_tokenizer=None)`. Runtime imports never invoke legacy evaluators.
No common APIs or dependency files were changed.

Validation in this worktree (Inspect 0.3.288): 18 focused fixture tests pass;
repository-wide Ruff formatting/checks pass; focused Pyrefly (module and tests)
has zero diagnostics. Full Pyrefly reports 41 diagnostics outside this change
(including missing optional MLflow/stubs and existing test typing). CLI file-path
evaluation succeeds on a local snapshot with explicit mock roles. Tests use JSON
logs because the sandbox hung loading Inspect's lazy binary sample log; binary
log readback is not validated here. API role tests stub Model.generate and do not
claim HTTP endpoint or provider inference validation.
