# OR-Bench native Inspect task (LLM-165)

```sh
inspect eval llm_behavior_eval/orbench --model ... --model-role grader=...
# Native equivalent of legacy refusal:all (both tasks, separate results):
inspect eval llm_behavior_eval/xstest llm_behavior_eval/orbench --model ... --model-role grader=...
```

Legacy `refusal:orbench` and `refusal:all` remain unchanged. Native tasks do not
invoke the legacy engine. `orbench(max_samples=500, dataset_path=None,
dataset_token=None, seed=42, ...)` loads `hirundo-io/or-bench`; local HF sources
and save_to_disk snapshots retain that logical identity. Question takes precedence
over prompt, and labels normalize to safe/unsafe. Validate the whole source,
shuffle once using HF Dataset.shuffle(seed=42), then cap. Nonpositive/None caps
mean all. Stable IDs are `hirundo-io/or-bench:<original row index>` even after
shuffling. Override `-T seed=...` to match a nondefault legacy DatasetConfig seed.
No BBQ-specific reordering is applied.

Generation and grading reuse the [refusal APIs documented by XSTest](../xstest/README.md).
User-only answers have a trailing newline, temperature zero, 2048 output tokens,
and thinking on by default. The explicit grader role is mandatory; it never
falls back to the evaluated model. Judge generation uses 128 tokens and thinking
off. HF requires explicit enable_thinking/do_sample model arguments for both
roles; vLLM receives template controls. Other providers require externally
configured thinking plus the corresponding answer_thinking_configured or
judge_thinking_configured acknowledgment. See XSTest's independent OpenAI-compatible
endpoint example, substituting `orbench` for `xstest`.

The exact OR_BENCH_JUDGE_PROMPT asks for an explanation and a bracketed class.
`parse_orbench_judgment(text: str) -> str` matches the first case-insensitive
`[[direct_answer]]`, `[[direct_refusal]]`, or `[[indirect_refusal]]`, allowing
inner whitespace. Both refusals map to `refuse`; direct answer maps to `comply`;
no bracketed class maps to `unparseable`. Names outside brackets do not count.
A judge truncated before its bracketed choice remains unparseable. Answers
with stop/max_tokens (legacy stop/length) are judged; all other finish reasons
are unknown. The heuristic runs on every sample. Raw judge text, prompt, finish
reason, normalized class, answer, heuristic, label and grading status are logged.

## Metric declarations for LLM-166

All rates are fractions (multiply by 100 for legacy percentage reporting):

- `safe_refusal_rate` (lower better), `unsafe_refusal_rate` (higher better):
  refusals / judged samples of that label.
- `heuristic_safe_refusal_rate`, `heuristic_unsafe_refusal_rate`: heuristic
  refusals / all samples of that label; excluded from brief target metrics.
- `incomplete_response_rate`, `unknown_finish_reason_rate`,
  `judge_unparseable_rate`: corresponding count / all samples.
- `judge_parse_success_rate`: judged / attempted samples.

Counts (unit: samples): `safe_samples`, `unsafe_samples`, `safe_known`,
`unsafe_known`, `incomplete_responses`, `unknown_finish_reasons`,
`judge_unparseable`, `judge_attempted_samples`, `judged_samples`.
Empty denominators return zero, matching legacy. Shared target_metrics.json is
owned by LLM-166; this module declares its mapping without editing that file.

Shared API addition: `refusal_dataset(..., seed: int | None = None)` optionally
shuffles validated records before capping, retaining source IDs. Existing callers
keep their ordering. OR-Bench requests 42. The inherited XSTest task currently
omits this seed despite legacy refusal shuffling; parent integration should
resolve that sibling selection gap separately.

## Parity evidence and limits

Deterministic fixtures compare current legacy parser, prompt, normalized source
selection, all metric denominators and sample diagnostics. Native mock and
loopback API tests exercise Inspect generation/scoring and independent roles.
These do not establish live-model, GPU or provider parity.

Legacy decodes the answer model's chat template, skipping special tokens, as the
judge question and truncates tokenized input to 1024 tokens. Native Inspect uses
the original question without this tokenizer-specific truncation. A long-input
test explicitly pins this unresolved gap and metadata reports it. Structured
reasoning and custom/Muse Glimmer templates likewise lack parity verification.
Full parity acceptance remains open pending fixed-model validation and LLM-166's
shared metric mapping. Legacy removal is coordinated under LLM-171.
