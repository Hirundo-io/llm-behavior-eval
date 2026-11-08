# HaluEval native Inspect task (LLM-160)

```sh
inspect eval llm_behavior_eval/halueval --model hf/org/model \
  -M enable_thinking=false -M do_sample=false \
  --model-role grader=openai-api/airgap/judge \
  -T grader_thinking_disabled=true
```

Configure AIRGAP_BASE_URL and AIRGAP_API_KEY independently for the grader. The
flag attests that this endpoint has thinking disabled; it cannot disable thinking
on an arbitrary API. Without that attestation unsupported providers fail clearly.
HF requires both model arguments on each HF role; vLLM receives native
chat_template_kwargs.enable_thinking=false for both roles. Mock models need no
thinking controls. Generation is native Inspect: greedy, 128 answer tokens,
32 grader tokens. A grader role is mandatory, even for truncated answers.

Python callers can supply `model_roles={"grader": get_model(
"openai-api/airgap/judge", base_url="http://judge.internal:8000/v1",
api_key="no-auth-placeholder")}` to `inspect_ai.eval`. Model and grader endpoint,
credentials, model arguments, and thinking attestations are independent.

The logical dataset is `hirundo-io/halueval` (legacy `hallu`/`hallucination`).
Task parameters `dataset_path` and `dataset_token` use the foundation loader,
including offline save_to_disk snapshots. All rows are validated, then shuffled
with legacy seed 42 before slicing.
Default cap is 500; zero, negative, or Python None means all. Sample IDs are
`hirundo-io/halueval:<zero-based-source-row>` and are stable across caps.

Metrics declared by `hallucination.scorer` (for LLM-166's shared target mapping):

| Name | Unit | Definition | Direction |
| --- | --- | --- | --- |
| accuracy | percentage | 100 * (1 - INCORRECT / total) | higher |
| empty_responses | count | NOT_ATTEMPTED, including skips and parse failures | lower |
| incomplete_response_rate | fraction | max_tokens / total | lower |

Every sample remains in the denominator. Legacy saves accuracy as a fraction
internally and displays percent; this task reports the displayed percentage.
The legacy incomplete denominator is `num_samples`; it equals scored sample count
on completed runs. Partial/error logs cannot establish legacy run-level parity.
No shared target_metrics.json is changed; LLM-166 owns its assembly.

Reusable family API: `hallucination_scorer(*, thinking_disabled=False)` and
`hallucination_generate(*, thinking_disabled=False, grader_thinking_disabled=False)`.
The scorer uses Sample.input/target, the exact legacy GRADER_TEMPLATE, standalone
case-sensitive A/B/C parsing, and common legacy inline-thinking stripping. Only
stop completions invoke the grader. Metadata logs raw judge output, rendered judge
prompt when called, source identity, stop reason, skip/parse status, grade, question,
target, formatted answer, and unresolved parity gaps.

## Parity limits

Deterministic fixtures compare parsing and aggregate results directly with the
current FreeTextHaluEvaluator, including missing/unknown/truncated finish reasons.
They are not live-model parity acceptance. No GPU, provider, quantization, or
measured benchmark-result parity is claimed.

Native chat input has the exact legacy system prompt and trailing user newline.
However, legacy judges receive the **decoded chat-templated input**, after input
truncation to 1024 tokenizer tokens, and targets truncated to 256 tokens then
decoded. Native Inspect receives raw question/target; it does not reproduce those
transformations. These three gaps are recorded in every sample's `parity_gaps`
metadata and covered by a deterministic tokenizer fixture. Reproducing them needs
the evaluated model's exact tokenizer/template and truncation behavior, including
Muse Glimmer/custom HF handling. Arbitrary API endpoints cannot expose this.
A fixed real-model parity run remains required before full acceptance.

The deterministic runtime tests use JSON logs and a monkeypatched API grader
generate method to verify independent endpoint role routing, not HTTP/provider parity.
An attempted HTTP-transport fixture produced successful saved scores but stalled
in event-loop shutdown (also with explicit async client cleanup); transport
lifecycle validation remains unresolved in this environment. In this sandbox, Inspect's lazy `.eval` sample reader stalled;
JSON files allow checking the actual saved diagnostics directly. This does not
establish compressed-log readback parity. Use writable XDG_DATA_HOME and
XDG_CACHE_HOME directories when running Inspect in a restricted workspace.
