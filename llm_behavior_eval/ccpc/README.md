# CCPC native Inspect adaptation (LLM-163)

Task: `inspect eval llm_behavior_eval/ccpc --model mockllm/answer --model-role grader=mockllm/judge`.
Real qualification uses an explicitly configured `grader=hf/google/gemma-4-26B-A4B-it`.
On **both** HF roles set model arguments `enable_thinking=false, do_sample=false`;
supply role arguments independently through Inspect model-role configuration.
The bare HF CLI from the issue needs these arguments. No judge defaults to the
evaluated model. vLLM receives thinking-off chat-template kwargs automatically.
Other providers require explicit endpoint configuration and task flags
`answer_thinking_configured` / `judge_thinking_configured` acknowledging thinking off.
An independent OpenAI-compatible grader can be supplied using
`get_model("openai-api/airgap/judge", base_url=..., api_key=...)` in
`eval(..., model_roles={"grader": grader})`. Credentials are not task metadata.
Judge templates must support the system role; there is no demotion fallback.

## Source and deliberate differences

Adapted, not independently implemented, from
[upstream task/scorer/dataset module at f93e3ff4ff80d09cd7baf4a116fd6dfc8d486bdc](https://github.com/Hirundo-io/chinese-censorship-bench/blob/f93e3ff4ff80d09cd7baf4a116fd6dfc8d486bdc/src/ccpc_bench/ccpc_bench.py).
Upstream MIT notice is retained in LICENSE.upstream. This was the head of closed,
unmerged PR #2 when inspected. Its publication parity is **not qualified** by this port.

The upstream metric functions, verdict metadata approach, parser, JSON prompt
renderer, and native judge call are retained/adapted into separate modules.
Deliberate differences implement the requested llm-behavior-eval v6 contract:

* Dataset is `hirundo-io/ccpc-500`, default/test, revision
  `e778807d1cbe31f5e4f4ff1941d3d6fe48faed70`, rather than upstream's
  `hirundo-io/ccpc-bench` JSONL artifact at `b5b3caeb518a7450e773dc38f029cb0770241b52`.
* Full 500-row schema, unique IDs, nonempty strings, and SHA-256 ordered content
  fingerprint are checked before any cap. No shuffling. Stable IDs are benchmark_id.
* Answer has only a user message, with the legacy trailing newline.
* Every answer is judged, including missing/unknown/max_tokens finish reasons;
  bare verdicts are parsed even when the judge reports max_tokens.
* Required grader role replaces upstream fallback and optional judge parameter.
* Explicit provider thinking checks replace upstream unconditional extra_body.
* Local `dataset_path` / `file_path` support HF sources and save_to_disk snapshots;
  saved DatasetDict uses test, never train. Default cap 500; nonpositive/None all.

`contract.py` is the native task's single frozen contract source. Its constants
are checked byte-for-byte against legacy censorship_utils in tests. The rubric
matches upstream exactly. Cross-repository contract extraction is still a
coordination item for RSCH-82: the repositories currently disagree on dataset
coordinates and finish handling, so claiming a shared installable contract would
be misleading. No publication repository changes were made.

## APIs and metric mapping for LLM-166

`load_ccpc_dataset(*, dataset_path=None, file_path=None, dataset_token=None,
max_samples=500)` returns a validated Inspect MemoryDataset.
`censorship_judge(*, judge_thinking_configured=False, judge_revision=None)`
is a native Scorer. `ccpc_config(model, *, judge=False,
externally_configured=False)` returns the generation settings.

| Metric | Native units | Summary units | Direction |
| --- | --- | --- | --- |
| censorship_rate | fraction | percentage, multiply by 100 | lower |
| stderr | fraction, sample stdev / sqrt(n), singleton 0 | fraction | diagnostic |
| judged_fraction | fraction over every row | fraction | diagnostic |

A single unparseable verdict invalidates rate **and** stderr. Inspect stores these
as NaN (not parsed-only means); actual .eval roundtrip tests verify this.
`reported_rate(value, percentage=False)` converts undefined values to None for
JSON/CSV consumers; it also supports summary percentage conversion. LLM-166 owns
the shared target_metrics.json write.

Task metadata records dataset/rubric/version, budgets, thinking, and declared
judge revision. Inspect eval metadata records actual role identity/config; sample
scores additionally record actual judge identity, raw output, verdict, both finish
reasons and generation config. `judge_revision` is a caller declaration, not a
resolved-weight attestation: pin the actual provider revision independently.

## Validation and limits

Deterministic fixtures compare the current legacy parser, fingerprint and metric
formulas; native mock evals exercise .eval serialization and all-row denominators.
Synthetic 500-row fixtures substitute only their expected fingerprint; they do
not establish that the remote pinned dataset is available or unchanged.
Endpoint routing is tested with an explicitly separate API Model and monkeypatched
generation, not a live server.

Production preprocessing now preserves the legacy 1024-token input boundary; CCPC intentionally retains the original source judge question. Original input/target remain logged. See [HF/vLLM configuration and live qualification limits](../../INSPECT_MIGRATION.md#native-hf-preprocessing). CPU boundary tests do not establish live generation or score parity.

Production HF/vLLM answers require an explicit matching `compatibility_tokenizer`.
See [native HF configuration](../../INSPECT_MIGRATION.md#native-hf-preprocessing)
for answer-only tokenizer caps and independently configured grader options.
