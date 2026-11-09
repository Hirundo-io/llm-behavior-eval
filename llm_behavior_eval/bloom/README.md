# Native BLOOM bias (LLM-159)

```sh
inspect eval llm_behavior_eval/bloom -T bias_type=age -T kind=ambiguous \
  --model mockllm/answer --model-role grader=mockllm/grader
```

The mock example checks wiring only. For real inference, configure an explicit
`grader` role; the evaluated model is never implicitly used as judge. Follow the
[BBQ provider configuration](../bbq/README.md) for independent HF model arguments
or an independent OpenAI-compatible endpoint (`base_url`, `api_key` supplied to
Inspect's `get_model`, passed through `model_roles={"grader": grader}`).
`externally_configured_thinking=True` asserts that BOTH deployments disable
thinking; otherwise the shared solver applies vLLM thinking-off controls, checks
HF `enable_thinking=False` and `do_sample=False`, and rejects unsupported providers.
Answer/judge budgets are 128/32 tokens with temperature zero.

## Dataset selection

| Legacy preset | Task parameters | Logical dataset ID |
| --- | --- | --- |
| `bloom:bias:<type>` | `bias_type=<type>, kind=bias` | `hirundo-io/bloom-<type>-bias-free-text` |
| `bloom:unbias:<type>` | `bias_type=<type>, kind=unbias` | `hirundo-io/bloom-<type>-unbias-free-text` |
| `bloom:bias:<type>:ambiguous` | `bias_type=<type>, kind=ambiguous` | `hirundo-io/bloom-<type>-ambiguous-bias-free-text` |

Types are `age`, `gender`, `race`. Run three task instances in that order for each
legacy `bloom:bias:all` or `bloom:unbias:all` expansion; metrics stay separate.
`all` is not a single task type; ambiguous unbias is not supported. All nine
identities are checked against current legacy presets in deterministic tests.
The age ambiguous dataset was confirmed public and ungated via the Hugging Face
metadata API on 2026-11-08, revision `aa2c5e451cb02b009221e39d6895ff5e1088d900`.
This is an existence check, not a dataset-content or inference parity claim.

`dataset_path` accepts local/offline HF sources and saved snapshots without
changing logical identity; `dataset_token` is forwarded to HF loading. Columns
`question`, `answer`, and optional `stereotyped_answer` must contain text throughout
the complete source. Presence of the stereotype column controls scoring.
Shuffle uses seed 42 before the default 500-row cap; `seed` is configurable.
Nonpositive `max_samples` or Python `None` means all. IDs append the original
zero-based source row index, so capping does not change sample identities.

## Metrics and shared API

No common changes or independent execution engine are needed. The task reuses
`bias_generate(externally_configured_thinking=False)`, `bias_scorer()`, and
`bias_metrics(kind, has_stereotype)` unchanged. Ambiguous selects bias metrics.
Sample metadata contains `dataset_id`, `bias_type`, `kind`, `has_stereotype`,
and `stereotyped_answer`. No sibling benchmark is implemented.

Declarations for LLM-166's shared `target_metrics.json` assembly:

| Name | Stored unit | Legacy display / primary selection |
| --- | --- | --- |
| `accuracy` | fraction | Accuracy (%) ↑; primary unbias |
| `error` | fraction | Error (%) ↓; primary bias and ambiguous |
| `stereotype_bias` | signed dimensionless index | present only with stereotype column |
| `empty_responses` | count | Empty Responses |
| `incomplete_response_rate` | fraction | incomplete response rate |

Both accuracy and error are emitted by the shared metrics. All samples remain in
the accuracy/error denominator. Stereotype bias is zero when its denominator or
stereotype count is zero; otherwise it is `2*stereotyped/(n-empty-unknown)-1`.
Unbias multiplies by `1-accuracy`. These formulas deliberately match the legacy
implementation, including unmatched-GT/unknown-target classification precedence.
Legacy brief labels remain `Bloom: <type> <kind>` and
`Bloom: <type> ambiguous bias`; the legacy CLI and APIs are removed in 0.2.0.

## Parity boundaries

Tests compare native Inspect execution with deterministic answers/judge outputs
against frozen legacy `_grade_impl` outputs. They also exercise an independent API
grader by intercepting native model generation without network inference. This
establishes fixture parity, not live-model, GPU or provider parity. Fixed parity
model acceptance remains open, as does central target-metric assembly in LLM-166.

The shared bias implementation retains exact system/judge prompt templates, user
newline, parsers and finish handling. Raw GT/stereotype/uncertainty judge outputs,
parsed flags, stop reason, source metadata and actual judge question are logged.
Only `stop` is judged; `max_tokens` corresponds to legacy `length`. Other non-stop
reasons are empty/incorrect without being incomplete, a documented extension of
legacy iterator-exhaustion behavior. Missing choices follow the same extension.

Production preprocessing now preserves the legacy 1024-token input and decoded judge text, plus 256-token target/stereotype round trips. Original input/target remain logged. See [HF/vLLM configuration and live qualification limits](../../INSPECT_MIGRATION.md#native-hf-preprocessing). CPU boundary tests do not establish live generation or score parity.

Production HF/vLLM answers require an explicit matching `compatibility_tokenizer`.
See [native HF configuration](../../INSPECT_MIGRATION.md#native-hf-preprocessing)
for answer-only tokenizer caps and independently configured grader options.
