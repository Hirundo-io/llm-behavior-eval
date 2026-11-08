# Native BBQ (LLM-157)

`inspect eval llm_behavior_eval/bbq -T bias_type=gender -T kind=bias --model hf/ORG/MODEL -M enable_thinking=false -M do_sample=false --model-role grader=hf/google/gemma-3-12b-it`

Configure the grader's HF model arguments independently as well (Python example
below); its tokenizer requires `enable_thinking=False`, `do_sample=False`.
The task rejects missing grader roles. Gemma is recommended, never implicitly
loaded or replaced with the evaluated model. Native Inspect generates and scores;
legacy CLI and evaluator APIs are unchanged.

One parameterized task selects one split. `bias_type` accepts gender, race,
nationality, physical, age, religion; `kind` accepts bias or unbias. Run six task
instances for the legacy `bias:all` / `unbias:all` presets; metrics remain per split.
Identity is `hirundo-io/bbq-{bias_type}-{kind}-free-text`, sample IDs append the
zero-based source row index. `dataset_path` accepts local HF sources or saved
snapshots, `dataset_token` is forwarded to HF loading. Legacy seed-42 shuffle before taking the first 500 (`seed` is configurable);
`max_samples=0`, negative, or Python None uses all. Entire sources are validated
before shuffling/capping. Unlike the foundation guide’s source-order example, this preserves actual legacy shuffle behavior. Stereotype presence follows the column schema.

```python
from inspect_ai import eval
from inspect_ai.model import get_model
from llm_behavior_eval.bbq.task import bbq

grader = get_model("hf/google/gemma-3-12b-it", enable_thinking=False, do_sample=False)
eval(
    bbq(),
    model="hf/ORG/MODEL",
    model_args={"enable_thinking": False, "do_sample": False},
    model_roles={"grader": grader},
)
```

Independent OpenAI-compatible judge endpoints are supported through Inspect:
construct `get_model("openai-api/airgap/judge", base_url=..., api_key=...)` and
pass it as the grader role. Set `externally_configured_thinking=True` only after
configuring BOTH deployments to disable thinking. This explicit assertion is
logged per sample. Otherwise HF arguments are checked, vLLM receives
`chat_template_kwargs.enable_thinking=false`, mockllm needs no control, and
unsupported providers fail explicitly. Budgets are 128 answer / 32 judge tokens,
temperature zero. API endpoints must honor those generation controls.

Shared family API for future UNQOVER/BLOOM: `bias_scorer()` consumes Sample target
and metadata `has_stereotype`, `stereotyped_answer`; `bias_metrics(kind,
has_stereotype)` aggregates scores; `bias_generate(externally_configured_thinking=False)`
validates roles, adds the legacy user newline, and generates natively. Parsing,
classification, exact system prompts and `agreement_user` are reusable. No sibling
benchmark is implemented.

Metric declarations for LLM-166 (no shared target_metrics.json changes):

| Name | Stored unit | Display / direction |
| --- | --- | --- |
| accuracy | fraction | Accuracy (%) ↑; primary unbias |
| error | fraction | Error (%) ↓; primary bias |
| stereotype_bias | signed dimensionless index | legacy formula; only with stereotype column |
| empty_responses | count | count |
| incomplete_response_rate | fraction | rate |

Accuracy includes every sample, even truncated/unmatched responses. Empty takes
classification precedence over unknown, then stereotype. Unknown targets may be
correct even with unmatched GT. Stereotype bias is zero if either its denominator
or stereotype count is zero; unbias multiplies by `1-accuracy`.

## Parity boundaries

Deterministic tests compare against the actual legacy `_grade_impl` with fixed
judge results and no inference; this is not live-model parity. GPU, provider,
quantization and fixed parity model acceptance remain unverified.

Exact system/user text is preserved except the judge Question uses the raw
question. Legacy decodes the evaluated tokenizer's chat template (including
system text), truncates model inputs at 1024 tokens, and tokenizes/decodes targets
and stereotyped targets too. Native Inspect does not replicate these
model-specific round trips or input truncation. These are unresolved parity gaps,
explicitly logged in `prompt_parity`; tests pin the raw-question choice. Long
inputs therefore can differ. Muse Glimmer custom chat templates are unsupported.

Only `stop` outputs reach the grader. Inspect `max_tokens` maps legacy `length`
and counts as incomplete/empty/incorrect. Other finish reasons are skipped and
count as empty/incorrect but not incomplete: legacy currently raises iterator
exhaustion for them. Missing choices use this same documented extension. Raw GT,
stereotype and uncertainty judge output, parsed flags, finish reason, source
metadata and the actual judge question are retained in score metadata.
