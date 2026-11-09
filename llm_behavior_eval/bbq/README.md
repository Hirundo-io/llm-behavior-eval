# Native BBQ (LLM-157)

Use the Python example below to configure both HF roles independently.

Configure the grader's HF model arguments independently as well (Python example
below); its tokenizer requires `enable_thinking=False`, `do_sample=False`.
The task rejects missing grader roles. Gemma is recommended, never implicitly
loaded or replaced with the evaluated model. Native Inspect generates and scores;
legacy CLI and evaluator APIs are removed in 0.2.0.

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

grader = get_model(
    "hf/local-judge",
    model_path="/models/gemma3-judge",
    auto_model_class="AutoModelForImageTextToText",
    enable_thinking=False,
    do_sample=False,
)
eval(
    bbq(compatibility_tokenizer="/models/answer"),
    model="hf/local",
    model_args={
        "model_path": "/models/answer",
        "enable_thinking": False,
        "do_sample": False,
        "tokenizer_call_args": {
            "truncation": True,
            "max_length": 1024,
            "add_special_tokens": True,
        },
    },
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

Deterministic tests compare against frozen legacy `_grade_impl` outputs with fixed
judge results and no inference; this is not live-model parity. GPU, provider,
quantization and fixed parity model acceptance remain unverified.

Production preprocessing now preserves the legacy 1024-token input and decoded judge text, plus 256-token target/stereotype round trips. Original input/target remain logged. See [HF/vLLM configuration and live qualification limits](../../INSPECT_MIGRATION.md#native-hf-preprocessing). CPU boundary tests do not establish live generation or score parity.

Only `stop` outputs reach the grader. Inspect `max_tokens` maps legacy `length`
and counts as incomplete/empty/incorrect. Other finish reasons are skipped and
count as empty/incorrect but not incomplete: legacy currently raises iterator
exhaustion for them. Missing choices use this same documented extension. Raw GT,
stereotype and uncertainty judge output, parsed flags, finish reason, source
metadata and the actual judge question are retained in score metadata.

Production HF/vLLM answers require an explicit matching `compatibility_tokenizer`.
See [native HF configuration](../../INSPECT_MIGRATION.md#native-hf-preprocessing)
for answer-only tokenizer caps and independently configured grader options.
