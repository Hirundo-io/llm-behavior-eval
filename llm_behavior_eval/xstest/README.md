# XSTest native Inspect task (LLM-164)

`inspect eval llm_behavior_eval/xstest --model ... --model-role grader=...`
loads `hirundo-io/XSTest`. Legacy `refusal:xstest` and `refusal:all` are unchanged.
Use `-T dataset_path=/snapshot` for a saved HF dataset or another HF loading
source; `dataset_token` is optional. Source row IDs are `hirundo-io/XSTest:<index>`.
The entire source is validated and shuffled with the legacy seed 42 before the
default 500-row cap; source-row IDs remain stable. Nonpositive or
None means all. Input is question (preferred) or prompt, with no target.

Answer generation is native Inspect, user-only with a trailing newline,
temperature zero, 2048 output tokens. Thinking defaults on; `-T thinking=false`
requests off. HF requires explicit role model args `enable_thinking=true` and
`do_sample=false` for answers, false/false for grader. vLLM receives the common
chat-template thinking setting automatically. Mock models need no control.
Other providers fail unless `answer_thinking_configured=true` and/or
`judge_thinking_configured=true` acknowledges external configuration for the
corresponding role. These flags are assertions, not a mechanism to turn thinking
off. Configure the answer endpoint to match `thinking`, and judge endpoint off.
Muse Glimmer/custom HF templates remain unsupported.

The explicit `grader` role is mandatory, even for skipped samples. Independent
OpenAI-compatible endpoints use the foundation's role isolation, for example:

```python
from inspect_ai import eval
from inspect_ai.model import get_model
from llm_behavior_eval.xstest.task import xstest

grader = get_model(
    "openai-api/airgap/judge",
    base_url="http://judge.internal:8000/v1",
    api_key="no-auth-placeholder",
)
eval(
    xstest(judge_thinking_configured=True, compatibility_tokenizer="/models/answer"),
    model="hf/local",
    model_args={
        "model_path": "/models/answer",
        "tokenizer_path": "/models/answer",
        "tokenizer_call_args": {
            "truncation": True,
            "max_length": 1024,
            "add_special_tokens": True,
        },
        "enable_thinking": True,
        "do_sample": False,
    },
    model_roles={"grader": grader},
)
```

Judge generation is user-only, 128 tokens, greedy, using the exact legacy prompt.
Inspect `max_tokens` maps legacy `length`; both this and `stop` are judged.
Other reasons (including absent choices) are unknown and skipped. Raw judge text,
judge prompt/reason, answer/reason, label, heuristic, parsed class, grading status,
and thinking configuration are logged as score/sample metadata. Inline thinking
stripping uses the common helper, with configurable markers/exclusion.

## Metrics and integration contract

Rates below are fractions, with **zero for empty denominators**, matching legacy:

- `safe_refusal_rate`, `unsafe_refusal_rate`: full or partial refusals / judged
  samples of the corresponding label. Common custom metric entry points delegate
  to shared refusal aggregation.
- `heuristic_safe_refusal_rate`, `heuristic_unsafe_refusal_rate`: heuristic refusals
  / all corresponding label samples. Excluded from brief target metrics.
- `incomplete_response_rate`, `unknown_finish_reason_rate`,
  `judge_unparseable_rate`: corresponding count / all scored samples.
- `judge_parse_success_rate`: judged / attempted.

Count metrics (unit: samples): `safe_samples`, `unsafe_samples`, `safe_known`,
`unsafe_known`, `incomplete_responses`, `unknown_finish_reasons`,
`judge_unparseable`, `judge_attempted_samples`, `judged_samples`.
Inspect score dictionaries group each metric by its declared name.
LLM-166 owns shared target_metrics.json: map safe_refusal_rate to percentage,
lower better; unsafe_refusal_rate to percentage, higher better; multiply native
fractions by 100. No shared mapping is written here.

Reusable family APIs: `refusal_dataset(dataset_id, *, dataset_path=None,
dataset_token=None, max_samples=500)`; `refusal_scorer(prompt, parse, *,
judge_thinking_configured=False, exclude_thinking_trace_for_judge=True,
thinking_start_token='<think>', thinking_end_token='</think>')`, where prompt
uses `{question}`/`{response}` and parse returns comply/refuse/partial/unparseable;
`refusal_generate(*, thinking=True, answer_thinking_configured=False,
judge_thinking_configured=False)`; `aggregate_refusal(scores)` and
`refusal_stat(name)`. No OR-Bench task is implemented here.

## Parity limits

Production preprocessing now preserves the legacy 1024-token input and decoded judge text, plus 256-token target/stereotype round trips. Original input/target remain logged. See [HF/vLLM configuration and live qualification limits](../../INSPECT_MIGRATION.md#native-hf-preprocessing). CPU boundary tests do not establish live generation or score parity.

Production HF/vLLM answers require an explicit matching `compatibility_tokenizer`.
See [native HF configuration](../../INSPECT_MIGRATION.md#native-hf-preprocessing)
for answer-only tokenizer caps and independently configured grader options.
