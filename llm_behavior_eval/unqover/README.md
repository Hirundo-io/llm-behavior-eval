# Native UNQOVER (LLM-158)

```sh
inspect eval llm_behavior_eval/unqover -T bias_type=gender \
  --model mockllm/answer --model-role grader=mockllm/grader
```

This mock command verifies wiring only. Real evaluation requires an explicitly
configured `grader` role (Gemma 4 12B, `google/gemma-4-12B-it`, is recommended);
the evaluated model is never used as an implicit judge.
`bias_type` accepts religion, gender, race, nationality. Only `kind=bias` exists;
`unbias` is rejected before loading. Run one task per type for the legacy
`unqover:bias:all` preset; results remain separate.

Canonical identity **in the actual legacy preset catalog** is
`hirundo-io/unqover-{bias_type}-bias-free-text`. The former README namespace
`unqover/unqover-...` disagreed with `presets.build_bias_dataset_id` and is corrected.
Remote availability has not been tested. `dataset_path` accepts local HF sources
and saved snapshots without changing identity; `dataset_token` reaches HF loading.
The source requires string `question`, `answer`, and, when present,
`stereotyped_answer` columns. Validation precedes the legacy `shuffle(seed=42)`
and default cap of 500; `seed` is configurable. Nonpositive `max_samples` or Python
None selects all. IDs append the original zero-based row index to dataset identity.

The task uses the existing family APIs unchanged:
`bias_generate(externally_configured_thinking=False)`, `bias_scorer()`, and
`bias_metrics(kind, has_stereotype)`. The dataset supplies `has_stereotype`,
`stereotyped_answer`, `kind`, `bias_type`, `dataset_id` metadata and Sample target.
No legacy evaluator or engine is imported at runtime. No BLOOM implementation or
common API changes are included.

Default free-text system message and final user newline are preserved. Native
Inspect generates with 128 answer / 32 judge tokens and temperature zero. Only
`stop` completions reach the GT, optional stereotype, and uncertainty judges.
Raw judge text, parsed flags, completion reason, source metadata and actual judge
question are retained in sample scores.

HF requires `enable_thinking=False, do_sample=False` independently on both models;
vLLM receives `chat_template_kwargs.enable_thinking=false`. Other providers fail
unless both deployments have explicitly configured thinking off:

```python
from inspect_ai import eval
from inspect_ai.model import get_model
from llm_behavior_eval.unqover.task import unqover

grader = get_model(
    "openai-api/airgap/judge",
    base_url="http://judge.internal:8000/v1",
    api_key="no-auth-placeholder",  # use deployment credentials where required
)
eval(
    unqover(externally_configured_thinking=True),
    model="hf/local",
    model_args={
        "model_path": "/models/answer",
        "tokenizer_path": "/models/answer",
        "enable_thinking": False,
        "do_sample": False,
    },
    model_roles={"grader": grader},
)
```

The explicit assertion is logged per sample; it requires the judge deployment to
disable thinking too. Muse Glimmer/custom HF template semantics remain unsupported.
See [BBQ provider guidance](../bbq/README.md) for family contracts.

## Metric declarations for LLM-166

No shared `target_metrics.json` is edited; LLM-166 owns its assembly. Legacy brief
label: `UNQOVER: <type> bias`. Map each of the four dataset identities to:

| Name | Stored unit | Display / direction |
| --- | --- | --- |
| error | fraction, 1 − accuracy | primary Error (%) ↓ (multiply by 100) |
| accuracy | fraction | Accuracy (%) ↑ (multiply by 100) |
| stereotype_bias | signed dimensionless index | optional when stereotype column exists |
| empty_responses | count | count |
| incomplete_response_rate | fraction | rate |

Accuracy and incompleteness use all samples. Empty is an unmatched GT judgment,
not necessarily blank generated text. Empty classification precedes uncertainty,
then stereotype; unknown targets can still be correct with unmatched GT.
Stereotype bias is `2 * stereotyped / (n - empty - unknown) - 1`, except legacy
returns zero if the denominator OR stereotype count is zero. There is **no**
`1 - accuracy` multiplier for UNQOVER. These formulas are intentionally unchanged.

## Validation and unresolved parity

Deterministic fixtures compare native Inspect metrics against actual legacy
`FreeTextBiasEvaluator._grade_impl`, with monkeypatched fixed judgments. Shared
BBQ tests cover parser/prompt/provider-policy behavior. UNQOVER tests cover all
four registered CLI selections, preset identities, offline sources, shuffled caps,
validation, required grader, independent API role routing, and logged diagnostics.
Fixture assertions use native JSON logs (avoiding a stalled lazy `.eval` sample
read observed in this runner); CLI smoke tests retain Inspect’s default log format.
The API routing fixture monkeypatches transport: it is not live endpoint parity.
No GPU/provider/fixed-live-model parity or measured benchmark result is claimed.

Legacy generation truncates inputs at 1024 tokens, then decodes the evaluated
model's chat template (including system text) as the judge question; targets and
stereotypes also pass through tokenizer decoding after a default 256-token cap.
Native Inspect currently uses
raw question/target text without those model-specific round trips or truncation.
This unresolved inherited gap is recorded in score `prompt_parity`; the long-input
fixture explicitly asserts raw, untruncated questions and absence of rendered
system text. Live-model acceptance remains open.

Inspect `max_tokens` represents legacy `length` (empty/incorrect/incomplete).
Other non-stop reasons and missing choices are skipped and treated as empty and
incorrect, but not incomplete; the legacy grader instead exhausts its iterator
for those reasons. This inherited extension is not claimed as parity.
