# OR-Bench (native Inspect)

Over-refusal benchmark on `hirundo-io/or-bench`, the native counterpart of the
legacy `refusal:orbench` preset. Prompts are labeled **safe** (the model should
answer) or **unsafe** (the model should refuse). A judge classifies each answer,
and a phrase heuristic gives a judge-free diagnostic. This package is
self-contained: it imports only generic mechanics from `llm_behavior_eval.common`.

```sh
inspect eval llm_behavior_eval/orbench --model <provider/model> \
  --model-role 'grader={model: <provider/judge>, temperature: 0.0, max_tokens: 128}'
```

The task binds no grader, so a run without `--model-role grader=...` fails before
any answer is generated. Legacy `refusal:all` maps to running XSTest and OR-Bench
together, for example `inspect eval-set llm_behavior_eval/xstest
llm_behavior_eval/orbench --log-dir <dir> ...` (XSTest ships in its own PR).

The model under test must report stop reasons. Inspect's `hf/` provider reports
every stop reason as `unknown`, so with an `hf/` model under test the scorer
raises `StopReasonUnavailableError` and the eval fails. Serve the model with
`vllm/` or an OpenAI-compatible endpoint (`openai-api/<service>/<model>`)
instead. The grader may use any provider, including `hf/`.

## Task arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `dataset_path` | `None` | Local HF dataset or `save_to_disk` snapshot; keeps the logical id `hirundo-io/or-bench`. |
| `revision` | `None` | Hub dataset revision (branch, tag or commit) passed to `load_hf_dataset(revision=...)`. Ignored for a `save_to_disk` snapshot. |
| `seed` | `42` | Sample order, reproducing the legacy `Dataset.shuffle(seed)` (`common.dataset.legacy_order`). |
| `exclude_thinking_trace_for_judge` | `true` | Strip inline reasoning before judging and the heuristic. `false` gives the grader and the heuristic the whole answer, the legacy default. |
| `thinking_start_token` / `thinking_end_token` | `<think>` / `</think>` | Inline reasoning markers. `null` turns stripping off. |

There is no token argument, because Inspect records every task argument in the
`.eval` log. For gated access set `HF_TOKEN` (or log in with
`huggingface-cli login`); the Hub loader reads it.

The thinking arguments are the shared `common.judge.THINKING_TASK_ARGS`, with
the same names and defaults as every benchmark. For the legacy behavior:

```sh
inspect eval llm_behavior_eval/orbench ... -T exclude_thinking_trace_for_judge=false
```

There are no inference arguments. Sample limits are Inspect's (`--limit N`).

## Dataset

The whole source is validated before conversion. Rows need a `question` or
`prompt` column (`question` wins) and a `label`, normalized to `safe`/`unsafe`
case-insensitively; anything else is rejected. A Pydantic model (`SourceRow`)
validates each row. Like the legacy engine, all rows
are shuffled once with `seed` and then capped, so `--limit 500` (or
`parity.yaml`) selects the legacy samples. Sample IDs are the original row
positions, `hirundo-io/or-bench:<row>`, and survive the shuffle.

## Solver and scorer

The solver sends the question followed by a newline as the only user message,
with no system prompt, as the legacy engine did. It sets no GenerateConfig.

The scorer (`orbench_scorer`) sends `OR_BENCH_JUDGE_PROMPT` (copied verbatim
from the legacy engine) to the `grader` role, with no GenerateConfig of its own.
The judge briefly explains and then names its class inside `[[ ]]`.
`parse_orbench_judgment` takes the first case-insensitive `[[direct_answer]]`,
`[[direct_refusal]]` or `[[indirect_refusal]]`, allowing inner whitespace. Both
refusal classes count as refusals. Class names outside the brackets do not count,
so a judge cut off before its bracketed choice is unparseable. Watch
`judge_unparseable_rate` under small judge token budgets.

Answers that stopped with `stop`, `max_tokens` or `model_length` (legacy `stop`
and `length`) are judged. Any other stop reason is an unknown finish reason and
is not judged. Because it filters on the stop reason, the scorer first calls
`common.judge.require_stop_reason`, which rejects an `hf/` model under test
(see above). The heuristic runs on every sample. Each sample's score value is 1 for a judged
refusal and 0 otherwise. Score metadata keeps the question, answer,
judge prompt, raw judge text, judge stop reason, class, grading status, heuristic
result and label.

## Metrics

One metric, `orbench_metrics`, reports every statistic under its plain name in
the single `orbench_scorer` result of the `.eval` log
(`[result] = log.results.scores`), which is where `target_metrics.json` and
inspect-brief look them up. Rates are fractions. An empty denominator yields 0,
as in the legacy engine, except `incomplete_response_rate`, which is NaN when no
stop reason is known.

**Epochs.** `orbench_metrics` is `@metric(scores="unreduced")`: every (sample,
epoch) result is one observation. With `--epochs N`, rates pool all N epochs of
every sample, and **counts count sample x epoch observations**, not samples
(`safe_samples` is N times the number of safe samples). `--epochs-reducer` does
not change these metrics. With one epoch, observations are samples.

| Metric | Denominator (observations) | Direction |
| --- | --- | --- |
| `safe_refusal_rate` | judged safe | lower is better |
| `unsafe_refusal_rate` | judged unsafe | higher is better |
| `heuristic_safe_refusal_rate`, `heuristic_unsafe_refusal_rate` | all of the label | diagnostic |
| `incomplete_response_rate` | known stop reason (not `unknown` or missing) | lower is better |
| `unknown_finish_reason_rate`, `judge_unparseable_rate` | all | lower is better |
| `judge_parse_success_rate` | attempted judgments | higher is better |

`incomplete_response_rate` follows the shared rule (`common.metrics`): an answer
stopped by `max_tokens` or `model_length` is incomplete, and answers whose stop
reason is `unknown` or missing are left out of the denominator. The legacy
engine counted only `length` and divided by all samples, so the rate differs
from the legacy figure when some stop reasons are unknown.

Counts: `safe_samples`, `unsafe_samples`, `safe_known`, `unsafe_known`,
`incomplete_responses`, `unknown_finish_reasons`, `judge_unparseable`,
`judge_attempted_samples`, `judged_samples`. The brief metrics in
`target_metrics.json` are the legacy `summary_brief.csv` columns:
`safe_refusal_rate`, `unsafe_refusal_rate`, `incomplete_response_rate` and
`judge_unparseable_rate`.

## Parity config

`parity.yaml` holds the legacy settings for both roles. Pass it explicitly:

```sh
inspect eval llm_behavior_eval/orbench --model vllm/<org>/<model> \
  -T exclude_thinking_trace_for_judge=false \
  --run-config llm_behavior_eval/orbench/parity.yaml
```

`-T exclude_thinking_trace_for_judge=false` keeps the answer's thinking trace
for the judge and the heuristic, as the legacy engine did. A run config holds
only inference settings, so this task argument stays on the command line.

The model under test runs on `vllm/` (or an OpenAI-compatible endpoint), which
reads `temperature`, `seed` and `extra_body` from the file; `hf/` cannot run
this scorer (see above). The grader stays on `hf/`.

| Setting | Value | Legacy source |
| --- | --- | --- |
| Answer decoding | `temperature: 0.0` | `EvaluationConfig.sample=False` |
| Answer budget | `max_tokens: 2048` | `FAMILY_TOKEN_DEFAULTS["refusal"]["max_answer_tokens"]` |
| Answer thinking | on: `extra_body.chat_template_kwargs.enable_thinking: true` (vLLM/SGLang) | `FAMILY_TOKEN_DEFAULTS["refusal"]["enable_thinking"]` |
| Judge decoding | `temperature: 0.0`, `do_sample: false` | `FAMILY_TOKEN_DEFAULTS["refusal"]["sample_judge"]=False` |
| Judge budget | `max_tokens: 128` | `FAMILY_TOKEN_DEFAULTS["refusal"]["max_judge_tokens"]` |
| Judge thinking | off: `enable_thinking: false` | the legacy judge always applied its chat template with `enable_thinking=False` |
| Seed (both roles) | `42` | `DatasetConfig.seed` |
| Sample cap | `eval_config.limit: 500` | `EvaluationConfig.max_samples` |
| Judge model | `hf/google/gemma-3-12b-it` | parity evidence uses Gemma 3 12B judges |

To keep parity settings with another judge endpoint, copy the file and change the
grader's `model`, `args` or `base_url`; for a vLLM or OpenAI-compatible judge,
replace the HF `args` with `extra_body: {chat_template_kwargs: {enable_thinking:
false}}` in the grader `config`.

## Parity status and known gaps

Deterministic tests compare the prompt, parser, heuristic phrases, shuffled
selection, every metric and per-sample diagnostics against the legacy evaluator.
Mock and loopback API tests cover native Inspect runs and an independent
`openai-api` grader. These tests do not establish live-model, GPU or provider
parity.

* The legacy judge received the answer model's decoded chat-templated prompt as
  the question, and inputs were truncated to 1024 tokens. Native runs use the
  raw question without truncation. Score metadata `prompt_parity` records this.
* The legacy engine stripped thinking traces only when its thinking tokens were
  configured (off by default). Native runs strip inline `<think>...</think>` by
  default; pass `-T exclude_thinking_trace_for_judge=false` to turn this off.
* Custom chat-template thinking (for example Muse Glimmer) has no native
  equivalent.
