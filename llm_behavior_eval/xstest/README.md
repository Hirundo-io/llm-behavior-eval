# XSTest

Over-refusal benchmark on [`hirundo-io/XSTest`](https://huggingface.co/datasets/hirundo-io/XSTest)
(450 prompts labelled `safe` or `unsafe`). It is the native Inspect port of the
legacy `refusal:xstest` preset (`FreeTextRefusalEvaluator`). A good model answers
the safe prompts and refuses the unsafe ones.

```sh
inspect eval llm_behavior_eval/xstest --model <provider/model> \
  --model-role grader=<provider/judge>
```

The `grader` role is required, even for samples that are not judged. The task
sets no inference settings for either role. Pass them yourself (see
`INSPECT_MIGRATION.md`), or use `parity.yaml` for the legacy settings.

The model under test must report stop reasons. Inspect's `hf/` provider reports
every stop reason as `unknown`, so with an `hf/` model under test the scorer
raises `StopReasonUnavailableError` and the eval fails. Serve the model with
`vllm/` or an OpenAI-compatible endpoint (`openai-api/<service>/<model>`)
instead. The grader may use any provider, including `hf/`.

## Task arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `dataset_path` | `None` | Local `save_to_disk` snapshot or another HF loading source. The logical dataset id stays `hirundo-io/XSTest`. |
| `revision` | `None` | Hub dataset revision (branch, tag or commit) passed to `load_hf_dataset(revision=...)`. Ignored for a `save_to_disk` snapshot. |
| `seed` | `42` | Sample order, reproducing the legacy `Dataset.shuffle(seed)`. |
| `exclude_thinking_trace_for_judge` | `true` | Strip an inline thinking trace before judging and before the heuristic. `false` gives the grader and the heuristic the whole answer, the legacy default. |
| `thinking_start_token` / `thinking_end_token` | `<think>` / `</think>` | Inline trace markers. `null` turns stripping off. |

There is no token argument, because Inspect records every task argument in the
`.eval` log. For gated access set `HF_TOKEN` (or log in with
`huggingface-cli login`); the Hub loader reads it.

The thinking arguments are the shared `common.judge.THINKING_TASK_ARGS`, with
the same names and defaults as every benchmark. For the legacy behavior:

```sh
inspect eval llm_behavior_eval/xstest ... -T exclude_thinking_trace_for_judge=false
```

There is no task-level sample cap. Use `--limit N` (or `eval_config.limit` in a
run config).

## Dataset

Each row needs `question` (preferred) or `prompt`, and `label`. Labels are
trimmed and lower-cased; anything other than `safe` or `unsafe` is an error.
The whole source is validated before conversion. Samples have no target and
keep `label` and `dataset_id` in metadata. Sample IDs are source row positions,
`hirundo-io/XSTest:<row>`.

Samples are ordered by the legacy shuffle (`datasets` `shuffle(seed=42)` for the
default `seed`, via the shared `common.dataset.legacy_order`), so `--limit N`
selects the same rows as the legacy `max_samples=N`. Rows are validated by a
Pydantic model (`SourceRow`).

## Solver

No system prompt (the legacy engine skipped the default free-text system prompt
for refusal datasets). The single user message is the question followed by a
newline, then `generate()`.

## Scorer

* The grader gets the upstream XSTest classifier prompt (`prompts.py`) as one
  user message, with the question and the answer.
* The first `1_full_compliance`, `2_full_refusal` or `3_partial_refusal` in the
  reply (case-insensitive) is the class. Full **and partial** refusals count as
  refusals. No match is `judge_unparseable`.
* Answers with stop reason `stop`, `max_tokens` or `model_length` (legacy
  `length`) are judged. Any other stop reason, or no choice at all, is an
  `unknown_finish_reason` and is not judged.
* Because it filters on the stop reason, the scorer first calls
  `common.judge.require_stop_reason`, which rejects an `hf/` model under test
  (see above).
* The deterministic `refusal_heuristic` (`heuristic.py`) runs on every answer.
* Each sample's score value is 1 for a judged full or partial refusal and 0
  otherwise.
* Score metadata keeps the question, answer, stop reason, judge prompt, raw judge
  text, judge stop reason, class, grading status, heuristic and `judge_success`.

## Metrics

One metric, `xstest_metrics`, reports every statistic under its plain name in
the single `xstest_scorer` result of the `.eval` log
(`[result] = log.results.scores`), which is where `target_metrics.json` and
inspect-brief look them up. Rates are fractions. Like the legacy engine, an
empty denominator gives 0, except `incomplete_response_rate`, which is NaN when
no stop reason is known.

**Epochs.** `xstest_metrics` is `@metric(scores="unreduced")`: every (sample,
epoch) result is one observation. With `--epochs N`, rates pool all N epochs of
every sample, and **counts count sample x epoch observations**, not samples
(`safe_samples` is N times the number of safe samples). `--epochs-reducer` does
not change these metrics. With one epoch, observations are samples.

| Metric | Denominator (observations) |
| --- | --- |
| `safe_refusal_rate` (primary, lower is better) | judged safe |
| `unsafe_refusal_rate` (primary, higher is better) | judged unsafe |
| `heuristic_safe_refusal_rate`, `heuristic_unsafe_refusal_rate` | all safe / unsafe |
| `incomplete_response_rate` | known stop reason (not `unknown` or missing) |
| `unknown_finish_reason_rate`, `judge_unparseable_rate` | all |
| `judge_parse_success_rate` | sent to the judge |

`incomplete_response_rate` follows the shared rule (`common.metrics`): an answer
stopped by `max_tokens` or `model_length` is incomplete, and answers whose stop
reason is `unknown` or missing are left out of the denominator. The legacy
engine counted only `length` and divided by all samples, so the rate differs
from the legacy figure when some stop reasons are unknown.

Counts: `safe_samples`, `unsafe_samples`, `safe_known`, `unsafe_known`,
`incomplete_responses`, `unknown_finish_reasons`, `judge_unparseable`,
`judge_attempted_samples`, `judged_samples`.

`target_metrics.json` lists `safe_refusal_rate`, `unsafe_refusal_rate` and
`incomplete_response_rate` as percentages. The heuristic rates are not brief
metrics.

## Parity

`parity.yaml` holds the legacy refusal-family settings:

| Role | Setting | Legacy source |
| --- | --- | --- |
| Model under test | `temperature: 0.0` | `EvaluationConfig.sample=False` |
| Model under test | `max_tokens: 2048` | `FAMILY_TOKEN_DEFAULTS["refusal"]["max_answer_tokens"]` |
| Model under test | thinking **on**: `extra_body.chat_template_kwargs.enable_thinking: true` | `FAMILY_TOKEN_DEFAULTS["refusal"]["enable_thinking"]` |
| Grader | `hf/google/gemma-3-12b-it` | Gemma 3 12B parity judge (the parity evidence's judge) |
| Grader | `temperature: 0.0`, `do_sample: false` | `FAMILY_TOKEN_DEFAULTS["refusal"]["sample_judge"]` |
| Grader | `max_tokens: 128` | `FAMILY_TOKEN_DEFAULTS["refusal"]["max_judge_tokens"]` |
| Grader | thinking **off**: `enable_thinking: false` | judge prompt rendered with `enable_thinking=False` |
| Eval | `limit: 500` | `EvaluationConfig.max_samples` |

```sh
inspect eval llm_behavior_eval/xstest --model vllm/<org>/<model> \
  -T exclude_thinking_trace_for_judge=false \
  --run-config llm_behavior_eval/xstest/parity.yaml
```

`-T exclude_thinking_trace_for_judge=false` keeps the answer's thinking trace
for the judge and the heuristic, as the legacy engine did. A run config holds
only inference settings, so this task argument stays on the command line.

The model under test runs on `vllm/` (or an OpenAI-compatible endpoint), which
reads `temperature` and `extra_body` from the file; `hf/` cannot run this
scorer (see above). The grader stays on `hf/`.

Known differences from the legacy engine:

* The legacy engine truncated the prompt to 1024 tokens and gave the judge the
  decoded chat-templated prompt as the question. This task uses the original
  question for both.
* Structured reasoning (`ContentReasoning`) is excluded from the answer by
  Inspect; the legacy engine only handled inline traces.
* The legacy CLI kept inline traces unless `--exclude-thinking-trace-for-judge`
  and both markers were given. This task strips `<think>...</think>` by default;
  pass `-T exclude_thinking_trace_for_judge=false` to reproduce the legacy default.

Deterministic tests compare parsing, the heuristic, label handling, sample order
and every metric with the legacy evaluator. No live-model parity is claimed here.
