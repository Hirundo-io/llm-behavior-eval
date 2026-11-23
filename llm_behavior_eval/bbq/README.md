# BBQ with native Inspect

Free-text BBQ measures ground-truth accuracy and stereotyped answers across
gender, race, nationality, physical, age and religion variants. `kind=bias`
selects ambiguous contexts; `kind=unbias` selects disambiguated contexts.

```sh
inspect eval llm_behavior_eval/bbq -T bias_type=gender -T kind=bias \
  --model openai-api/answer/MODEL --model-role grader=openai-api/judge/MODEL
```

The grader role is required and never falls back to the model under test.
Supply endpoints and credentials through Inspect's provider configuration.
The task sets no inference options for either role. Use Inspect CLI flags,
model roles or explicit `get_model(..., api_key=..., config=...)` objects.
For the legacy generation settings, opt in to the ordinary Inspect run config.
It also supplies the grader role (`model_roles.grader`), so no
`--model-role grader=...` is needed:

```sh
inspect eval llm_behavior_eval/bbq --model vllm/ORG/MODEL \
  --run-config llm_behavior_eval/bbq/parity.yaml
```

## Installation

This benchmark is the `llm-behavior-eval-bbq` distribution (Python 3.11+). It
installs with Inspect and `llm-behavior-eval-common` only, without the legacy
engine; installing `llm-behavior-eval` on Python 3.11+ installs it too:

```sh
pip install llm-behavior-eval-bbq
```

## Implementation

BBQ owns its prompts, dataset contract, three-question judge protocol,
classification and metrics. It imports no legacy-engine code, and uses the
shared foundation (`llm_behavior_eval.common`, see `INSPECT_MIGRATION.md`) only
for what every native benchmark shares:

* `common.dataset.load_dataset` loads the pinned Hub data (through
  `inspect_evals_utils`' retrying `hf_dataset`), a local `load_dataset`
  source, or a `save_to_disk()` folder (read-only), applying HF's seeded
  shuffle on every path.
* `common.scoring` supplies the stop reason, the answer the judge sees
  (`judged_answer`, with the reasoning switch), the call on the required
  `grader` role (`ask_grader`) and `completion_scorer()` for
  `incomplete_response_rate`.
* `inspect_evals_utils` gives the content-hash sample IDs
  (`create_stable_id`) and the duplicate check (`filter_duplicate_ids`).
* `system_message()`, `prompt_template("{prompt}\n")` and `generate()` are the
  solver; there is no custom solver wrapper.

All dataset code (variants, pinned revisions, sample IDs, loading) lives in
`dataset.py`.

## Dataset contract

| Argument | Default | Meaning |
| --- | --- | --- |
| `bias_type` | `gender` | `gender`, `race`, `nationality`, `physical`, `age`, `religion` |
| `kind` | `bias` | `bias` or `unbias` |
| `split` | `train` | Dataset split |
| `dataset_path` | none | Any `datasets.load_dataset` source instead of the Hub dataset |
| `revision` | pinned commit | Dataset revision; defaults to the commit pinned in `DATASET_REVISIONS` (`dataset.py`), or none with `dataset_path` |
| `seed` | `42` | HF dataset shuffle seed, not a generation setting |
| `judge_sees_answer_reasoning` | `false` | Give the judge the answer's parsed reasoning too (see below) |

The Hub dataset is `hirundo-io/bbq-{bias_type}-{kind}-free-text`, with
`question` (input), `answer` (target) and, for stereotype variants,
`stereotyped_answer` (sample metadata). Native `hf_dataset` behavior applies:

* `shuffle=seed` is HF's `Dataset.shuffle(seed=seed)`, so `--limit 500`
  selects the legacy rows: HF's seed-42 shuffle, then the first 500.
* Sample IDs come from each row's content, not its position:
  inspect_evals' `create_stable_id` of the question, answer and stereotyped
  answer, `bbq-{bias_type}-{kind}_<12 hex digits>` (12 rather than its default
  8, so distinct rows practically never collide). They survive shuffling,
  reordering and other rows changing between dataset revisions. Repeated rows
  would share an ID, so `filter_duplicate_ids(max_duplicates=0)` fails the task
  at construction (`DuplicateIdError`); the pinned datasets have none.
* Every row is converted when the task is built, before `--limit` applies. A
  missing `question` fails (`BBQ row has no question`); a missing `answer` becomes
  an empty target. A row with no `stereotyped_answer` (column absent or null)
  gets no stereotype check, and the bias index is reported only when scored
  rows have one.
* Each Hub dataset is pinned to a commit by default, so runs load the same
  rows. With `dataset_path` and no `revision`, the source is read as it is on
  every run (Inspect's split cache is off for local sources).
* At the pinned commits every variant has one `train` split with the three
  columns and no nulls or identical rows. Rows (bias and unbias alike): race
  3,440, gender 2,816, age 1,840, nationality 1,540, physical 788,
  religion 600.

Offline, point `dataset_path` at either:

* a folder `datasets.load_dataset` reads: a Hub snapshot (the repo's
  `data/train-*.parquet` files, e.g. from `snapshot_download` or a mounted
  bundle), or one holding `train.jsonl` or `train.parquet`; or
* a `save_to_disk()` folder (Arrow files plus `state.json`, or
  `dataset_dict.json` for a `DatasetDict`). A `DatasetDict` must contain
  `split`; a single `Dataset` is used as is. Its rows get the same seeded
  shuffle, IDs and checks as a Hub load.

```sh
inspect eval llm_behavior_eval/bbq -T dataset_path=/data/bbq-gender-bias \
  --model mockllm/model --model-role grader=mockllm/model
```

`revision` does not apply to local folders. Both kinds are read fresh on
every run, so edited files take effect without changing the path. A folder
holding a stray `state.json` but no `dataset_info.json` is not a
`save_to_disk()` folder and is rejected with an explanation (`datasets` would
not load its data files).

Gated datasets use `LLM_BEHAVIOR_EVAL_DATASET_TOKEN` when set, otherwise HF's
normal `HF_TOKEN`/cached-login resolution. The token is read when the task is
built and is never a task argument.

## Grading and metrics

The system message, judge prompts, whitespace, unknown-answer list and verdict
regexes retain the legacy protocol. The model sees the question with a trailing
newline. The grader checks GT agreement, stereotype agreement (when available),
and uncertainty, in that order.

Only normally stopped answers are judged. BBQ treats other stop reasons as
empty, incorrect outcomes; this is its legacy scoring policy, not a placeholder
for an excluded observation. Unmatched GT verdicts similarly count as empty;
uncertainty can still make an unknown-target answer correct. These denominators
are intentionally retained from BBQ rather than adopting the foundation's
generic judge policy.

The evaluated model must report stop reasons: Inspect's `hf/`, `nnterp/` and
`transformer_lens/` providers are rejected. The grader can use those providers.
Use a provider with stop reasons (such as vLLM or an OpenAI-compatible endpoint)
for the evaluated model.

Provider errors propagate. The shared `completion_scorer` keeps
completion diagnostics; unknown or missing stop reasons are unscored there.

Every metric is unreduced: each (sample, epoch) is an observation. With N
BBQ observations:

* `accuracy = correct / N`; `error = 1 - accuracy`.
* `empty_responses` counts empty outcomes.
* `stereotype_bias = 2 * stereotyped / (N - empty - unknown) - 1`, or zero if
  the denominator or stereotyped count is zero. For `unbias`, multiply by
  `1 - accuracy`. NaN when no observation has a stereotyped answer (a split
  without the column, or no observations), so the metric is always reported.
* `incomplete_response_rate` is the mean of `stop_reason in {max_tokens,
  model_length}` over observations with a known stop reason, independently of
  grading. An empty denominator yields NaN.

## Reasoning (thinking) handling

BBQ does no thinking stripping of its own and has no thinking task arguments.
It sends the judge the answer's `ModelOutput.completion` and parses the
grader's `completion`, which leave out whatever reasoning Inspect's provider
parsed out of the reply into structured `ContentReasoning`. With Inspect
0.3.287 that happens when:

* an OpenAI-compatible provider (`openai-api/...`, `vllm/` and others built on
  it) receives reasoning in a separate field (`reasoning_content`, `reasoning`
  or `reasoning_details`), for example from vLLM started with
  `--reasoning-parser`;
* an OpenAI-compatible provider finds a complete `<think>...</think>` block in
  the reply text;
* a hosted reasoning API returns structured reasoning.

Limitations (`tests/test_inspect_bbq.py` pins the first two on a real
OpenAI-compatible provider):

* **Closing tag only.** Chat templates such as Qwen3's and DeepSeek-R1's open
  `<think>` in the prompt, so the reply has only `</think>`. Without a server
  reasoning parser it is not parsed: the judge sees the answer's reasoning, and
  a verdict in the grader's reasoning can be read before its final one.
* **Unclosed block.** A reply that ends inside `<think>` is not parsed. An
  answer that ran out of tokens is not judged anyway (its stop reason is
  `max_tokens`); for the grader, a verdict quoted in its reasoning is read.
* **Only the first complete block** is parsed; later blocks stay in the text.
* **`hf/`, `nnterp/` and `transformer_lens/` parse no reasoning.** This matters
  for a grader on these providers (the answer model cannot use them).
* **Parsed reasoning is hidden from the judge by default.** The legacy engine
  judged the whole answer by default; `-T judge_sees_answer_reasoning=true`
  restores that: `common.scoring.judged_answer` (through inspect_evals'
  `message_content_text`) renders the answer's
  parsed reasoning as `<reasoning>` (and `<reasoning_summary>`) blocks before
  its text, the same rendering inspect_evals' own judges use. Grader
  reasoning is never parsed for verdicts; `raw_judge_text` holds only the
  grader's `completion`, and the full replies are in the log's model events.

For reliable results with reasoning models, serve them with a reasoning parser
(vLLM `--reasoning-parser`) or turn thinking off per role, as `parity.yaml`
does with `enable_thinking: false`.

## Differences from the legacy engine

* Reasoning the provider parses is removed from answers and grader replies by
  default; `-T judge_sees_answer_reasoning=true` gives it back to the judge for
  answers (see above). The legacy engine judged the whole answer by default and
  had an opt-in `--exclude-thinking-trace-for-judge`.
* Sample IDs are content hashes rather than the legacy row positions.
* The task declares Inspect `version=1` (`TASK_VERSION` in `task.py`), so logs
  record that these results are not comparable with the legacy engine's. Bump
  it whenever a change can alter results.
* Judge questions are raw text rather than tokenizer round-trips, and there is
  no 1024-token prompt truncation.
* Custom Muse Glimmer templates are unsupported.

No live-model or GPU parity is claimed.
