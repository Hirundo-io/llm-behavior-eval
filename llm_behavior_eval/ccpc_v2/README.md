# CCPC-500-v2 with native Inspect

Measures CCP-aligned political censorship on CCPC-500-v2, the owner-frozen
500-prompt set from RSCH-250, with the binary `ccpc_binary_v4` judge rubric.
Results are labelled separately: they are **not comparable** with CCPC-500
(v1) results, which come from a different prompt population.

```sh
inspect eval llm_behavior_eval/ccpc_v2 \
  --model openai-api/answer/MODEL --model-role grader=openai-api/judge/MODEL
```

The grader role is required and never falls back to the model under test. The
task sets no inference options for either role. For the legacy settings, pass
the run config, which also supplies the grader role:

```sh
inspect eval llm_behavior_eval/ccpc_v2 --model vllm/ORG/MODEL \
  --run-config llm_behavior_eval/ccpc_v2/parity.yaml
```

## Implementation

The package imports no legacy-engine code. It owns its dataset contract,
rubric, verdict parsing and metrics, and uses the shared foundation
(`llm_behavior_eval.common`, see `INSPECT_MIGRATION.md`) only for what every
native benchmark shares: `common.dataset.load_dataset` for the pinned Hub load
(through `inspect_evals_utils`' retrying `hf_dataset`), local and
`save_to_disk` sources; and `common.scoring` for the stop reason, the answer
the judge sees (`judged_answer`), the grader call (`ask_grader`) and
`completion_scorer()`.

| File | Contents |
| --- | --- |
| `dataset.py` | Pinned coordinates, local sources, full-source validation, `record_to_sample` |
| `prompts.py` | Rubric `ccpc_binary_v4`, verbatim from the legacy engine; JSON judge input; verdict parser |
| `scorer.py` | The judge scorer (the task also lists the shared `completion_scorer()`) |
| `metrics.py` | `censorship_rate`, `stderr`, `judged_fraction`, `incomplete_response_rate` |
| `task.py` | The task, its version and contract metadata |
| `parity.yaml` | Legacy settings for both roles as an Inspect run config |

## Dataset

| | |
| --- | --- |
| Hub dataset | `hirundo-io/ccpc-500-v2` (**private**), config `default`, split `test` |
| Revision | `a00eee8cf918ee0d00197a9ca1aebb097756c849` (owner-frozen R8) |
| Rows | 500, in published order |
| Sample ID | `benchmark_id` |
| Columns | Exactly `benchmark_id`, `question`, `topic`, `request_form`, all non-empty strings |
| Input | `question`; `benchmark_id`, `topic` and `request_form` are sample metadata |
| Fingerprint | `a4de18b4fc728e45c2f8aec01d025255a3892530fb15835b370b31fc537705bc` |

The repository is private, so loading needs a token with access:
`LLM_BEHAVIOR_EVAL_DATASET_TOKEN` when set, otherwise `HF_TOKEN` or a cached
login. The token is read when the task is built and is never a task argument.
Redistribution rights are unresolved: do not copy rows into code, tests, logs
shared outside the organization, or issues.

Every source, Hub or local, is validated in full before `--limit` applies:
exactly 500 rows with exactly the four columns as non-empty strings, unique IDs,
and a SHA-256 fingerprint of `(question, benchmark_id)` for every row in
published order (the same serialization as the CCPC-500 v1 fingerprint).
There is no shuffle, so `--limit N` takes the first N rows, as in the legacy
engine.

### Offline sources

`-T dataset_path=<folder>` reads a local copy instead of the Hub: a folder
`datasets.load_dataset` reads (a Hub snapshot, or one holding `test.jsonl` or
`test.parquet`), or a `save_to_disk()` folder (a `DatasetDict` must contain
`test`). It must hold the same rows as the frozen release. A `save_to_disk()`
folder is only read; Inspect caches a `load_dataset` folder under its path.

## Task arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `dataset_path` | none | Local source instead of the pinned Hub dataset |
| `judge_sees_answer_reasoning` | `false` | Give the judge the answer's parsed reasoning too |

## Grading and metrics

The model sees the question plus a trailing newline, with no system prompt.
The judge gets the rubric as its system message and a JSON object
`{"question", "response"}` (non-ASCII kept, compact separators) as its user
message; its chat template must support a system role. The verdict must be a
bare `True` or `False` (trimmed, any case); anything else is unparseable.
Every answer is judged, whatever its stop reason.

Metrics are unreduced: each (sample, epoch) is an observation.

| Metric | Meaning | Direction |
| --- | --- | --- |
| `censorship_rate` | Share of judgments that are censored; **NaN if any verdict is unparseable** | lower is better |
| `stderr` | Standard error of the rate, clustering each sample's epochs; NaN with the rate | diagnostic |
| `judged_fraction` | Share of judgments with a parseable verdict | diagnostic |
| `incomplete_response_rate` | Share of answers stopped by `max_tokens` or `model_length`, from the shared `completion_scorer`; unscored when the provider reports no stop reason | lower is better |

## Reasoning (thinking) handling

The task does no thinking stripping of its own and has no thinking-trace
arguments. The judge gets the answer's `ModelOutput.completion` and the
verdict is parsed from the grader's `completion`; both leave out reasoning
that Inspect's provider parsed out (a separate reasoning field, or a complete
`<think>...</think>` block on OpenAI-compatible providers). Reasoning Inspect
does not parse (a closing tag only, an unclosed block, a second block, any
reply from `hf/`) stays in the text: the judge sees it, and a verdict inside a
grader's unparsed reasoning makes the reply unparseable. Serve reasoning models
with a reasoning parser or turn thinking off per role, as `parity.yaml` does.

`-T judge_sees_answer_reasoning=true` gives the judge the answer with its
parsed reasoning, rendered by `common.scoring.judged_answer` as `<reasoning>` blocks,
as the legacy engine judged the whole answer.

## Parity

`parity.yaml` holds the legacy CCPC settings: answers greedy, 8192 tokens,
thinking off; grader `hf/google/gemma-4-26B-A4B-it`, the frozen historical CCPC
judge, at revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, greedy, 4096
tokens, thinking off; `limit: 500`. Inspect's `hf/` provider applies the
revision to the model weights only; the judge's tokenizer and chat template
load from `main`. A prospective "Sol-6.1 High" judge is out of scope until it
qualifies.

The task declares Inspect `version=1` (`TASK_VERSION` in `task.py`); bump it
whenever a change can alter results. No live-model, GPU or private-dataset run
has been done.
