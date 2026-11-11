# Native Inspect foundation

Legacy CLI, evaluators, and public imports remain available. No benchmark has
migrated in this layer; each benchmark arrives in its own PR. Inspect 0.3.287 is
tested and pinned to `<0.3.288`. New dependencies are locked to releases at
least 7 days old. Every Inspect behavior described below was
checked against that version's source and is covered by tests in
`tests/test_inspect_foundation.py`, `tests/test_inspect_packages.py` and
`tests/test_inspect_template.py`.
Revalidate before widening the pin. Inspect requires Python 3.11+, so its
dependency has a Python-version marker and Inspect tests are not collected on
Python 3.10, which remains legacy-only.

```sh
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -e .
source .venv/bin/activate
inspect list tasks llm_behavior_eval
inspect eval llm_behavior_eval/inspect_template --model mockllm/model
```

The template is a wiring demonstration: its **default grader is a mock that always
returns SCORE: 1**. Real benchmarks bind no grader and fail unless one is passed.

This layer writes `.eval` logs only. MLflow tracking of Inspect runs arrives with
the reporting PR ([#207](https://github.com/Hirundo-io/llm-behavior-eval/pull/207)).

## Design rules

1. **Benchmark code sets no inference settings**, for the model under test or for
   the grader: no temperature, top_p, top_k, max_tokens, reasoning or thinking
   switches. Unset settings fall through to the serving stack's defaults (HF
   `generation_config.json`, the vLLM/SGLang server's generation config, the chat
   template's thinking default, hosted-API defaults). Users set values through
   Inspect's own mechanisms, separately for each role (see below). Every Inspect
   provider works for either role, including `openai-api/<service>/<model>`.
2. **Parity is an explicit config file.** Each benchmark may ship a `parity.yaml`
   with the legacy engine's exact settings. It applies only when passed.
3. **Benchmarks are self-contained.** `common/` holds benchmark-agnostic
   mechanics only. Share mechanics, not semantics.
4. **No credentials in task arguments.** Inspect writes every task argument
   into the `.eval` log, so no task takes a token, key, secret or password.
   Credentials come from environment variables, set directly or with the
   `credentials` context manager, or from Inspect's per-model `api_key`, which
   Inspect does not log (see [Credentials](#credentials)).

## Benchmark layout

Each benchmark is a regular subpackage `llm_behavior_eval/<benchmark>/` (it has an
`__init__.py`) that owns:

| File | Contents |
| --- | --- |
| `task.py` | The `@task` function. No `config=`, no default grader. Takes the [thinking-trace task arguments](#thinking-trace-task-arguments). |
| `dataset.py` | Loading, full-source validation, `record_to_sample`. |
| `prompts.py` | System and judge prompts, copied verbatim from the legacy engine. |
| `solver.py` | Solver chain. No GenerateConfig. |
| `scorer.py` | Scorer, usually wrapping `common.judge.judge_scorer`. |
| `metrics.py` | Benchmark metrics, when `common.metrics` is not enough. Every one is `@metric(scores="unreduced")` (see [Epochs](#epochs)). |
| `README.md` | What it measures, task arguments, metrics, parity status. |
| `parity.yaml` | Legacy-engine settings as an Inspect run config (see below). |
| `target_metrics.json` | Report metrics for each of its tasks (see below). |

Its tests live outside the package, in `tests/test_inspect_<benchmark>.py`:
registry discovery, dataset conversion, scorer, parity-config tests, an
`--epochs 2` regression and `reduced_metrics(task) == []` (see
[Epochs](#epochs)), and a check that inspect-brief finds every
`target_metrics.json` name in a real `.eval` log (`missing_target_metrics`, see
[Target metrics](#target-metrics)).

Generic test helpers live in `tests/inspect_helpers.py` (importable as
`inspect_helpers`; `pytest`'s `pythonpath` adds `tests/`) and
`tests/conftest.py`; use them instead of copying:

| Helper | Use |
| --- | --- |
| `generation_settings(config)` | The inference fields a `GenerateConfig` sets; `{}` means nothing was set. |
| `eval_metrics(log)` | Metrics of a successful log's single, unreduced `EvalScore`. |
| `sample_scores(log, scorer)` | Every sample's `Score` from one scorer, in log order. |
| `rate(metric)`, `mapping(metric)` | A metric typed as returning a float or a mapping, to call it on `SampleScore`s directly. |
| `only_log(log_dir)` | The one `.eval` log in a directory. |
| `run_inspect_cli(*args, cwd=...)` | `python -m inspect_ai ... --display none`; asserts it succeeded. Pass a `cwd` other than the repository root (see [Target metrics](#target-metrics)). |
| `mock_grader(reply)` | A `mockllm/grader` model that always replies `reply`. |
| `patch_scorer_grader(monkeypatch, scorer_module, reply)` | Routes a scorer module's `get_model` to a canned reply, for scorers called directly; returns the prompts it got. |
| `snapshot(directory, rows)` | Saves rows as a `save_to_disk` snapshot and returns its path. |
| `mock_parity_config(package, directory, grader="mockllm/model")` | Writes the shipped `parity.yaml` with only the grader swapped for a mock model. |
| `assert_grader_required(task, tmp_path)` | Runs `task` without a grader role and asserts it fails, naming the grader. |
| `assert_no_credentials_logged(log, secret)` | Asserts no file of a `.eval` log contains `secret`. |
| `openai_judge` fixture (conftest), `local_openai_judge(reply)` | A loopback OpenAI-compatible chat endpoint (`LocalOpenAIJudge`): `.base_url`, `.reply` (text, or a function of the request body), `.requests` (bodies), `.paths` and `.authorizations` (`Authorization` headers). Use it for `openai-api` grader tests instead of patching `Model.generate`: a patched memoized model leaks into later tests. `message_text(message)` reads a request message's text. |

Registry names are global. Inspect keeps one registry per object type, keyed by
`llm_behavior_eval/<name>`, and a later registration under a name silently
replaces the earlier one; `inspect score` and anything else that rebuilds a
metric, scorer or solver from a log then runs the wrong benchmark's code. Name
them after the package: `<benchmark>_metrics`, `<benchmark>_scorer`,
`<benchmark>_<step>` for solvers. A single metric that returns a mapping keeps
the reported names plain (`accuracy`, `stderr`, ...) while its registry name
stays package specific.

Rules: no imports between benchmarks; use absolute imports in task files
(Inspect also loads them by file path). `llm_behavior_eval/inspect_template/` is
the reference copy of this layout. Copy it, then replace its mock grader,
illustrative parity values and example target metrics.

### Adding a benchmark touches no shared file

A benchmark PR adds exactly two things: its package directory
`llm_behavior_eval/<benchmark>/` and its test file
`tests/test_inspect_<benchmark>.py`. It does not edit `_registry.py`, any
root-level metrics file, the root `README.md`, `.gitignore` or `pyproject.toml`:

* **Registration is automatic.** `llm_behavior_eval/_registry.py` (Inspect's
  `inspect_ai` entry point) imports the `task` module of every subpackage that
  has one, in sorted package order, with `pkgutil`/`importlib`. Discovery works
  the same from a source checkout, an editable install and an installed wheel.
  Tasks register under the `llm_behavior_eval/` namespace, are bound as
  attributes of `_registry` and listed in its `__all__` (sorted). A task module
  that fails to import fails registration, as an explicit import would; two
  packages registering the same task name raise an error, and so does any
  metric, scorer or solver name defined by two modules
  (`_registry.check_unique_registry_names()`). Packages without a
  `task` module (`common/`, shared helpers) are not benchmarks.
* **Target metrics are per package.** `llm_behavior_eval/<benchmark>/target_metrics.json`
  holds only that benchmark's entries; `common.target_metrics` merges them.
  `.gitignore` excludes `*.json` but re-includes
  `llm_behavior_eval/*/target_metrics.json`, and the wheel ships it with the
  package (`tool.setuptools.package-data`).
* **Documentation is per package.** The benchmark's own `README.md` documents
  it; the root README points to these files generically.

`tests/test_inspect_packages.py` enforces the convention: every registered task
has target metrics and vice versa and exposes the thinking-trace task arguments,
every metric defined in the package and every metric a scorer declares is
unreduced, package files are tracked and ship in the wheel, and an installed
wheel discovers the same tasks and metrics.

### `inspect_template` is registered on purpose

The template is discovered like any benchmark, so `inspect list tasks` shows it
and `inspect eval llm_behavior_eval/inspect_template --model mockllm/model`
works. It is the only task in this foundation, and running it is the
end-to-end check that discovery, the entry point and the CLI work for a package
laid out as benchmarks copy it; excluding it would need a special case in
discovery that a copied benchmark could inherit by mistake. It needs no
credentials or weights (mock grader, one inline record). Tools that run or
report "all benchmarks" should skip `llm_behavior_eval/inspect_template`.

## Shared API (`llm_behavior_eval.common`)

* `dataset.load_hf_dataset(dataset_id, *, dataset_path=None, revision=None)`
  loads the complete split of an HF dataset or a `save_to_disk` snapshot.
  `revision` pins the Hub dataset revision. `dataset_path` changes where data is
  read from, not the logical id. Selects `train` when present, otherwise
  requires exactly one split. It takes no token: gated Hub datasets
  authenticate with `credentials.resolve_dataset_token()`,
  `LLM_BEHAVIOR_EVAL_DATASET_TOKEN` then `HF_TOKEN` (see
  [Credentials](#credentials)).
* `credentials.resolve_dataset_token()` returns the dataset credential from the
  environment. `credentials.credentials(*, dataset_token=None, model_env=None,
  judge_env=None)` sets credential variables for a `with` block and restores
  them (see [Credentials](#credentials)). `credentials.DATASET_CREDENTIAL_VAR` is
  `"LLM_BEHAVIOR_EVAL_DATASET_TOKEN"`.
* `dataset.legacy_order(num_rows, seed=LEGACY_SHUFFLE_SEED)` returns source row
  positions in the legacy engine's `Dataset.shuffle(seed=42)` order. Convert
  records in that order so `--limit N` keeps the rows the legacy `max_samples`
  cap kept; keep the source position in the sample ID.
* `dataset.to_inspect_dataset(records, record_to_sample, *, dataset_id)` converts
  every record, in source order, into a `MemoryDataset`. Validate the complete
  source first. Supply stable sample IDs and subgroup metadata.
* `judge.judge_scorer(prompt, pattern, *, parse=float, only_stop=False,
  exclude_thinking_trace_for_judge=True, thinking_start_token="<think>",
  thinking_end_token="</think>")` calls the `grader` role and parses its reply.
  Prompt fields are `{question}`, `{answer}`, `{target}`, `{metadata}`. Regex
  group 1 goes to `parse`. The grader role is required and never falls back to the
  model under test. **The scorer passes no GenerateConfig**: the grader uses only
  its role's config. `only_stop=True` skips answers whose stop reason is not
  `stop`, and raises `StopReasonUnavailableError` when the model under test runs
  on a provider that reports no stop reasons, such as Inspect's `hf/` (see
  [Stop reasons](#stop-reasons)). The thinking
  arguments are the benchmark's [task arguments](#thinking-trace-task-arguments).
  The verdict regex runs on the grader's final answer, after its inline
  `<think>...</think>` trace is removed (always, whatever the task arguments);
  `raw_judge_text` keeps the whole reply. A grader reply that ends inside an
  unclosed `<think>` trace (for example, cut off at `max_tokens`) has no final
  answer, so it is a `parse_error` even if the trace contains a verdict.
  Unmatched, unparsable and non-finite (`nan`, `inf`) grades become
  `parse_error` scores. Skips and parse errors have placeholder value 0 and are
  excluded by `field_rate`; never average them with a plain mean. Score metadata
  holds the sample metadata, `stop_reason` and `raw_judge_text`.
* `judge.strip_thinking(answer, ...)` keeps only the text after the last end
  marker when both markers are configured and either appears. An unclosed start
  marker keeps the whole answer. Structured `ContentReasoning` is already
  excluded from `ModelOutput.completion`.
* `judge.THINKING_TASK_ARGS` maps the thinking-trace task argument names to their
  defaults (see [below](#thinking-trace-task-arguments)).
* `judge.require_stop_reason(state)` raises `StopReasonUnavailableError` when the
  model under test runs on a provider in `judge.NO_STOP_REASON_PROVIDERS`
  (`{"hf", "nnterp", "transformer_lens"}`). Benchmark scorers that filter or score on `stop_reason` without
  `judge_scorer(only_stop=True)` call it first.
* `metrics.field_rate(field=None, *, metadata_key=None, metadata_value=None)`
  averages scalar scores, or one field of dict scores, over eligible epoch
  observations, optionally restricted to a score-metadata subgroup.
  `metrics.incomplete_response_rate()` is the share of observations with a known
  stop reason that stopped at `max_tokens` or `model_length`. Empty denominators
  yield NaN. Both are `@metric(scores="unreduced")`.
* `metrics.scores_by_sample(scores)` groups an unreduced metric's observations by
  `sample_id`, in first-seen order. `metrics.reduced_metrics(task_or_scorer)`
  lists the metrics that are not unreduced. `metrics.EPOCH_SCORES` is
  `"unreduced"`; `metrics.INCOMPLETE_STOP_REASONS` and
  `metrics.UNKNOWN_STOP_REASON` name the stop reasons above.
* `packages.benchmark_packages(root="llm_behavior_eval")` returns the sorted
  names of subpackages that have a `task` module, without importing them.
  `packages.subpackages(root)` lists every regular subpackage.
* `target_metrics.load_target_metrics(root="llm_behavior_eval")` merges every
  package's `target_metrics.json` into one mapping with sorted task keys.
  Each file is validated by the Pydantic models `TargetMetricsFile` and
  `TargetMetric` (keys `llm_behavior_eval/<task>`, non-empty metric lists,
  objects with exactly a non-empty `name` and the JSON booleans
  `is_percentage`, `is_higher_better` and `is_normalized`), and returned in
  inspect-brief's JSON format. A task key defined twice, in one
  file or across packages, raises `ValueError`. `target_metrics_files(root)`
  maps packages to their files; `read_target_metrics_file(path)` validates one.
  `write_target_metrics(path)` writes the merged JSON for tools that need one
  file, such as inspect-brief; from a shell,
  `python -m llm_behavior_eval.common.target_metrics --output PATH` does the
  same, and without `--output` it prints the JSON.
* `parity.parity_config_path(package)` returns the installed path of a benchmark's
  `parity.yaml`. `parity.read_parity_config(path)` returns `inspect_ai.eval()`
  keyword arguments, parsed by Inspect's own `--run-config` reader. It rejects
  files that set `task`, `model`, `solver`, `tags`, `metadata` or `sandbox`.

Sample limits are Inspect's: `--limit N`, `eval(limit=N)`, or `eval_config.limit`
in a run config. There is no task-level sample cap.

## Epochs

`--epochs N` runs every sample N times. Inspect's epoch reducers keep only the
first epoch's score **metadata**, and the benchmark metrics read metadata
(`skipped`, `parse_error`, `stop_reason`, grades, subgroups). With reduced
metrics, an N-epoch run silently reports epoch 1's metadata. So every metric a
benchmark reports takes unreduced scores.

**Metric layout.**

* Every metric is declared `@metric(scores="unreduced")`. It receives one
  `SampleScore` per sample per epoch, each with its own metadata; `sample_id`
  repeats across epochs.
* `log.results.scores` holds exactly one `EvalScore` per scorer, with
  `reducer` `None` and the same metric keys as a one-epoch run. `--epochs-reducer`
  does not change these metrics, and `log.reductions` is empty. With one epoch,
  values equal the previous layout's.
* Never mix reduced and unreduced metrics on one scorer. Inspect then writes two
  `EvalScore` entries for it (a reduced view tagged `mean` and an unreduced
  one), and `log.results.scores[0].metrics` no longer holds every metric.
  Inspect built-ins (`accuracy()`, `mean()`, `stderr()`, ...) are reduced, so
  don't attach them to a benchmark scorer.
* Rates and means pool every (sample, epoch) observation. With the same number
  of epochs per sample and no exclusions, that equals the mean of per-sample
  means. Counts count observations. A standard error must treat a sample's
  epochs as one cluster: compute it over per-sample values from
  `scores_by_sample`.

**Adopting it in a benchmark.**

1. Declare every metric in `<benchmark>/metrics.py` as
   `@metric(scores="unreduced")`, including mapping-valued metrics and wrappers
   around `common.metrics`. Keep the metric names; the log keys do not change.
2. Remove Inspect built-in metrics from `@scorer(metrics=...)` and
   `Task(metrics=...)`. Replace a built-in `stderr()` with a package metric
   that clusters by sample (`scores_by_sample`).
3. Check each metric's formula reads correctly over observations: any
   denominator that was "samples" is now "samples x epochs".
4. In `tests/test_inspect_<benchmark>.py`, assert
   `reduced_metrics(<task>(...)) == []`, and add an `epochs=2` eval whose
   epochs differ (for example a truncated then a finished answer) and assert
   the metrics count both. Read metrics from the single
   `[result] = log.results.scores`.
5. `tests/test_inspect_packages.py` must pass with the benchmark merged: it
   checks every metric defined in the package and every metric a scorer
   declares.

## Target metrics

`target_metrics.json` maps each task the package registers to the metrics a
report shows, in inspect-brief's format:

```json
{
  "llm_behavior_eval/<benchmark>": [
    {
      "name": "<metric name as written in the .eval log>",
      "is_percentage": true,
      "is_higher_better": false,
      "is_normalized": true
    }
  ]
}
```

List one key per registered task (usually one, named after the package). Metric
order is kept; merged keys are sorted.

inspect-brief selects a log's targets by `log.eval.task` and finds a metric when
its name is a **key** of some `log.results.scores[i].metrics` mapping. Inspect
writes a single-value metric under its unqualified registry name (but sets that
entry's `EvalMetric.name` to the qualified `llm_behavior_eval/<metric>`), each
entry of a mapping-valued metric under the mapping key (entries whose value is
`None` are dropped), and metrics of a `{"key": [...]}` metrics dict under
qualified names, which inspect-brief cannot match. Check the real log, not the
metric definitions:

```python
from llm_behavior_eval.common.target_metrics import missing_target_metrics

[log] = inspect_eval("llm_behavior_eval/<benchmark>", model="mockllm/model", ...)
assert missing_target_metrics(log.location) == []
```

`missing_target_metrics(log)` takes an `EvalLog` or a `.eval` path, mirrors
inspect-brief's lookup and returns the names it would report as not found. It
raises when inspect-brief would not apply the targets at all: the log did not
succeed, or its task is not a key of the merged target metrics. Run the task by
its registered name or its `@task` function. `python -m inspect_ai` started from
the repository root puts the source tree first on `sys.path`, where setuptools'
egg-info hides the install metadata Inspect needs to prefix names, so the log's
task becomes `<benchmark>` and the targets do not apply; use the `inspect`
command or another working directory.

## Setting inference options per role

Inspect keeps the two roles apart. Eval-level GenerateConfig (CLI flags,
`--generate-config`, `eval(**kwargs)`) applies **only to the model under test**.
The grader gets only its role's GenerateConfig, plus operational settings
(`max_connections`, `max_retries`, `timeout`, `cache`). In 0.3.287 this is
`Model._resolve_config`. `-M`/`model_args` also reach only the model under test.

### Model under test

| Form | Example |
| --- | --- |
| CLI flags | `--temperature 0 --max-tokens 128` (also `--top-p`, `--top-k`, `--seed`, `--reasoning-effort`, `--reasoning-tokens`, `--extra-body`, ...) |
| CLI file | `--generate-config answer.yaml`, a YAML/JSON mapping of GenerateConfig fields, for example `temperature: 0.0` and `max_tokens: 128`. Unknown keys are rejected. Flags override file values. |
| Provider args | `-M name=value`, or `--model-config args.yaml` |
| Python | `eval(task, model="vllm/org/repo", model_args={"max_model_len": 8192}, temperature=0.0, max_tokens=128)` |

### Grader role

| Form | Example |
| --- | --- |
| CLI inline YAML/JSON | `--model-role 'grader={model: hf/google/gemma-4-12B-it, temperature: 0.0, max_tokens: 32, model_args: {do_sample: false}}'`. GenerateConfig fields sit at the top level and provider args go under `model_args`. |
| CLI file | `--run-config run.yaml` with `model_roles: {grader: {model: ..., config: {...}, args: {...}, base_url: ...}}`. Here GenerateConfig goes under `config` and provider args under `args`. The same file can hold the model under test's settings under `generate_config`. `--run-config` cannot be combined with `--generate-config`; a `--model-role grader=...` flag replaces the file's whole grader entry. |
| Python | `model_roles={"grader": get_model("hf/google/gemma-4-12B-it", config=GenerateConfig(temperature=0.0, max_tokens=32), do_sample=False, memoize=False)}` |

`--model-role grader=<name>` alone sets no options: the grader then runs on its
serving stack's defaults.

### OpenAI-compatible judge endpoint

`openai-api/<service>/<model>` reads `<SERVICE>_BASE_URL` and `<SERVICE>_API_KEY`
from the environment, or takes `base_url`/`api_key` as model args.

```sh
export AIRGAP_BASE_URL=http://judge.internal:8000/v1
export AIRGAP_API_KEY=...   # or a placeholder for a no-auth server
inspect eval llm_behavior_eval/<benchmark> \
  --model vllm/org/answer --temperature 0 --max-tokens 128 \
  --model-role 'grader={model: openai-api/airgap/judge, temperature: 0.0, max_tokens: 32}'
```

```python
from inspect_ai import eval
from inspect_ai.model import GenerateConfig, get_model

grader = get_model(
    "openai-api/airgap/judge",
    base_url="http://judge.internal:8000/v1",
    api_key="no-auth-placeholder",
    config=GenerateConfig(temperature=0.0, max_tokens=32),
    memoize=False,
)
eval(
    "llm_behavior_eval/<benchmark>",
    model="vllm//models/answer",
    model_roles={"grader": grader},
    temperature=0.0,
    max_tokens=128,
)
```

Without a role config, Inspect sends no temperature or token limit to the
endpoint, so the server's own defaults apply. Hosted judges use their native
provider name (for example `openai/<model>`) and provider credentials.

### Thinking and reasoning

Nothing turns thinking on or off for you. Set it per role if you need it:

* HF provider: `-M enable_thinking=false` (forwarded to the chat template), or
  `model_args: {enable_thinking: false}` in a grader role. Unset, Inspect passes
  `enable_thinking=None`, so the template's own default applies.
* vLLM/SGLang (OpenAI-compatible): `--extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'`,
  or `extra_body: {chat_template_kwargs: {enable_thinking: false}}` in a grader
  role config. Templates that use another keyword need that keyword here.
* Hosted APIs: `--reasoning-effort`, `--reasoning-tokens` or provider-specific
  fields, as each provider supports.

### Thinking-trace task arguments

By default, an inline `<think>...</think>` trace in the answer is stripped before
the grader sees it, whatever the thinking setting: the grader gets only the text
after the last end marker. Every benchmark exposes the same task arguments
(`common.judge.THINKING_TASK_ARGS`), with these names and defaults, and passes
them unchanged to `judge_scorer` or `strip_thinking`:

| Task argument | Default | Effect |
| --- | --- | --- |
| `exclude_thinking_trace_for_judge` | `true` | `false` gives the grader the whole answer. |
| `thinking_start_token` | `<think>` | Marker that opens the trace. `null` turns stripping off. |
| `thinking_end_token` | `</think>` | Marker that closes the trace. `null` turns stripping off. |

```sh
inspect eval llm_behavior_eval/<benchmark> ... -T exclude_thinking_trace_for_judge=false
inspect eval llm_behavior_eval/<benchmark> ... \
  -T thinking_start_token='<reasoning>' -T thinking_end_token='</reasoning>'
```

**The legacy engine defaulted to off**: `--exclude-thinking-trace-for-judge` was
opt-in (`exclude_thinking_trace_for_judge=False`), with no default markers.
Legacy runs set thinking per evaluator family (off except for refusal). With no
inference settings in code, chat templates such as Qwen3's think by default, so
stripping is on here. For the legacy behavior pass
`-T exclude_thinking_trace_for_judge=false`.

The grader's own inline trace is always removed before its verdict is parsed,
so a verdict quoted in its reasoning is never read as the grade.

To adopt this in a benchmark: add the three arguments to its `@task` with
defaults `True`, `THINKING_START` and `THINKING_END` from `common.judge`, and pass
them to its scorer, which passes them to `judge_scorer` or `strip_thinking`
(see `inspect_template/task.py` and `scorer.py`). Document them in its README.
`tests/test_inspect_packages.py` checks every registered task's signature.

### Stop reasons

`judge_scorer(only_stop=True)` and `incomplete_response_rate` read the answer's
stop reason. Inspect's `hf/`, `nnterp/` and `transformer_lens/` providers cannot
report one: they leave every `stop_reason` at `unknown`. `hf/` also reports the
padded batch length as every answer's output token count, so a finished answer
cannot be told from a truncated one, even after the fact. Therefore:

* `only_stop=True` with a model under test on one of these providers raises
  `StopReasonUnavailableError` (the eval fails) instead of skipping every answer.
  Serve the model with `vllm/` or an OpenAI-compatible endpoint. Scorers that
  filter on the stop reason themselves call `require_stop_reason(state)`.
* `incomplete_response_rate` counts `max_tokens` and `model_length` as
  incomplete and leaves `unknown` (and missing) stop reasons out of its
  denominator. On these providers it is NaN, not 0.
* Elsewhere, an `unknown` answer under `only_stop=True` is skipped.

The parity evidence was produced with the model under test served by vLLM,
which reports stop reasons, so these rules do not change it.

## Parity configs

A benchmark's `parity.yaml` is an ordinary Inspect run config holding only
inference settings: `generate_config` (model under test), `model_roles.grader`
(judge model, `config`, `args`) and optionally `eval_config` (for example
`limit: 500`, the legacy sample cap). It never sets `task`, `model` or `solver`,
so the model under test is still chosen on the command line, and never `tags`,
`metadata` or `sandbox`; `read_parity_config` rejects all six. It also rejects
a key repeated within any mapping, YAML or JSON, which plain YAML and Inspect's
CLI reader would silently resolve to the last value; the error names the key
and, for YAML, its line. Every shipped `parity.yaml` is checked by the tests.

```sh
inspect eval llm_behavior_eval/<benchmark> --model vllm/org/answer \
  --run-config llm_behavior_eval/<benchmark>/parity.yaml
```

From an installed package, find the file with
`python -c 'from llm_behavior_eval.common.parity import parity_config_path; print(parity_config_path("llm_behavior_eval.<benchmark>"))'`.

```python
from inspect_ai import eval
from llm_behavior_eval.common.parity import parity_config_path, read_parity_config

eval(
    "llm_behavior_eval/<benchmark>",
    model="vllm/org/answer",
    **read_parity_config(parity_config_path("llm_behavior_eval.<benchmark>")),
)
```

Like the CLI, `read_parity_config` constructs the grader model, which may load its
weights. To keep parity settings with another judge endpoint, copy the file and
change the grader's `model`, `args` or `base_url`. Model-under-test provider args
such as `-M max_model_len=8192` stay on the command line because the file does
not name that model. Benchmarks that grade only finished answers need a model
under test that reports stop reasons, so these examples serve it with `vllm/`
(see [Stop reasons](#stop-reasons)). `inspect_template/parity.yaml` shows the format with
illustrative values.

## Credentials

Inspect records every task argument (`-T`, `task_args`) in the `.eval` log, and
logs are shared. No task argument carries a credential. Each role reads
environment variables instead, and `llm_behavior_eval.common.credentials` holds
the shared helpers.

| Role | Variables, in order of precedence | Read by |
| --- | --- | --- |
| Dataset (Hub) | `LLM_BEHAVIOR_EVAL_DATASET_TOKEN`, then `HF_TOKEN`; with neither, `huggingface_hub`'s own sources (such as a cached `hf auth login`) | `load_hf_dataset`, through `credentials.resolve_dataset_token()`, when the task is built |
| Model under test | Its provider's own: `HF_TOKEN` (`hf/`, and Hub downloads by the vLLM server Inspect starts), `<SERVICE>_API_KEY` and `<SERVICE>_BASE_URL` (`openai-api/<service>/...`), `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `VLLM_API_KEY` (vLLM server key), ... | The Inspect provider |
| Grader | The same, for the grader's provider | The Inspect provider |

Empty or whitespace-only values count as unset. `save_to_disk` snapshots
(`-T dataset_path=...`) take no token. This package adds no variables for
models: Inspect's providers already read their own.

### Separate model and grader credentials

* **Different variables.** Providers and services read different variables, so
  give each role its own service name:

  ```sh
  export LLM_BEHAVIOR_EVAL_DATASET_TOKEN=...  # dataset only
  export ANSWER_BASE_URL=http://answer.internal:8000/v1 ANSWER_API_KEY=...
  export AIRGAP_BASE_URL=http://judge.internal:8000/v1 AIRGAP_API_KEY=...
  inspect eval llm_behavior_eval/<benchmark> --model openai-api/answer/model \
    --model-role 'grader={model: openai-api/airgap/judge}'
  ```

* **Same variable** (for example two `hf/` models that need different Hub
  tokens, or two `openai/` models on different accounts). One environment
  cannot hold both. Use Inspect's per-model `api_key`, which Inspect removes
  from the model arguments it logs: `get_model(..., api_key=...)` in Python,
  `-M api_key=...` for the model under test and `model_args: {api_key: ...}` in
  the grader role on the CLI. For `hf/`, `api_key` is the Hub token for that
  model's weights and tokenizer; without it, `huggingface_hub` reads `HF_TOKEN`.
  CLI arguments are visible in the process list and shell history, so prefer
  Python, for example `api_key=os.environ["JUDGE_HF_TOKEN"]`.
* **Two `vllm/` models with different Hub tokens.** `vllm/`'s `api_key` is the
  server key, not a Hub token, and the server Inspect launches inherits
  `HF_TOKEN` when it starts. This cannot be separated in one process: download
  the weights with each token first (`hf download org/repo --token ...
  --local-dir ...`) and pass local paths, start the servers yourself and pass
  `-M base_url=...`, or run separate processes.

### Python: the `credentials` context manager

```python
from inspect_ai import eval
from inspect_ai.model import get_model

from llm_behavior_eval.common.credentials import credentials

with credentials(
    dataset_token=dataset_secret,
    model_env={"HF_TOKEN": model_secret},
    judge_env={"AIRGAP_API_KEY": judge_secret},
):
    eval(
        "llm_behavior_eval/<benchmark>",
        model=get_model("vllm/org/answer", memoize=False),
        model_roles={"grader": get_model("openai-api/airgap/judge", memoize=False)},
    )
```

`credentials(*, dataset_token=None, model_env=None, judge_env=None)` takes
credentials as ordinary Python arguments, never as task or logged model
arguments. `dataset_token` sets `LLM_BEHAVIOR_EVAL_DATASET_TOKEN`; `model_env`
and `judge_env` map each role's provider variables to values. `None` leaves a
variable unchanged.

* **Restore.** On exit, normal or by exception, every variable the context set
  gets its previous value back, or is removed if it was unset. Changes made
  inside the block to those variables are discarded; other variables are not
  touched. Contexts nest, and the innermost value wins while it is open.
* **Validation.** Before changing anything, it raises `ValueError` for an empty
  value, an invalid name, or two roles that set one variable to different
  values (use `api_key` as above). Messages name variables, never values.
* **Scope.** Keep the block open around the whole `eval` call: the dataset
  token is read when the task is built, provider keys when models are built,
  and a `vllm/` server starts lazily during the run. A server Inspect started
  keeps the environment it started with.
* **Environment variables are process-global.** Every thread and async task in
  the process sees them. Do not overlap contexts that set different
  credentials (threads, concurrent `eval_async` calls). Run such evaluations in
  separate processes, for example one `inspect eval` per credential set.
* **Inspect caches models given by name.** `eval(model="openai-api/answer/m")`
  reuses the model built by an earlier `eval` in the same process, with the key
  it read then. Build models inside the block with `get_model(...,
  memoize=False)`, as above, or use a separate process.

### Mapping from `main`

On `main`, PR #186 added `--dataset-token` (`dataset_token` in the
evaluation config, excluded from saved configs), independent of
`--model-token` and `--judge-token`. Without it, datasets used the loader's
default authentication (`HF_TOKEN` or a cached login) and never the model
token. `main` defined no environment variable for it, so the native layer adds
`LLM_BEHAVIOR_EVAL_DATASET_TOKEN`:

| `main` | Native Inspect |
| --- | --- |
| `--dataset-token X` | `LLM_BEHAVIOR_EVAL_DATASET_TOKEN=X`, or `credentials(dataset_token=X)` |
| No `--dataset-token` | `HF_TOKEN`, then a cached login: the same loader default |
| `--model-token X` | `hf/`: `HF_TOKEN=X`, or `-M api_key=X` / `get_model(..., api_key=X)`. `vllm/`: `HF_TOKEN` when the server starts. Other providers: their own variable. |
| `--judge-token X` | `hf/` grader: `model_args: {api_key: X}` / `get_model(..., api_key=X)`. Other providers: their own variable, under a service name distinct from the model under test's. |

One difference: `main` applied `--model-token` only while loading the model,
so datasets never saw it. Natively, a model token set as `HF_TOKEN` is also the
dataset fallback. To keep it from dataset access, set
`LLM_BEHAVIOR_EVAL_DATASET_TOKEN`, or give the model its token through
`api_key` instead of `HF_TOKEN`.

### Tests

`tests/test_inspect_packages.py` fails if any registered task has a parameter
whose name contains the word `token`, `key`, `secret` or `password` (the
thinking-trace markers are not credentials), and each benchmark's tests assert
that a dummy `HF_TOKEN` never appears in its `.eval` log.
`tests/test_inspect_credentials.py` covers dataset token precedence, nested
contexts, restoration on exceptions (including variables that were unset),
dataset, model and grader separation, per-model `api_key` for `hf/`, model
memoization, and an eval with dataset, model, grader and `HF_TOKEN` sentinels,
none of which appear in its `.eval` log.

## Caveats

* **Inspect's HF provider samples by default.** It passes `do_sample=True` to
  `generate()` unless you pass `-M do_sample=false` (or `do_sample: false` in a
  grader's model args). Greedy settings such as `temperature: 0` need it.
* **Inspect's HF provider caps output at 2048 tokens** when `max_tokens` is unset,
  instead of the model's own `generation_config.json` limit. vLLM, SGLang and
  `openai-api` send no limit when unset; some hosted providers (for example
  Anthropic, Together, Groq) substitute their own Inspect default.
* **Inspect's HF provider reports no stop reason** (nor do `nnterp/` and
  `transformer_lens/`). See [Stop reasons](#stop-reasons): `only_stop` scorers
  fail on them, and
  `incomplete_response_rate` is NaN.
* **Grading is non-deterministic unless you override it.** Model-recommended
  generation defaults for judges are often sampling (for example
  `generation_config.json` with `do_sample: true`). Set the grader's temperature
  and, on HF, `do_sample: false`.
* **The `.eval` log records only explicitly passed settings.** Defaults from
  `generation_config.json`, a vLLM server or a hosted API are not in the log.
  Pin the model revision (for example a local snapshot path or an immutable
  served model) and record it with the run.
* **Recommended judge:** `google/gemma-4-12B-it`, matching the legacy
  `--judge-model` default. It is not parity-qualified. Parity evidence used Gemma 3
  12B judges with explicit parity configs.

## Legacy flag mapping

The mapping below was checked against installed Inspect 0.3.287 provider source.
Local GPU execution and full benchmark parity are not claimed by CPU tests. `-M`
applies to the model under test; put the same keys under the grader's
`model_args` (inline) or `args` (run config) for the judge.

| Legacy setting | Native Inspect equivalent / limitation |
| --- | --- |
| HF repo | `--model hf/org/repo` |
| Local HF | `--model hf/local -M model_path=/models/model -M tokenizer_path=/models/model` |
| vLLM repo | `--model vllm/org/repo` (Inspect launches a vLLM server) |
| Local vLLM | `--model vllm//models/model`; `vllm/local` alone means a model literally named `local`, not a model_path override. For a prestarted endpoint use `vllm/local -M base_url=http://127.0.0.1:8000/v1`, with that served model name. |
| `--judge-model` | `--model-role grader=...` with any provider. The grader is never implied, so always pass it. |
| Answer/judge temperature, token budgets, thinking | No defaults. Pass them per role as above, or use the benchmark's `parity.yaml`. |
| `--exclude-thinking-trace-for-judge` / `--include-thinking-trace-for-judge`, `--thinking-start-token`, `--thinking-end-token` | `-T exclude_thinking_trace_for_judge=true/false`, `-T thinking_start_token=...`, `-T thinking_end_token=...`. Stripping defaults to **on** with `<think>`/`</think>`; legacy defaulted to off. |
| `--dataset-path` | Benchmark task parameter `-T dataset_path=...`. |
| `--dataset-token` | No task parameter. Set `LLM_BEHAVIOR_EVAL_DATASET_TOKEN` (falls back to `HF_TOKEN`), or `credentials(dataset_token=...)` in Python; see [Credentials](#credentials). |
| `--model-token`, `--judge-token` | No task parameter. Each role's provider variable, or Inspect's per-model `api_key`; see [Mapping from `main`](#mapping-from-main). |
| max_samples | `--limit N` / `eval(limit=N)`, or `eval_config.limit` in a parity config. Without a limit, Inspect runs the full split. |
| use-4bit / use-4bit-judge | HF forwards remaining model args to Transformers. Use Python `quantization_config=BitsAndBytesConfig(load_in_4bit=True)`; no portable boolean CLI alias is implemented. Requires compatible bitsandbytes/GPU. Not vLLM. |
| device_map | HF `-M device=auto` (Inspect supplies this as device_map); don't pass device_map again |
| trust_remote_code | `-M trust_remote_code=true`; opt in explicitly |
| batch_size / judge_batch_size | HF `batch_size` model arg independently on each role; vLLM uses server scheduling and Inspect max_connections, not legacy batches |
| `--vllm-max-model-len N` | `-M max_model_len=N` (vLLM server `--max-model-len N`) |
| `--vllm-gpu-memory-utilization F` | `-M gpu_memory_utilization=F` (`--gpu-memory-utilization F`) |
| `--vllm-tokenizer-mode MODE` | `-M tokenizer_mode=MODE` (`--tokenizer-mode MODE`; `auto`, `slow`, `mistral` or `custom`) |
| `--vllm-config-format FMT` / `--vllm-load-format FMT` | `-M config_format=FMT` / `-M load_format=FMT` (`--config-format` / `--load-format`) |
| vLLM tokenizer path, language_model_only | `-M tokenizer=PATH`; `-M language_model_only=true` (boolean flag, omitted when false) |
| vLLM server-argument contract | Inspect forwards every `-M name=value` it does not recognize to the server it launches, as `--name-with-dashes value` (`true` becomes a bare flag). Inspect does not validate values; support depends on the vLLM version and model. With `base_url` no server is launched, so configure these on the prestarted server. Gemma 4 still needs GPU validation. |
| vllm-max-lora-rank | vLLM `-M max_lora_rank=...`; native adapter model syntax `vllm/base:adapter-path` |
| mlflow://, s3://, gs://, git:// adapters | Pre-resolve using legacy `util_functions.maybe_download_adapter` in a separate setup step, then pass the local adapter path. No URI resolver or custom provider in this foundation. HF adapter parity is unverified. |
| Muse Glimmer / custom HF chat-template thinking | Inspect's HF provider accepts only `enable_thinking` and does not implement legacy `SafeApplyChatTemplate`. Remains a parity blocker pending a provider adaptation. |

## Validation and remaining work

CPU tests cover registration and CLI execution, lightweight discovery, parsing,
finish filtering, the `hf/` stop-reason error, inline thinking and its task
arguments, metrics over every epoch, split selection, offline snapshots,
per-role settings via Python, CLI flags, `--generate-config` and `--run-config`,
the absence of code-set settings for both roles, an `openai-api` judge on
loopback with and without a role config, parity-config loading, and HF/vLLM
provider configuration with stub loaders. They are not GPU inference tests.

The sequential vLLM eval lifecycle still needs a regression test for repeated
real GPU evaluations; it is tracked separately. Benchmark parity, GPU/quantization/LoRA testing,
Muse Glimmer support and optional Inspect-brief integration remain follow-up
work. Keep the legacy engine until benchmark-by-benchmark comparisons pass.
