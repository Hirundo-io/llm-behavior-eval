# Native Inspect migration (LLM-155)

Every benchmark runs as a native Inspect task; see the root README for the task
list and reporting. Since 0.2.0 (LLM-171) the package is native-only: the legacy
evaluation engine, its model loading and built-in inference defaults, and its
Python API (`EvaluateFactory`, `EvaluationConfig`, ...) are removed, and the
legacy CLI only prints a migration hint and exits 2 (see `CHANGELOG.md`).
Historical legacy runs need the frozen pre-removal checkout documented in
`scripts/parity/README.md`. Inspect 0.3.288 is tested and pinned to `<0.3.289`. Every
Inspect behavior described below was checked against that version's source and
is covered by tests in `tests/test_inspect_foundation.py`,
`tests/test_inspect_packages.py`, `tests/test_inspect_template.py` and, for
reporting, `tests/test_inspect_smoke.py`, `tests/test_inspect_brief.py` and
`tests/test_inspect_mlflow.py`.
Revalidate before widening the pin. The package requires Python 3.12 or 3.13
(inspect-brief, the `[brief]` extra, requires 3.12+).

```sh
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -e .
source .venv/bin/activate
inspect list tasks llm_behavior_eval
inspect eval llm_behavior_eval/inspect_template --model mockllm/model
```

The template is a wiring demonstration: its **default grader is a mock that always
returns SCORE: 1**. Real benchmarks bind no grader and fail unless one is passed.

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

## Benchmark layout

Each benchmark is a regular subpackage `llm_behavior_eval/<benchmark>/` (it has an
`__init__.py`) that owns:

| File | Contents |
| --- | --- |
| `task.py` | The `@task` function. No `config=`, no default grader. |
| `dataset.py` | Loading, full-source validation, `record_to_sample`. |
| `prompts.py` | System and judge prompts, copied verbatim from the legacy engine. |
| `solver.py` | Solver chain. No GenerateConfig. |
| `scorer.py` | Scorer, usually wrapping `common.judge.judge_scorer`. |
| `metrics.py` | Benchmark metrics, when `common.metrics` is not enough. |
| `README.md` | What it measures, task arguments, metrics, parity status. |
| `parity.yaml` | Legacy-engine settings as an Inspect run config (see below). |
| `target_metrics.json` | Report metrics for each of its tasks (see below). |

Its tests live outside the package, in `tests/test_inspect_<benchmark>.py`:
registry discovery, dataset conversion, scorer, parity-config tests, and a check
that inspect-brief finds every `target_metrics.json` name in a real `.eval` log
(`missing_target_metrics`, see [Target metrics](#target-metrics)).

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
has target metrics and vice versa, package files are tracked and ship in the
wheel, and an installed wheel discovers the same tasks and metrics.

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

* `dataset.load_hf_dataset(dataset_id, *, dataset_path=None, dataset_token=None)`
  loads the complete split of an HF dataset or a `save_to_disk` snapshot.
  `dataset_path` changes where data is read from, not the logical id. Selects
  `train` when present, otherwise requires exactly one split. The token is passed
  only to HF Hub loading.
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
  `stop`. Unmatched, unparsable and non-finite (`nan`, `inf`) grades become
  `parse_error` scores. Skips and parse errors have placeholder value 0 and are
  excluded by `field_rate`; never average them with a plain mean. Score metadata
  holds the sample metadata, `stop_reason` and `raw_judge_text`.
* `judge.strip_thinking(answer, ...)` keeps only the text after the last end
  marker when both markers are configured and either appears. An unclosed start
  marker keeps the whole answer. Structured `ContentReasoning` is already
  excluded from `ModelOutput.completion`.
* `metrics.field_rate(field=None, *, metadata_key=None, metadata_value=None)`
  averages scalar scores, or one field of dict scores, over eligible samples,
  optionally restricted to a score-metadata subgroup.
  `metrics.incomplete_response_rate()` is the share of **all** scored samples that
  stopped at `max_tokens`. Empty denominators yield NaN.
* `packages.benchmark_packages(root="llm_behavior_eval")` returns the sorted
  names of subpackages that have a `task` module, without importing them.
  `packages.subpackages(root)` lists every regular subpackage.
* `target_metrics.load_target_metrics(root="llm_behavior_eval")` merges every
  package's `target_metrics.json` into one mapping with sorted task keys.
  Each file is validated (keys `llm_behavior_eval/<task>`, non-empty metric
  lists, objects with exactly `name`, `is_percentage`, `is_higher_better`,
  `is_normalized`, booleans as JSON booleans). A task key defined twice, in one
  file or across packages, raises `ValueError`. `target_metrics_files(root)`
  maps packages to their files; `read_target_metrics_file(path)` validates one.
  `write_target_metrics(path)` writes the merged JSON for tools that need one
  file, such as inspect-brief; from a shell,
  `python -m llm_behavior_eval.common.target_metrics --output PATH` does the
  same, and without `--output` it prints the JSON.
* `parity.parity_config_path(package)` returns the installed path of a benchmark's
  `parity.yaml`. `parity.read_parity_config(path)` returns `inspect_ai.eval()`
  keyword arguments, parsed by Inspect's own `--run-config` reader. It rejects
  files that set `task`, `model` or `solver`.
* `preprocessing` is the **opt-in** legacy text preprocessing for parity runs
  (LLM-169). `legacy_text_preprocessing(tokenizer, revision=None, *,
  legacy_thinking, target=False, metadata_fields=())` is a solver step a
  benchmark places just before `generate()`. It renders the current messages
  with the answer model's tokenizer as the legacy engine did, and records under
  `metadata["legacy_text"]` the decoded 1024-token prompt and its token IDs,
  plus 256-token round trips of the target and of the named metadata fields.
  `legacy_prompt(state, default)` and `legacy_text(state, name, default)` return
  those views, or `default` when the step did not run. It changes no message,
  sets no GenerateConfig and inspects no provider, so every provider works.
  `load_tokenizer`, `render_chat` and `roundtrip` are the tokenizer mechanics.
  Templates the legacy engine rewrote (Nemotron `/no_think`, Muse Glimmer, the
  Gemma v1 system merge) raise instead of rendering a wrong prompt.

Sample limits are Inspect's: `--limit N`, `eval(limit=N)`, or `eval_config.limit`
in a run config. There is no task-level sample cap.

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

## Reporting with inspect-brief

inspect-brief reads `.eval` logs and appends one CSV row per task and target
metric, through its Inspect hook (`INSPECT_BRIEF_CSV_PATH`,
`INSPECT_BRIEF_TARGET_METRICS`, optional `INSPECT_BRIEF_TASKS` and
`INSPECT_BRIEF_SKIP_EXISTING`) or its CLI (`inspect-brief --log-dir ...
--target-metrics ... --csv-path ...`). There is no root `target_metrics.json`:
give inspect-brief the merged per-package mapping as a **file**, written by
`python -m llm_behavior_eval.common.target_metrics --output targets.json` or
`write_target_metrics(path)`. inspect-brief 0.1.1 also accepts inline JSON, but
it first checks whether the value is a file, and on Python 3.12/3.13 that check
raises `ENAMETOOLONG` for a value as long as the merged mapping.

Benchmark-specific report choices live in each package's `target_metrics.json`;
`common/` only merges and validates. In particular the bias tasks list both
`error` and `accuracy`: inspect-brief selects targets by `log.eval.task`, not by
task arguments, so a report cannot pick the legacy primary metric per `kind`.
The previous layer's `target_metrics(bias_kind=...)` filter was dropped rather
than moved into `common/`.

Known gaps a reader sees in the CSV:

* BBQ, BLOOM and UNQOVER emit `stereotype_bias` only when the split has a
  `stereotyped_answer` column (legacy behavior). Otherwise inspect-brief writes
  `Metric 'stereotype_bias' not found` in that row's `Score`.
* Logs whose task is not `llm_behavior_eval/<task>` get no targets; see
  `python -m inspect_ai` under [Target metrics](#target-metrics). Use the
  `inspect` command.
* `ccpc` declares only `censorship_rate`; it has no `incomplete_response_rate`
  target.

`tests/test_inspect_brief.py` runs every registered task on mock models and
checks that the hook and the CLI write the same rows, with the units and
direction from the targets.

## Presets and the retired CLI

Presets select tasks and task arguments only; they carry no inference settings.
`DatasetPreset(name, dataset_ids)`, `list_dataset_presets()` and
`expand_dataset_preset()` keep their contract (case- and space-insensitive
names; `hallucination` and `hallucination-med` stay aliases).
`DatasetPreset.inspect_tasks` is a tuple of `InspectTaskMapping(task_name,
task_args, dataset_id)` aligned with `dataset_ids`; `dict(task_args)` gives task
keyword arguments and `cli_args` the `-T` options. `dataset_id` is the asset's
logical identity (CCPC's is `chinese_censorship`), not a task argument.
`presets.py` imports nothing heavy.

The root README has the full preset table. `llm-behavior-eval`,
`python -m llm_behavior_eval` and `python -m llm_behavior_eval.evaluate` print
the migration hint and exit 2 without loading models. When an argument names a
preset, the hint lists its `inspect eval` commands with placeholder models; no
other argument is echoed, because legacy invocations may carry credentials. No
legacy flag is translated: the old answer/judge token budgets and thinking flags
correspond to per-role Inspect settings or to a benchmark's `parity.yaml`.

## MLflow reporting hook

`llm_behavior_eval/hooks/mlflow.py` is an Inspect hook (`on_run_start`,
`on_run_end`). `hooks/` has no `task` module, so discovery skips it;
`_registry.py` imports it explicitly after discovery and before the
registry-name check. Inspect imports every `inspect_ai` entry point when an eval
starts, so the hook is present for any task once the package is installed. Its
`enabled()` is false unless `LLM_BEHAVIOR_EVAL_MLFLOW_ENABLED` is set, and it
imports MLflow only then. It reads no inference settings. See
`MLFLOW_INTEGRATION.md`.

## Setting inference options per role

Inspect keeps the two roles apart. Eval-level GenerateConfig (CLI flags,
`--generate-config`, `eval(**kwargs)`) applies **only to the model under test**.
The grader gets only its role's GenerateConfig, plus operational settings
(`max_connections`, `max_retries`, `timeout`, `cache`). In 0.3.288 this is
`Model._resolve_config`. `-M`/`model_args` also reach only the model under test.

### Model under test

| Form | Example |
| --- | --- |
| CLI flags | `--temperature 0 --max-tokens 128` (also `--top-p`, `--top-k`, `--seed`, `--reasoning-effort`, `--reasoning-tokens`, `--extra-body`, ...) |
| CLI file | `--generate-config answer.yaml`, a YAML/JSON mapping of GenerateConfig fields, for example `temperature: 0.0` and `max_tokens: 128`. Unknown keys are rejected. Flags override file values. |
| Provider args | `-M name=value`, or `--model-config args.yaml` |
| Python | `eval(task, model="hf/org/repo", model_args={"do_sample": False}, temperature=0.0, max_tokens=128)` |

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
  --model hf/org/answer -M do_sample=false --temperature 0 --max-tokens 128 \
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
    model="hf/local",
    model_args={"model_path": "/models/answer", "do_sample": False},
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

Inline `<think>...</think>` traces in answers are stripped before judging by
`judge_scorer`, whatever the thinking setting.

## Parity configs

A benchmark's `parity.yaml` is an ordinary Inspect run config holding only
inference settings: `generate_config` (model under test), `model_roles.grader`
(judge model, `config`, `args`) and optionally `eval_config` (for example
`limit: 500`, the legacy sample cap). It never sets `task`, `model` or `solver`,
so the model under test is still chosen on the command line.

A parity run combines three things, and only parity runs use them:

1. **`--run-config <benchmark>/parity.yaml`**: the legacy engine's greedy
   decoding, token budgets, seed 42 and thinking settings for both roles, and,
   for vLLM/SGLang/OpenAI-compatible servers, the legacy prompt tokenization:
   special tokens added to the rendered template (`add_special_tokens: true`)
   and answer prompts truncated to 1024 tokens on the right
   (`truncate_prompt_tokens`, `truncation_side`). Inspect's HF provider ignores
   `extra_body`.
2. **`-M` provider args for the model under test**, which the file cannot carry
   because it does not name that model. On HF: `-M do_sample=false`,
   `-M enable_thinking=<legacy value>` and
   `-M tokenizer_call_args='{truncation: true, max_length: 1024}'`.
3. **`-T legacy_tokenizer=<answer snapshot>`**, the benchmark's opt-in legacy
   text preprocessing (shared mechanics in `common/preprocessing.py`). The
   legacy judge did not see raw sample text: it saw the decoded, truncated chat
   template as the question, and targets and stereotypes after a 256-token
   round trip. With this task argument each benchmark shows its judge those
   views; without it (the default) the task is native Inspect. Pass
   `-T legacy_tokenizer_revision=<40-hex commit>` when the tokenizer is a cached
   repository rather than a local directory. Inspect's run config can only set
   task arguments inside a `task:` block that also names the task, which
   `parity.yaml` must not do, so the flag stays on the command line.

```sh
inspect eval llm_behavior_eval/<benchmark> \
  -T legacy_tokenizer="$ANSWER" \
  --model hf/local -M model_path="$ANSWER" \
  -M do_sample=false -M enable_thinking=false \
  -M tokenizer_call_args='{truncation: true, max_length: 1024}' \
  --run-config llm_behavior_eval/<benchmark>/parity.yaml
```

The header of each `parity.yaml` and each benchmark README give that
benchmark's exact command (XSTest and OR-Bench use `enable_thinking=true`; BBQ,
UNQOVER and BLOOM add their split arguments). Do not add `--model-role grader=`:
a command-line role replaces the file's whole grader entry, its `config`
included. To serve the judge another way (the qualification served it with
vLLM), copy the file and change only the grader's `model` and `args`.
`scripts/parity/README.md` has the vLLM variant, the runners and the evidence
status per benchmark.

From an installed package, find the file with
`python -c 'from llm_behavior_eval.common.parity import parity_config_path; print(parity_config_path("llm_behavior_eval.<benchmark>"))'`.

```python
from inspect_ai import eval
from llm_behavior_eval.common.parity import parity_config_path, read_parity_config

eval(
    "llm_behavior_eval/<benchmark>",
    model="hf/org/answer",
    model_args={"do_sample": False},
    **read_parity_config(parity_config_path("llm_behavior_eval.<benchmark>")),
)
```

Like the CLI, `read_parity_config` constructs the grader model, which may load its
weights. To keep parity settings with another judge endpoint, copy the file and
change the grader's `model`, `args` or `base_url`. Model-under-test provider args
such as `-M do_sample=false` stay on the command line because the file does not
name that model. `inspect_template/parity.yaml` shows the format with
illustrative values.

## Caveats

* **Inspect's HF provider samples by default.** It passes `do_sample=True` to
  `generate()` unless you pass `-M do_sample=false` (or `do_sample: false` in a
  grader's model args). Greedy settings such as `temperature: 0` need it.
* **Inspect's HF provider caps output at 2048 tokens** when `max_tokens` is unset,
  instead of the model's own `generation_config.json` limit. vLLM, SGLang and
  `openai-api` send no limit when unset; some hosted providers (for example
  Anthropic, Together, Groq) substitute their own Inspect default.
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

The mapping below was checked against installed Inspect 0.3.288 provider source.
Local GPU execution and full benchmark parity are not claimed by CPU tests. `-M`
applies to the model under test; put the same keys under the grader's
`model_args` (inline) or `args` (run config) for the judge.

| Legacy setting | Native Inspect equivalent / limitation |
| --- | --- |
| HF repo | `--model hf/org/repo` (install the `hf` extra: Transformers, Torch, Accelerate) |
| Local HF | `--model hf/local -M model_path=/models/model -M tokenizer_path=/models/model` |
| vLLM repo | `--model vllm/org/repo` (Inspect launches a vLLM server; install the `vllm` extra on Linux) |
| Local vLLM | `--model vllm//models/model`; `vllm/local` alone means a model literally named `local`, not a model_path override. For a prestarted endpoint use `vllm/local -M base_url=http://127.0.0.1:8000/v1`, with that served model name. |
| `--judge-model` | `--model-role grader=...` with any provider. The grader is never implied, so always pass it. |
| Answer/judge temperature, token budgets, thinking | No defaults. Pass them per role as above, or use the benchmark's `parity.yaml`. |
| dataset path/token | Benchmark task parameters `-T dataset_path=... -T dataset_token=...`. HF_TOKEN keeps credentials out of CLI history and logged task arguments. |
| max_samples | `--limit N` / `eval(limit=N)`, or `eval_config.limit` in a parity config. Without a limit, Inspect runs the full split. |
| use-4bit / use-4bit-judge | HF forwards remaining model args to Transformers. Use Python `quantization_config=BitsAndBytesConfig(load_in_4bit=True)`; no portable boolean CLI alias is implemented. Requires compatible bitsandbytes/GPU, which 0.2.0 no longer installs. Not vLLM. |
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
| `--use-mlflow`, `--mlflow-*` | `LLM_BEHAVIOR_EVAL_MLFLOW_ENABLED=1` and the other environment settings in `MLFLOW_INTEGRATION.md`. Set `LLM_BEHAVIOR_EVAL_MLFLOW_ADAPTER_REF` to the original adapter reference so the hook attaches to its run and step. |
| `--base-output-dir`, `--model-output-dir` | `--log-dir` |
| Muse Glimmer / custom HF chat-template thinking | Inspect's HF provider accepts only `enable_thinking` and does not implement legacy `SafeApplyChatTemplate`. Remains a parity blocker pending a provider adaptation. |

## Validation and remaining work

CPU tests cover registration and CLI execution, lightweight discovery, parsing,
finish filtering, inline thinking, metrics, split selection, offline snapshots,
per-role settings via Python, CLI flags, `--generate-config` and `--run-config`,
the absence of code-set settings for both roles, an `openai-api` judge on
loopback with and without a role config, parity-config loading, and HF/vLLM
provider configuration with stub loaders. Reporting tests run every registered
task offline on mock models and check inspect-brief's hook and CLI, the preset
mapping against each task's signature and dataset identity, and the MLflow hook
against local file stores. They are not GPU inference tests.

Parity tooling (LLM-169) lives in `scripts/parity/`. The representative GPU
qualification (31 configurations x 100 samples, exact) ran the pre-redesign
implementation, which set inference settings in code. A CPU replay of its
recorded requests through the current tasks, configured only by `parity.yaml`
and the opt-in preprocessing, reproduces every request, setting, score and
metric exactly; a GPU re-run with `parity.yaml` is still required to confirm
it (`scripts/parity/README.md`). The judge for parity is Gemma 3 12B (CCPC:
its pinned Gemma 4 26B-A4B); `google/gemma-4-12B-it` is not qualified.

Tests that compared native code with the legacy engine now compare it with the
legacy outputs frozen in `tests/golden/` (captured at the pre-removal commit; see
`tests/golden/README.md`), so they keep their assertions without the engine.

Track LLM-141 for sequential vLLM eval lifecycle: repeated real GPU evaluations
still need a regression test. A GPU parity re-run with `parity.yaml`,
GPU/quantization/LoRA testing and Muse Glimmer support remain follow-up work.
