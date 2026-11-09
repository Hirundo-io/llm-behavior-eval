# [llm-behavior-eval](https://github.com/Hirundo-io/llm-behavior-eval) &middot; [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/hirundo-io/llm-behavior-eval/blob/main/LICENSE) [![pypi version](https://img.shields.io/pypi/v/llm-behavior-eval)](https://pypi.org/project/llm-behavior-eval/)

[![pyrefly](https://github.com/hirundo-io/llm-behavior-eval/actions/workflows/pyrefly.yaml/badge.svg)](https://github.com/hirundo-io/llm-behavior-eval/actions/workflows/pyrefly.yaml) [![ruff](https://github.com/hirundo-io/llm-behavior-eval/actions/workflows/ruff.yaml/badge.svg)](https://github.com/hirundo-io/llm-behavior-eval/actions/workflows/ruff.yaml) [![Unit tests](https://github.com/hirundo-io/llm-behavior-eval/actions/workflows/tests.yaml/badge.svg)](https://github.com/hirundo-io/llm-behavior-eval/actions/workflows/tests.yaml) [![Vulnerability scan](https://github.com/hirundo-io/llm-behavior-eval/actions/workflows/vulnerability-scan.yaml/badge.svg)](https://github.com/hirundo-io/llm-behavior-eval/actions/workflows/vulnerability-scan.yaml)

Measure undesirable LLM behaviors (bias, hallucination, prompt injection,
over-refusal and Chinese political censorship) with native
[Inspect](https://inspect.aisi.org.uk/) tasks. The model under test and the judge
(Inspect's `grader` role) are configured independently, on any Inspect provider:
`hf/`, `vllm/`, `sglang/`, hosted APIs, and OpenAI-compatible `openai-api/`
endpoints. Results are Inspect `.eval` logs; [inspect-brief](https://github.com/Hirundo-io/inspect-brief)
summarizes them into a CSV, and an optional hook reports them to MLflow.

## What it evaluates

Task names have the prefix `llm_behavior_eval/`. Each benchmark is a
self-contained package with its own README (`llm_behavior_eval/<task>/README.md`)
listing its task arguments, metrics, judge prompt and parity status.
`inspect list tasks llm_behavior_eval` lists what is installed.

| Task | Hugging Face dataset | `-T` arguments | Report metrics |
| --- | --- | --- | --- |
| `bbq` | `hirundo-io/bbq-<type>-<bias\|unbias>-free-text` | `bias_type` (gender, race, nationality, physical, age, religion), `kind` (bias, unbias) | error, accuracy, stereotype_bias |
| `unqover` | `hirundo-io/unqover-<type>-bias-free-text` | `bias_type` (religion, gender, race, nationality); bias split only | error, accuracy, stereotype_bias |
| `bloom` | `hirundo-io/bloom-<type>-<bias\|unbias\|ambiguous-bias>-free-text` | `bias_type` (age, gender, race), `kind` (bias, unbias, ambiguous) | error, accuracy, stereotype_bias |
| `halueval` | `hirundo-io/halueval` | none | accuracy |
| `medhallu` | `hirundo-io/medhallu` | none | accuracy, empty_responses |
| `prompt_injection` | `hirundo-io/prompt-injection-purple-llama` | none | attack_success_rate |
| `ccpc` | `hirundo-io/ccpc-500` (pinned revision, 500 rows) | none | censorship_rate |
| `xstest` | `hirundo-io/XSTest` | none | safe_refusal_rate, unsafe_refusal_rate |
| `orbench` | `hirundo-io/or-bench` | none | safe_refusal_rate, unsafe_refusal_rate, judge_unparseable_rate |

Every task except `ccpc` also reports `incomplete_response_rate`, the share of
answers that stopped at the token limit. Every task takes `-T dataset_path=...`
(a local copy saved with `save_to_disk`, for offline runs) and
`-T dataset_token=...` (prefer `HF_TOKEN`, which keeps the token out of logged
task arguments). `llm_behavior_eval/inspect_template` is a wiring demonstration
with a mock grader, not a benchmark.

## Installation

Inspect needs Python 3.11 or newer; the `[brief]` reporting extra needs 3.12 or
newer. Use Python 3.12 or 3.13.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install 'llm-behavior-eval[brief]'        # or -e '.[brief]' from a checkout
```

Extras: `brief` (inspect-brief CSV summaries), `mlflow` (MLflow reporting hook),
`vllm` (Inspect's vLLM provider on Linux). For development, `uv sync --locked
--extra brief --extra mlflow` installs the pinned environment CI uses; it still
includes the legacy engine's GPU dependencies, which the parity tooling compares
against until the legacy engine is removed.

### vLLM and Gemma 4

The `vllm` extra selects `vllm>=0.30.0,<0.31` on Linux x86_64/aarch64 with
Transformers `>=5.20.2,<5.21`. vLLM 0.30 resolves Gemma 4 attention parameters
from each layer config; older vLLM with Transformers 5.20.2 raises
`AmbiguousGlobalPerLayerAttributeError` instead. Do not disable that guard or
downgrade Transformers to work around it. Install the locked stack together:
vLLM 0.30 requires Torch 2.13.0 (CUDA 13.0.3 and Triton 3.7.1 wheels), NVIDIA
driver 580 or newer, and, for FlashInfer JIT compilation, a CUDA toolkit with
`nvcc`, C++ build tools and Python development headers matching the interpreter
(for example `python3.12-dev`). vLLM 0.30 does not support bitsandbytes
quantization.

```bash
uv venv --python /usr/bin/python3.12 .venv-gemma4
UV_PROJECT_ENVIRONMENT=.venv-gemma4 uv sync --locked --no-dev --extra vllm --extra brief
```

Inspect launches the vLLM server itself (`--model vllm/<repo-or-path>`) and
forwards unrecognized `-M name=value` options to it as server flags, for example
`-M max_model_len=8192 -M gpu_memory_utilization=0.85 -M language_model_only=true`.
See the [legacy flag mapping](INSPECT_MIGRATION.md#legacy-flag-mapping).

## Running a benchmark

```bash
inspect eval llm_behavior_eval/bbq -T bias_type=gender -T kind=bias \
  --model hf/meta-llama/Llama-3.1-8B-Instruct \
  --model-role grader=hf/google/gemma-4-12B-it \
  --log-dir logs
inspect view --log-dir logs
```

The grader is never implied: always pass `--model-role grader=...`. The
recommended judge is `google/gemma-4-12B-it` (the legacy `--judge-model`
default; not yet parity-qualified). CCPC pins `google/gemma-4-26B-A4B-it`; see
its README. `--limit N` caps the number of samples (the legacy `--max-samples`);
each benchmark orders its samples like the legacy engine did, so the same rows
are selected.

Run the `inspect` command, or run Python from a directory other than the
repository root. `python -m inspect_ai eval ...` started **from the repository
root** puts the source tree first on `sys.path`, where setuptools' egg-info hides
the install metadata Inspect uses to prefix task names. The log's task is then
`bbq` instead of `llm_behavior_eval/bbq`, and inspect-brief applies no target
metrics to it.

### Inference settings

Benchmarks set **no** inference settings for either role: no temperature,
top_p, top_k, token budgets, or thinking switches. Unset values fall through to
the serving stack's defaults (the model's `generation_config.json`, the vLLM or
SGLang server, the chat template, or the hosted API). Set them through Inspect,
separately for each role:

```bash
inspect eval llm_behavior_eval/halueval \
  --model hf/org/answer -M do_sample=false --temperature 0 --max-tokens 512 \
  --model-role 'grader={model: hf/google/gemma-4-12B-it, temperature: 0.0, max_tokens: 64, model_args: {do_sample: false}}'
```

Eval-level flags (`--temperature`, `--max-tokens`, `--top-p`, `--top-k`,
`--reasoning-effort`, `--reasoning-tokens`, `--generate-config FILE`) and `-M`
provider arguments apply **only to the model under test**. The grader uses only
its role's own settings, given inline as above or in a `--run-config` file.
Thinking is a per-role provider argument too: `-M enable_thinking=false` on HF,
or `--extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'` for
vLLM/SGLang.

Caveats: Inspect's HF provider samples unless you pass `-M do_sample=false`
(`do_sample: false` in the grader's `model_args`), and caps output at 2048 tokens
when `--max-tokens` is unset. Model-recommended judge defaults often sample, so
grading is non-deterministic unless you set the grader's temperature. The `.eval`
log records only settings you passed, so pin and record model revisions.

An OpenAI-compatible judge endpoint (for example an air-gapped server) is just
another provider:

```bash
export AIRGAP_BASE_URL=http://judge.internal:8000/v1
export AIRGAP_API_KEY=...   # or a placeholder for a no-auth server
inspect eval llm_behavior_eval/xstest --model hf/org/answer \
  --model-role 'grader={model: openai-api/airgap/judge, temperature: 0.0, max_tokens: 64}'
```

[INSPECT_MIGRATION.md](INSPECT_MIGRATION.md) has the full per-role reference
(CLI, run-config and Python forms), provider caveats and the legacy flag mapping.

### Reproducing legacy results

Each benchmark ships a `parity.yaml`, an Inspect run config with the legacy
engine's exact settings for both roles (greedy decoding, its token budgets and
thinking settings, the parity judge, and the legacy sample cap). It applies only
when you pass it:

```bash
inspect eval llm_behavior_eval/bbq -T bias_type=gender -T kind=bias \
  --model hf/org/answer -M do_sample=false \
  --run-config "$(python -c 'from llm_behavior_eval.common.parity import parity_config_path; print(parity_config_path("llm_behavior_eval.bbq"))')"
```

The parity judge is `google/gemma-3-12b-it` (CCPC: its pinned
`google/gemma-4-26B-A4B-it`); copy the file and change the grader's `model` to
keep the parity settings with another judge.

## Summaries with inspect-brief

`.eval` logs are the source of truth. inspect-brief appends one CSV row per task
and metric. Each benchmark package declares the metrics a report shows, and
their units and direction, in its `target_metrics.json`;
`llm_behavior_eval.common.target_metrics` merges them. Write the merged mapping
to a file and point inspect-brief at it:

```bash
python -m llm_behavior_eval.common.target_metrics --output targets.json
```

Pass the file path, not its contents: inspect-brief 0.1.1 first treats the value
as a path, and on Python 3.12/3.13 a value as long as the merged mapping makes
that check fail.

**During evaluation** (the hook, which inspect-brief registers with Inspect):

```bash
export INSPECT_BRIEF_TARGET_METRICS="$PWD/targets.json"
export INSPECT_BRIEF_CSV_PATH="$PWD/brief.csv"
export INSPECT_BRIEF_SKIP_EXISTING=true
# Optional: export INSPECT_BRIEF_TASKS=llm_behavior_eval/bbq,llm_behavior_eval/halueval
inspect eval llm_behavior_eval/halueval --model hf/org/answer \
  --model-role grader=hf/google/gemma-4-12B-it --log-dir logs
```

**After evaluation** (the CLI, over a log directory):

```bash
inspect-brief --log-dir logs --target-metrics targets.json \
  --csv-path brief.csv --skip-existing
```

Both append rows; `--skip-existing` (`INSPECT_BRIEF_SKIP_EXISTING=true`) skips
tasks already in the CSV. Use a fresh CSV when the target metrics change. A CSV
must have a single writer process: inspect-brief does not lock it across
processes. Only `.eval` logs are read, not JSON logs.

Things to know when reading the CSV:

* **Units and direction come from each package's `target_metrics.json`.**
  `is_normalized: true` metrics are fractions that inspect-brief multiplies by
  100 when `is_percentage` is set. HaluEval and Med-Hallu `accuracy` and Purple
  Llama `attack_success_rate` are already on a 0–100 scale. The metric column
  carries the unit and direction, for example `accuracy (%) ⬆️`.
* **BBQ, BLOOM and UNQOVER report `stereotype_bias` only on splits with a
  `stereotyped_answer` column**, as the legacy engine did. On other splits the
  CSV row reads `Metric 'stereotype_bias' not found`; that is expected, not a
  zero. `stereotype_bias` is a signed index (0 is neutral) without a percent
  sign.
* **Both `error` and `accuracy` are listed for the bias tasks.** The legacy
  summaries showed `error` for bias/ambiguous splits and `accuracy` for unbias
  splits; inspect-brief selects targets by task name, not task arguments, so it
  reports both. Read the one that matches the split (`kind` is in the log).
* **Targets apply only to logs named `llm_behavior_eval/<task>`.** A log run by
  `python -m inspect_ai` from the repository root (see above) gets raw metric
  values without the target conventions.

The CSV is long-form (`Created, Run ID, Benchmark, Metric, Score, Runtime (sec)`),
not the legacy wide `summary_brief.csv`/`summary_full.csv`. `Benchmark` is the
registered task name, so the split (`bias_type`, `kind`), the models and their
settings come from the `.eval` log matched by `Run ID`.

## MLflow reporting (optional)

Install the `mlflow` extra and set `LLM_BEHAVIOR_EVAL_MLFLOW_ENABLED=1`. A hook
registered with Inspect then logs each run's aggregate metrics and uploads its
`.eval` files, attaching to an explicit, active or LoRA-adapter-inferred run.
It is configured only through environment variables and reads or sets no
inference settings. See [MLFLOW_INTEGRATION.md](MLFLOW_INTEGRATION.md) and
[`examples/mlflow_example.py`](examples/mlflow_example.py). Do not enable it
together with the platform's own Inspect-MLflow integration for the same run.

## Migrating from the legacy CLI

`llm-behavior-eval <model> <preset>` is retired: it runs nothing, prints how to
run the preset with `inspect eval`, and exits with status 2. Presets map to
tasks and `-T` arguments only; they carry no inference settings.

| Legacy preset | Inspect task (prefix `llm_behavior_eval/`) | `-T` arguments |
| --- | --- | --- |
| `bias:<type>`, `unbias:<type>` | `bbq` | `bias_type=<type> kind=bias` or `kind=unbias` |
| `unqover:bias:<type>` | `unqover` | `bias_type=<type> kind=bias` |
| `bloom:bias:<type>`, `bloom:unbias:<type>` | `bloom` | `bias_type=<type> kind=bias` or `kind=unbias` |
| `bloom:bias:<type>:ambiguous` | `bloom` | `bias_type=<type> kind=ambiguous` |
| `hallu` (`hallucination`) | `halueval` | none |
| `hallu-med` (`hallucination-med`) | `medhallu` | none |
| `prompt-injection` | `prompt_injection` | none |
| `chinese_censorship` | `ccpc` | none |
| `refusal:xstest`, `refusal:orbench` | `xstest`, `orbench` | none |

`<preset>:all` expands to one task per type, and `refusal:all` to both refusal
tasks, in catalog order. From Python, `list_dataset_presets()` and
`get_dataset_preset(name)` return each preset's `dataset_ids` (unchanged) and
`inspect_tasks`, a tuple of `InspectTaskMapping(task_name, task_args,
dataset_id)` with `cli_args` for the `-T` options;
[`examples/presets_customization.py`](examples/presets_customization.py) runs
presets as Inspect tasks with per-role settings. Inspect runs tasks from
different legacy families in one `eval()`.

| Legacy option | Inspect |
| --- | --- |
| `<model>` | `--model <provider>/<model>` (`hf/`, `vllm/`, `openai-api/<service>/`, ...) |
| `--judge-model`, judge engine | `--model-role grader=<provider>/<model>` |
| `--max-samples N` | `--limit N` |
| answer/judge temperature, token budgets, thinking | per-role Inspect settings (above), or `--run-config <benchmark>/parity.yaml` |
| `--base-output-dir`, `--model-output-dir` | `--log-dir` |
| `--use-mlflow`, `--mlflow-*` | `LLM_BEHAVIOR_EVAL_MLFLOW_*` environment variables |
| vLLM and HF loading flags | `-M` provider arguments; see [INSPECT_MIGRATION.md](INSPECT_MIGRATION.md#legacy-flag-mapping) |

Outputs changed: `.eval` logs replace `responses.json`, `generations.jsonl`,
`run_config.json` and per-dataset `metrics.csv`; inspect-brief's CSV replaces
`summary_brief.csv` and `summary_full.csv`. The Notion uploader, which read the
legacy summaries, is removed (see `hirundo_notion_tools/README.md`). The legacy
evaluator classes (`EvaluateFactory`, `EvaluationConfig`, ...) still import, with
a `DeprecationWarning`, for parity comparison only; they will be removed.

## Development

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv sync --locked --extra brief --extra mlflow
pytest                       # pyproject.toml sets --import-mode=importlib
ruff format --check --diff . && ruff check . && pyrefly check
```

`tests/test_inspect_smoke.py` and `tests/test_inspect_brief.py` run every
registered task offline on local datasets with mock answer and grader models,
and check inspect-brief's hook and CLI on the real `.eval` logs. The dev
container (`.devcontainer/`) uses Python 3.12 and installs the `brief` and
`mlflow` extras; set `LLM_BEHAVIOR_EVAL_INSTALL_EXTRAS` (for example
`brief,mlflow,vllm`, or empty for the base package) before running
`.devcontainer/setup.sh` to change that. See [AGENTS.md](AGENTS.md) for
contributor conventions and [INSPECT_MIGRATION.md](INSPECT_MIGRATION.md) for
how to add a benchmark.

## Tested on

The legacy engine was validated with these evaluated models:
`google/gemma-3-12b-it`, `meta-llama/Meta-Llama-3.1-8B-Instruct`,
`meta-llama/Llama-3.2-3B-Instruct`, `google/gemma-7b-it`, `google/gemma-2b-it`
and `google/gemma-3-4b-it`, with `google/gemma-3-12b-it` and
`meta-llama/Llama-3.3-70B-Instruct` judges. Native-task parity with the legacy
engine is tracked per benchmark in each package README.

## License

This project is licensed under the MIT License. See the LICENSE file for more information.
