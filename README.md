# [llm-behavior-eval](https://github.com/Hirundo-io/llm-behavior-eval) &middot; [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/hirundo-io/llm-behavior-eval/blob/main/LICENSE) [![pypi version](https://img.shields.io/pypi/v/llm-behavior-eval)](https://pypi.org/project/llm-behavior-eval/)

[![pyrefly](https://github.com/hirundo-io/llm-behavior-eval/actions/workflows/pyrefly.yaml/badge.svg)](https://github.com/hirundo-io/llm-behavior-eval/actions/workflows/pyrefly.yaml) [![ruff](https://github.com/hirundo-io/llm-behavior-eval/actions/workflows/ruff.yaml/badge.svg)](https://github.com/hirundo-io/llm-behavior-eval/actions/workflows/ruff.yaml) [![Unit tests](https://github.com/hirundo-io/llm-behavior-eval/actions/workflows/tests.yaml/badge.svg)](https://github.com/hirundo-io/llm-behavior-eval/actions/workflows/tests.yaml) [![Vulnerability scan](https://github.com/hirundo-io/llm-behavior-eval/actions/workflows/vulnerability-scan.yaml/badge.svg)](https://github.com/hirundo-io/llm-behavior-eval/actions/workflows/vulnerability-scan.yaml)

A Python 3.11+ native Inspect toolkit for measuring undesirable LLM behaviors with Hugging Face models or local model paths.

It evaluates bias, hallucinations, prompt injection, refusal behavior, and Chinese censorship. All evaluations support Transformers instruct models. See [Tested on](#tested-on) for the models used to validate the pipeline.

## What it evaluates

This toolkit evaluates five classes of behaviors:

- **Bias (BBQ, UNQOVER, BLOOM)**
  - **BBQ** (Bias Benchmark for QA): hand‑crafted questions that probe stereotypes across protected dimensions. Supports paired splits:
    - **bias** (ambiguous) and **unbias** (disambiguated) for: `gender`, `race`, `nationality`, `physical`, `age`, `religion`.
    - BBQ provides both ambiguous and disambiguated versions.
  - **UNQOVER**: crowd‑sourced templates probing stereotypes; provides only the ambiguous/bias split for: `religion`, `gender`, `race`, `nationality`.

- **Hallucinations (HaluEval, Med‑Hallu)**
  - **HaluEval (halueval)**: general‑domain factuality/consistency checks.
  - **Med‑Hallu (medhallu)**: medical‑domain hallucination benchmark.

- **Prompt Injection (Purple Llama)**
  - **Purple Llama Prompt Injection**: measures susceptibility to instruction overriding and jailbreaks using curated prompt‑injection attacks. Reuses the hallucination judging pipeline with Yes/No grading.

- **Chinese censorship (CCPC-Bench)**
  - **CCPC-Bench**: measures CCP-aligned censorship on pinned `hirundo-io/ccpc-500`.
    Native metrics and versioning are documented in [the CCPC guide](llm_behavior_eval/ccpc/README.md).
    The bundled Notion uploader has been retired.

- **Refusal (XSTest, OR-Bench)**
  - **[XSTest](https://github.com/paul-rottger/xstest)**: tests exaggerated safety behavior. It combines prompts that a model should answer with prompts that it should refuse, so it exposes both over-refusal on safe requests and under-refusal on unsafe requests.
  - **[OR-Bench](https://github.com/justincui03/or-bench)**: evaluates appropriate refusal behavior using the same safe/unsafe distinction. The evaluator counts both direct and indirect refusals as refusals.

Dataset identifiers:

- BBQ: `hirundo-io/bbq-<bias_type>-<bias|unbias>-free-text`
- UNQOVER: `hirundo-io/unqover-<bias_type>-bias-free-text` (native Inspect: [UNQOVER guide](llm_behavior_eval/unqover/README.md))
- HaluEval: `hirundo-io/halueval`
- Med‑Hallu: `hirundo-io/medhallu`
- Prompt Injection (Purple Llama): `hirundo-io/prompt-injection-purple-llama`
- Chinese censorship (CCPC-Bench): CLI preset `chinese_censorship`; Hugging Face repository `hirundo-io/ccpc-500`
- XSTest: `hirundo-io/XSTest`
- OR-Bench: `hirundo-io/or-bench`

The stable Python preset catalog supports these keys (the legacy CLI is retired):

- BBQ: `bias:<bias_type>` or `unbias:<bias_type>`
- UNQOVER: `unqover:bias:<bias_type>`
- Hallucinations:
  - HaluEval: `hallu`
  - Med‑Hallu: `hallu-med`
- Prompt Injection:
  - Purple Llama: `prompt-injection`
- Chinese censorship:
  - CCPC-Bench: `chinese_censorship` with explicit grader `hf/google/gemma-4-26B-A4B-it`
- Refusal:
  - XSTest: `refusal:xstest`
  - OR-Bench: `refusal:orbench`
  - Both: `refusal:all`

You can also run across all supported bias types using `all`:

- BBQ (all ambiguous/bias splits): `bias:all`
- BBQ (all unambiguous/unbias splits): `unbias:all`
- UNQOVER (all bias splits): `unqover:bias:all`
---

## Installation

Install Python 3.11 through 3.13, then create a virtual environment and install the package:

```bash
# Create and activate a virtual environment.
python -m venv .venv
source .venv/bin/activate

# Install the package.
pip install llm-behavior-eval
```

If you use [uv](https://docs.astral.sh/uv/), run `uv venv .venv` and
`uv pip install llm-behavior-eval` instead. For a local checkout, install with
`pip install -e .` or `uv pip install -e .`.

### vLLM extra

The `vllm` extra selects `vllm>=0.30.0,<0.31` on Linux x86_64/aarch64. The
Transformers range is `>=5.20.2,<5.21`. vLLM 0.30 resolves Gemma 4 attention
parameters from each layer config, preserving heterogeneous head dimensions and
KV-head counts. vLLM 0.23 with Transformers 5.20.2 instead reads global
`config.head_dim` and raises `AmbiguousGlobalPerLayerAttributeError`. Do not disable
that guard or downgrade Transformers to work around it. See the upstream
[per-layer resolver](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/transformers_utils/configs/gemma4.py)
and [architecture converter](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/transformers_utils/model_arch_config_convertor.py).
As of 2026-10-20, the newer 0.31/0.32 releases cap Transformers below 5.20.2; they are
excluded deliberately to preserve this tested Transformers version.

Use the lockfile to install the complete stack together. On Linux, vLLM 0.30
requires Torch 2.13.0; its wheels select CUDA 13.0.3 and Triton 3.7.1. Do not
independently replace Torch or Triton. NVIDIA driver 580 or newer is required for
the CUDA 13 runtime. Runtime libraries come from the wheels; FlashInfer JIT
compilation also needs a compatible CUDA toolkit with `nvcc`, C++ build tools,
and `ninja` on PATH (the latter is installed with vLLM). The validation host has
CUDA 13.0.88 at `/usr/local/cuda`.
The project supports Python 3.10.12–3.13; the Gemma 4 GPU validation uses CPython
3.12.3, an H100, and driver 580.159.04. macOS uses the Transformers backend;
the vLLM dependency is excluded there by a platform marker.

```bash
# Choose an interpreter with matching development headers.
uv venv --python /usr/bin/python3.12 .venv-gemma4
UV_PROJECT_ENVIRONMENT=.venv-gemma4 uv sync --locked --no-dev --extra vllm
source .venv-gemma4/bin/activate
# Activation also puts the wheel-provided ninja/build tools on PATH.
.venv-gemma4/bin/python -c 'import pathlib, sysconfig; p = pathlib.Path(sysconfig.get_path("include"), "Python.h"); print(p); assert p.is_file()'
```

A Triton C-extension compilation error mentioning missing `Python.h` is a Python
build-environment problem, separate from Gemma 4 configuration compatibility.
Install development headers matching the selected interpreter (for example,
`python3.12-dev` for the distribution Python 3.12), or choose a uv-managed Python
that includes its headers. Do not point a Python 3.12 build at another minor
version of the headers. For shared installations, validate in a separate venv;
`UV_PROJECT_ENVIRONMENT` prevents `uv sync` from modifying the shared `.venv`.

Gemma 4 text generation uses `runner="generate"` and
`language_model_only=True`; judges always use this text-only path and never inherit
the evaluated model adapter. Rank-110 LoRA adapters require
`--vllm-max-lora-rank 128` (LoRA is enabled automatically when an adapter is supplied); use
`--vllm-language-model-only` for a text-only evaluated model too.

vLLM 0.30 does not support bitsandbytes quantization. Leave `--use-4bit` and
`--use-4bit-judge` disabled for vLLM, or select the Transformers backend for the
role that requires bitsandbytes 4-bit loading.

## Development Container

The repository ships a VS Code Dev Container definition (`.devcontainer/`). The setup script installs the base project dependencies to keep the image lean. If you need optional extras (for example MLflow or vLLM), set `LLM_BEHAVIOR_EVAL_INSTALL_EXTRAS` before the container runs:

```bash
# Example: install MLflow extra inside the devcontainer
export LLM_BEHAVIOR_EVAL_INSTALL_EXTRAS="mlflow"
bash .devcontainer/setup.sh

# Example: install both MLflow and vLLM (requires more disk space)
export LLM_BEHAVIOR_EVAL_INSTALL_EXTRAS="mlflow,vllm"
bash .devcontainer/setup.sh
```

If the requested extras exhaust the available disk, the script falls back to a base install so the container remains usable. Re-run the script with a smaller set of extras when needed.

## Run with native Inspect

The old `llm-behavior-eval <model> <preset>` command is retired and exits with
migration guidance. Native Inspect owns execution, retries, batching, and logs.
For example, with HF credentials in the environment:

```bash
inspect eval llm_behavior_eval/bbq -T bias_type=gender -T kind=bias \
  --model hf/meta-llama/Llama-3.1-8B-Instruct \
  --model-role grader=hf/google/gemma-3-12b-it --log-dir results/inspect
```

Release tooling can still call `list_dataset_presets()` and
`expand_dataset_preset(name)` from `llm_behavior_eval`. The latter returns the
unchanged ordered asset list. Each catalog entry now also exposes `inspect_tasks`:
one mapping per asset with `task_name`, immutable `task_args` pairs, and logical
`dataset_id`. Pass `dict(mapping.task_args)` to native task constructor keyword arguments.
CCPC's logical ID is `chinese_censorship`; its asset is `hirundo-io/ccpc-500`.
Use `dataset_path` to override physical loading without changing logical identity.

See [the native Python example](examples/presets_customization.py) for mixed
families and an independent grader. See [migration guidance](INSPECT_MIGRATION.md)
for every preset mapping, provider/thinking options, caps, and legacy flag changes.
Purple Llama's `prompt_injection` catalog target is reserved pending integration.

### Output and integrations

Native runs write Inspect `.eval` logs, not `summary_brief.csv`,
`summary_full.csv`, per-dataset response JSON, or `run_config.json`. Read model,
grader, task args, samples, and scores from Inspect logs. LLM-166 owns
inspect-brief reporting; its CSV is long-form, not the old wide summary schema.
Do not assume `Dataset`, `Thinking`, `Accuracy (%)`, or `Error (%)` columns.

The old Notion uploader is removed; see [retirement note](hirundo_notion_tools/README.md).
MLflow migration and `examples/mlflow_example.py` remain owned by LLM-167;
[MLFLOW_INTEGRATION.md](MLFLOW_INTEGRATION.md) describes the legacy integration
until that work lands. `--use-mlflow` no longer runs an evaluation.
Legacy Python evaluator/config imports remain deprecated for parity comparisons
and will be removed under LLM-171 after semantic validation.

## Tested on

Legacy pipeline validation used the following models; this is not a native GPU parity claim:

- `"google/gemma-3-12b-it"`

- `"meta-llama/Meta-Llama-3.1-8B-Instruct"`

- `"meta-llama/Llama-3.2-3B-Instruct"`

- `"google/gemma-7b-it"`

- `"google/gemma-2b-it"`

- `"google/gemma-3-4b-it"`

Using the next models as judges:

- `"google/gemma-3-12b-it"`

- `"meta-llama/Llama-3.3-70B-Instruct"`

## License

This project is licensed under the MIT License. See the LICENSE file for more information.

Native Inspect is the public execution path; the legacy engine remains for parity validation.
See [Inspect migration and shared API](INSPECT_MIGRATION.md) for the reference task,
provider mappings, air-gapped grader setup, and remaining parity limitations.

### BLOOM bias

BLOOM supports `age`, `gender`, and `race` bias types. Legacy presets are
`bloom:bias:<type|all>`, `bloom:unbias:<type|all>`, and
`bloom:bias:<type>:ambiguous`. Paired datasets are
`hirundo-io/bloom-<type>-<bias|unbias>-free-text`; ambiguous-only datasets are
`hirundo-io/bloom-<type>-ambiguous-bias-free-text`.

Native Inspect: `inspect eval llm_behavior_eval/bloom -T bias_type=age -T kind=ambiguous
--model <provider/model> --model-role grader=<provider/judge>` (enter on one line).
Kinds are `bias`, `unbias`, and `ambiguous`; run separate tasks for each type to
expand `all`. The grader must be explicitly configured, including thinking-off
controls for both roles. See [BLOOM task documentation](llm_behavior_eval/bloom/README.md)
for provider setup, offline sources, metric declarations, and unresolved parity gaps.
Legacy brief labels are `Bloom: <type> <kind>` and `Bloom: <type> ambiguous bias`.

## Native Inspect summaries (LLM-166)

For native tasks, `.eval` logs are the source of truth and
[inspect-brief](https://github.com/Hirundo-io/inspect-brief) replaces the legacy
`summary_brief.csv` / `summary_full.csv` reporting flow. The legacy evaluator and
comparison CSV code remain temporarily for parent-coordinated parity/retirement;
native tasks do not invoke them or write per-dataset `metrics.csv`/`responses.json`.

Use Python **3.12 or 3.13** for the optional brief integration (inspect-brief
0.1.1 requires >=3.12). Python 3.11 supports native Inspect without brief;
Python 3.10 is legacy-only. The extra is version-marked and installs no brief
package on those older interpreters.

```sh
uv pip install --python .venv/bin/python -e '.[brief]'
source .venv/bin/activate
mkdir -p results/native
cd results/native
export INSPECT_BRIEF_TARGET_METRICS="$(python -m llm_behavior_eval.target_metrics --path)"
export INSPECT_BRIEF_CSV_PATH=brief_results.csv
export INSPECT_BRIEF_TASKS=llm_behavior_eval/halueval,llm_behavior_eval/medhallu
export INSPECT_BRIEF_SKIP_EXISTING=true
inspect eval llm_behavior_eval/halueval --model mockllm/model \
  --model-role grader=mockllm/grader --log-dir logs
```

Run from this separate directory: a checkout-relative task directory can be
interpreted by Inspect as a source path and produce unqualified names, which do
not match the packaged mapping. For direct Python task construction, use
`task_with(task, name="llm_behavior_eval/bbq")` (from `inspect_ai`) to set the
canonical reporting name.

The mock command demonstrates wiring only; configure real answer/grader models
and their thinking policies as described in each benchmark section. Tasks are
`llm_behavior_eval/{bbq,unqover,bloom,halueval,medhallu,ccpc,xstest,orbench}`;
`inspect_template` is a wiring demonstration. Remove `INSPECT_BRIEF_TASKS` to
include all tasks, or set a comma-separated list (not a wildcard).
The reserved `llm_behavior_eval/prompt_injection` mapping matches the separately
implemented Purple Llama scorer in commit `295ca545`: ASR is **already 0–100**.
Its live task smoke remains pending parent integration in this checkout; the
reporting test uses a synthetic result serialized into a real `.eval` file.

To summarize existing logs without the hook:

```sh
unset INSPECT_BRIEF_CSV_PATH
inspect-brief --log-dir logs \
  --target-metrics "$(python -m llm_behavior_eval.target_metrics --path)" \
  --csv-path brief_results.csv --skip-existing
```

Both flows append rows; `--skip-existing` / `INSPECT_BRIEF_SKIP_EXISTING=true`
prevents repeat task IDs. Use a fresh CSV when changing target metrics, otherwise
skip-existing also skips those changed targets. Only Inspect `.eval` input is
supported, not JSON logs. **Each output CSV must have a single writer process**:
Inspect-brief has no cross-process locking. Export concurrent runs to separate
files, or run one CLI export after evaluations finish.

### Metric selection and units

The installed JSON includes accuracy and error for bias tasks. Inspect-brief
selects by task name, not task parameters. To retain the legacy primary metric,
generate a small configuration for each kind and export each kind's logs
separately (do not mix kinds under one task name in that export):

```sh
python -m llm_behavior_eval.target_metrics --bias-kind bias > bias-metrics.json
python -m llm_behavior_eval.target_metrics --bias-kind unbias > unbias-metrics.json
# Set INSPECT_BRIEF_TARGET_METRICS or --target-metrics to the corresponding file.
```

Bias and BLOOM ambiguous use `error` (lower better); unbias uses `accuracy`
(higher better). UNQOVER is bias-only. The static default exposes both explicitly.
`stereotype_bias` is a signed dimensionless index, displayed without percent,
with the legacy downward direction; zero is neutrality, so a more negative
value is not necessarily a less biased model. It is only emitted when the source
has a stereotype column; otherwise brief reports a missing metric, not zero.

Bias accuracy/error, safe/unsafe refusal, censorship, and incomplete-response
rates are fractions (`is_normalized=true`) displayed as percentages.
HaluEval/Med-Hallu accuracy and Purple Llama ASR are already percentages
(`is_normalized=false`). Accuracy and unsafe refusal point up; error, safe
refusal, censorship, ASR, and incomplete response point down. CCPC undefined
censorship is preserved as `nan`; never interpret this as zero censorship.
Heuristic refusal metrics, counts, and diagnostics remain in the `.eval` log
and are excluded from brief. Refusal dict scores have distinct target metric
keys: `safe_refusal_rate`, `unsafe_refusal_rate`, `incomplete_response_rate`.
Configured targets match the metric key itself; unconfigured exports add
`scorer: metric` prefixes when there are multiple score entries.

### Downstream schema migration

The new CSV is long-form: `Created,Run ID,Benchmark,Metric,Score,Runtime (sec)`.
Each task/metric produces one row; there are no wide metric columns, `Dataset`
or `Thinking` columns, or empty-column dropping. `Benchmark` uses the registered
name (for example `llm_behavior_eval/bbq`), not `BBQ: gender bias`. `Metric` uses
snake-case keys with percentage/direction suffixes, such as `accuracy (%) ⬆️`.
Use the task ID in `Run ID` to join back to `.eval` logs for model, task parameters
(kind/bias_type), thinking configuration, sample responses and diagnostics.
Consumers must migrate explicitly rather than rename the old wide CSV file.
