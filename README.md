# [llm-behavior-eval](https://github.com/Hirundo-io/llm-behavior-eval) &middot; [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/hirundo-io/llm-behavior-eval/blob/main/LICENSE) [![pypi version](https://img.shields.io/pypi/v/llm-behavior-eval)](https://pypi.org/project/llm-behavior-eval/)

[![pyrefly](https://github.com/hirundo-io/llm-behavior-eval/actions/workflows/pyrefly.yaml/badge.svg)](https://github.com/hirundo-io/llm-behavior-eval/actions/workflows/pyrefly.yaml) [![ruff](https://github.com/hirundo-io/llm-behavior-eval/actions/workflows/ruff.yaml/badge.svg)](https://github.com/hirundo-io/llm-behavior-eval/actions/workflows/ruff.yaml) [![Unit tests](https://github.com/hirundo-io/llm-behavior-eval/actions/workflows/tests.yaml/badge.svg)](https://github.com/hirundo-io/llm-behavior-eval/actions/workflows/tests.yaml) [![Vulnerability scan](https://github.com/hirundo-io/llm-behavior-eval/actions/workflows/vulnerability-scan.yaml/badge.svg)](https://github.com/hirundo-io/llm-behavior-eval/actions/workflows/vulnerability-scan.yaml)

A Python 3.10+ toolkit for measuring undesirable LLM behaviors with Hugging Face models or local model paths.

It evaluates bias, hallucinations, prompt injection, refusal behavior, and Chinese censorship. All evaluations support Transformers instruct models. See [Tested on](#tested-on) for the models used to validate the pipeline.

## What it evaluates

This toolkit evaluates five classes of behaviors:

- **Bias (BBQ, UNQOVER)**
  - **BBQ** (Bias Benchmark for QA): hand‑crafted questions that probe stereotypes across protected dimensions. Supports paired splits:
    - **bias** (ambiguous) and **unbias** (disambiguated) for: `gender`, `race`, `nationality`, `physical`, `age`, `religion`.
    - Only BBQ provides both ambiguous and disambiguated versions.
  - **UNQOVER**: crowd‑sourced templates probing stereotypes; provides only the ambiguous/bias split for: `religion`, `gender`, `race`, `nationality`.

- **Hallucinations (HaluEval, Med‑Hallu)**
  - **HaluEval (halueval)**: general‑domain factuality/consistency checks.
  - **Med‑Hallu (medhallu)**: medical‑domain hallucination benchmark.

- **Prompt Injection (Purple Llama)**
  - **Purple Llama Prompt Injection**: measures susceptibility to instruction overriding and jailbreaks using curated prompt‑injection attacks. Reuses the hallucination judging pipeline with Yes/No grading.

- **Chinese censorship (CCPC-Bench)**
  - **CCPC-Bench**: measures CCP-aligned political censorship using the pinned 500-row [`hirundo-io/ccpc-500`](https://huggingface.co/datasets/hirundo-io/ccpc-500) test split. This implements task version 6 for compatibility with the public benchmark. The frozen `ccpc_binary_v4` judge emits bare `True` or `False`; raw `metrics.csv` reports fractional `censorship_rate`, `stderr`, and `judged_fraction`. Presentation summaries (`summary_brief.csv` and `summary_full.csv`) report `Censorship rate (%) ⬇️` on the 0–100 scale, matching the Purple Llama percentage/direction convention. Compatible summary consumers should read that header without multiplying by 100; consumers requiring the previous fractional field should use `metrics.csv`. Version strings and revision hashes are unchanged.
  - CCPC upload through the bundled `hirundo_notion_tools` uploader is not yet supported.

- **Refusal (XSTest, OR-Bench)**
  - **[XSTest](https://github.com/paul-rottger/xstest)**: tests exaggerated safety behavior. It combines prompts that a model should answer with prompts that it should refuse, so it exposes both over-refusal on safe requests and under-refusal on unsafe requests.
  - **[OR-Bench](https://github.com/justincui03/or-bench)**: evaluates appropriate refusal behavior using the same safe/unsafe distinction. The evaluator counts both direct and indirect refusals as refusals.

Dataset identifiers:

- BBQ: `hirundo-io/bbq-<bias_type>-<bias|unbias>-free-text`
- UNQOVER: `unqover/unqover-<bias_type>-bias-free-text`
- HaluEval: `hirundo-io/halueval`
- Med‑Hallu: `hirundo-io/medhallu`
- Prompt Injection (Purple Llama): `hirundo-io/prompt-injection-purple-llama`
- Chinese censorship (CCPC-Bench): CLI preset `chinese_censorship`; Hugging Face repository `hirundo-io/ccpc-500`
- XSTest: `hirundo-io/XSTest`
- OR-Bench: `hirundo-io/or-bench`

Pass the behavior preset as the second positional CLI argument:

- BBQ: `bias:<bias_type>` or `unbias:<bias_type>`
- UNQOVER: `unqover:bias:<bias_type>`
- Hallucinations:
  - HaluEval: `hallu`
  - Med‑Hallu: `hallu-med`
- Prompt Injection:
  - Purple Llama: `prompt-injection`
- Chinese censorship:
  - CCPC-Bench: `chinese_censorship` with the pinned `--judge-model google/gemma-4-26B-A4B-it`
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

Install Python 3.10.12 through 3.13, then create a virtual environment and install the package:

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

## Run the Evaluator

Use the CLI with the required model and behavior positional arguments. The behavior preset selects datasets for you.

```bash
llm-behavior-eval <model_repo_or_path> <behavior_preset>
```

You can pass comma-separated presets from one evaluator family, such as
`bias:gender,unbias:gender`. The CLI runs one evaluator family per invocation
and rejects mixed families because they use different evaluator and scoring paths.
Run each family separately.

### Examples

- **BBQ (bias)** — evaluate a model on a biased split (free‑text):
```bash
llm-behavior-eval google/gemma-2b-it bias:gender
```

- **BBQ (unbias)** — evaluate a model on an unambiguous split:
```bash
llm-behavior-eval meta-llama/Llama-3.1-8B-Instruct unbias:race
```

- **UNQOVER (bias)** — use UNQOVER source datasets (UNQOVER does not support 'unbias'):
```bash
llm-behavior-eval google/gemma-2b-it unqover:bias:gender
```

- **BBQ (all bias types)** — iterate all BBQ ambiguous splits:
```bash
llm-behavior-eval meta-llama/Llama-3.1-8B-Instruct bias:all
```

- **UNQOVER (all bias types)** — iterate all UNQOVER bias splits:
```bash
llm-behavior-eval meta-llama/Llama-3.1-8B-Instruct unqover:bias:all
```

- **Hallucination (general)** — HaluEval free‑text:
```bash
llm-behavior-eval google/gemma-2b-it hallu
```

- **Hallucination (medical)** — Med-Hallu:
```bash
llm-behavior-eval meta-llama/Llama-3.1-8B-Instruct hallu-med
```

- **Prompt Injection** — Purple Llama prompt injections:
```bash
llm-behavior-eval meta-llama/Llama-3.1-8B-Instruct prompt-injection
```

- **Chinese censorship.** Run CCPC-Bench with an explicitly selected judge:
```bash
llm-behavior-eval google/gemma-3-12b-it chinese_censorship \
  --judge-model google/gemma-4-26B-A4B-it
```

For a locally mounted CCPC Hub snapshot, set `DatasetConfig(file_path="/opt/assets/ccpc-500", dataset_id="chinese_censorship", dataset_type=DatasetType.BIAS)` in the Python API. The loader selects the `default` configuration and `test` split from that directory and validates all 500 rows against the published content/order fingerprint before applying `max_samples`. Keep the snapshot's Parquet files and directory layout intact. Using `file_path="chinese_censorship"` or `"hirundo-io/ccpc-500"` retains loading from the pinned Hub revision.

- **XSTest (refusal)** — measure over-refusal on safe prompts and refusal on unsafe prompts:
```bash
llm-behavior-eval meta-llama/Llama-3.1-8B-Instruct refusal:xstest
```

- **OR-Bench (refusal)** — evaluate appropriate direct or indirect refusal behavior:
```bash
llm-behavior-eval meta-llama/Llama-3.1-8B-Instruct refusal:orbench
```

- **All refusal benchmarks** — run XSTest and OR-Bench sequentially:
```bash
llm-behavior-eval meta-llama/Llama-3.1-8B-Instruct refusal:all
```

### Local and offline datasets

`DatasetConfig.file_path` is the physical loading source and may be a local
directory. Set `DatasetConfig.dataset_id` to the canonical dataset identifier
so evaluator dispatch, prompts, output names, metrics, and provenance match the
online preset. If omitted, `dataset_id` defaults to `file_path`.

```python
from llm_behavior_eval import DatasetConfig, DatasetType

config = DatasetConfig(
    file_path="/opt/assets/halueval",
    dataset_id="hirundo-io/halueval",
    dataset_type=DatasetType.BIAS,
)
```

Release tooling can call `list_dataset_presets()` to enumerate every supported
preset and its complete `dataset_ids` expansion. `expand_dataset_preset()`
provides the same expansion used by the CLI. Both functions are available from
`llm_behavior_eval`.

### Interpreting refusal results

Both datasets label prompts as **safe** (the model should answer) or **unsafe** (the model should refuse). The evaluator uses a judge model to classify the generated answer. For XSTest, full and partial refusals count as refusals; for OR-Bench, both direct and indirect refusals count as refusals.

Use the two primary metrics together:

- **Safe refusal rate (%) ⬇️**: the share of safe prompts that were refused. Lower is better because it measures over-refusal.
- **Unsafe refusal rate (%) ⬆️**: the share of unsafe prompts that were refused. Higher is better because it measures appropriate refusal.
- **Heuristic safe refusal rate (%) ⬇️** and **heuristic unsafe refusal rate (%) ⬆️**: the same rates estimated from refusal phrases in each answer, without a judge. Use them as a quick diagnostic, not as the benchmark result.
- **Incomplete response rate (%) ⬇️**: the share of configured samples where generation stopped because it reached the length limit. Truncated answers are still judged, so they can count toward the refusal rates.
- **Unknown finish reason rate (%) ⬇️**: the share of configured samples with a finish reason other than a normal stop or a length limit. Those responses are not judged.
- **Judge unparseable rate (%) ⬇️**: the share of configured samples where the judge did not produce a recognized refusal class. Those responses are excluded from the judge-based refusal rates.

The diagnostic rates use the configured sample count as their denominator. Unknown or unparseable responses reduce the judged sample set used for the primary rates.

### CLI options

- `--max-samples <N>` — cap how many rows to evaluate per dataset (defaults to 500). Use `0` or any negative value to run the entire split.
- `--use-4bit-judge/--no-use-4bit-judge` — toggle 4-bit (bitsandbytes) loading for the judge model so you can keep the evaluator in full precision while fitting the judge onto smaller GPUs.
- `--model-token` / `--judge-token` — supply Hugging Face credentials for the evaluated or judge models (the judge token defaults to the model token when omitted).
- `--judge-model` — pick a different judge checkpoint; the default is `google/gemma-3-12b-it`.
- `--inference-engine vllm` / `--inference-engine transformers` — switch between vLLM and transformers backends for the evaluated model. There are also `--model-engine` and `--judge-engine` flags for more explicit control.
- `--vllm-max-model-len` / `--vllm-gpu-memory-utilization` — configure vLLM's maximum context length and GPU memory utilization. Leave the maximum length unset to use the model's native context; the GPU utilization default is 0.8. Override either only after confirming the target GPU's KV-cache capacity; increasing utilization increases that capacity, while lowering it decreases available KV-cache capacity.
- `--vllm-tokenizer-mode`, `--vllm-config-format`, `--vllm-load-format` — forward advanced knobs directly to the underlying vLLM engine when you need to align tokenizer behavior, checkpoint formats, or tool-calling semantics with a particular deployment. Tokenizer mode accepts `auto`, `slow`, `mistral`, or `custom`.
- `--thinking-on/--thinking-off` — enable thinking modes on tokenizers that support them. Unset uses the evaluator-family default: **on for refusal**, off for other behaviors. The judge always runs with thinking off.
- `--enable-thinking-arg-name` — enable thinking argument name in tokenizer's `apply_chat_template` (e.g. 'enable_thinking').
- `--thinking-start-token` / `--thinking-end-token` — Thinking start/end token to use for the model (e.g. '<think>'/'</think>').
- `--use-mlflow` plus `--mlflow-tracking-uri`, `--mlflow-experiment-name`, and `--mlflow-run-name` — configure MLflow tracking for the run.

Need more control or wrappers around the library? Explore the scripts in `examples/` to see how to call the evaluators from Python directly, customize additional knobs, or embed the run inside your own orchestration logic.

See `examples/presets_customization.py` for a minimal script-based workflow.

### MLflow Integration (Optional)

Enable MLflow tracking with `--use-mlflow` to log simple parameters, metrics and artifacts.

Install: `pip install llm-behavior-eval[mlflow]` or `pip install mlflow`.

CLI example:
```bash
llm-behavior-eval google/gemma-2b-it bias:gender --use-mlflow
```

To find more documentation: see [`MLFLOW_INTEGRATION.md`](./MLFLOW_INTEGRATION.md).
Programmatic example: see [`examples/mlflow_example.py`](./examples/mlflow_example.py).

### Radar charts (Optional)

Add `--plot` to write one interactive Plotly radar chart per percentage metric in `summary_full.csv`, with the model's datasets around the axis. Charts are saved to `results/<model>/plots/radar_<metric>.html` and, when MLflow is enabled, logged to the run under `plots/`. A plotting failure is logged and does not fail the evaluation.

Install: `pip install llm-behavior-eval[plotly]`.

```bash
llm-behavior-eval google/gemma-2b-it bias:all --plot
```

To compare several models or label your own categories, call `draw_radar_chart` from `llm_behavior_eval.evaluation_utils.plotting` with one value list per model.

## Output

Evaluation reports are saved as metrics CSV files and full response JSON files in the results directory. By default, the CLI writes to:

- macOS: `~/Library/Application Support/llm-behavior-eval/results`
- Linux/Ubuntu: `$XDG_DATA_HOME/llm-behavior-eval/results` (or `~/.local/share/llm-behavior-eval/results` if `XDG_DATA_HOME` is unset)
- Windows: `%LOCALAPPDATA%\llm-behavior-eval\results` (fallback: `%APPDATA%\llm-behavior-eval\results`)

Override the default with `--base-output-dir` when you need a different path. You can also use `--model-output-dir` to explicitly override the name of the model under that base path; otherwise, the model path or repo ID will be used, with an added stub if using a LoRA adapter.

Outputs are organised as `results/<model>/<dataset>_<dataset_type>_<text_format>/`.
Per‑model summaries are saved as `results/<model>/summary_full.csv` (full metrics) and `results/<model>/summary_brief.csv`.

`summary_brief.csv` contains the following columns: `Dataset`, `Thinking`, and one or more metric columns (`Accuracy`/`Error`/`Attack success rate`). Labels are inferred as follows:

- BBQ: `BBQ: <gender|race|nationality|physical|age|religion> <bias|unbias>`
- UNQOVER: `UNQOVER: <religion|gender|race|nationality> <bias>`
- Hallucination: `halueval` or `medhallu`
- Prompt Injection: `prompt-injection-purple-llama`
- Chinese censorship: `chinese_censorship`
- Refusal: `XSTest` or `or-bench`

## Tested on

Validated the pipeline on the following models:

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
