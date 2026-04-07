# eval-assistant — LLM Behavior Evaluation Assistant

**Version:** 1.0.0

## Purpose

Guide users through configuring and running an `llm-behavior-eval` evaluation interactively. Walk through each decision step-by-step, catch common configuration mistakes early, and emit a ready-to-run CLI command.

---

## Workflow

### Step 1 — Choose behavior(s) to evaluate

Present the taxonomy below and ask the user what they want to measure:

**Social Bias — BBQ dataset**
- `bias:<type>` — measure stereotype bias in the model's answers
- `unbias:<type>` — measure bias in an unambiguous (disambiguation) context
- Supported `<type>` values: `age`, `gender`, `nationality`, `physical`, `race`, `religion`, or `all`
- Examples: `bias:gender`, `unbias:race`, `bias:all`

**Social Bias — UNQOVER dataset**
- `unqover:bias:<type>` — UNQOVER supports `bias` direction only (no `unbias`)
- Supported `<type>` values: `gender`, `nationality`, `race`, `religion`, or `all`
- Example: `unqover:bias:nationality`

**Hallucinations**
- `hallu` — general hallucination detection (HaluEval)
- `hallu-med` — medical domain hallucination detection (Med-Hallu)

**Prompt Injection**
- `prompt-injection` — Purple Llama prompt injection vulnerability (yes/no grading)

Multiple behaviors can be combined with commas:
`bias:gender,hallu,prompt-injection`

---

### Step 2 — Identify the model

Ask for:
- **Model**: HuggingFace repo ID (e.g. `meta-llama/Llama-3.1-8B-Instruct`) or local filesystem path
- **Token**: If the model is gated (requires HF login), ask for `--model-token`

**`--trust-remote-code` handling**: automatically enabled for known-safe providers (`hirundo-io`, `nvidia`, `meta-llama`, `google`, `aisingapore`, `LGAI-EXAONE`). For all other providers, ask the user whether to pass `--trust-remote-code`.

---

### Step 3 — Choose inference engine

Ask whether the user has a GPU and which engine to use:

**`transformers`** (default)
- Works everywhere, automatic batch sizing, no extra install
- Add `--use-4bit` if GPU memory is limited (loads model in 4-bit via bitsandbytes)

**`vllm`** (higher throughput, GPU required)
- `--inference-engine vllm` applies vLLM to both model and judge
- Or tune separately: `--model-engine vllm` / `--judge-engine vllm`
- ⚠️ Do NOT combine `--inference-engine` with `--model-engine` / `--judge-engine`
- Optional tuning: `--vllm-gpu-memory-utilization` (default 0.9), `--vllm-max-model-len`

---

### Step 4 — LoRA adapter (optional)

Ask if the user wants to evaluate a LoRA-adapted model:
- **Requires vLLM** — if they say yes and haven't chosen vLLM, update the engine choice
- `--lora-path-or-repo-id <value>`: accepts local path, HF repo ID, or URI schemes:
  - `mlflow://<run_id>/<artifact_path>`
  - `git://<repo_url>#<rev>:<subdir>`
  - `s3://<bucket>/<path>` or `gs://<bucket>/<path>`

---

### Step 5 — Judge model (optional override)

The judge model grades free-text responses. Default: `google/gemma-3-12b-it`

Ask if the user wants to change it:
- `--judge-model <repo_id_or_path>`
- `--judge-engine vllm` if running judge on GPU
- `--use-4bit-judge` for memory-constrained setups
- `--judge-token` if the judge model is gated (defaults to `--model-token` if not set)

---

### Step 6 — MLflow tracking (optional)

Ask if they want experiment tracking:
- `--use-mlflow` (flag to enable)
- `--mlflow-tracking-uri <uri>`
- `--mlflow-experiment-name <name>`
- `--mlflow-run-name <name>` (auto-generated if omitted)
- Auth: via env vars `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`, or `--mlflow-username` / `--mlflow-password`
- `--mlflow-artifact-path-subfolder <name>` — use `"timestamp"` to auto-prepend a timestamp

---

### Step 7 — Generate the command

Emit the final `llm-behavior-eval` command and explain the outputs.

**Output files** (default location: `~/.local/share/llm-behavior-eval/results/<model>/`):
- `summary_brief.csv` — per-dataset error rate, stereotype bias, empty response ratio
- `summary_full.csv` — full metric breakdown
- `<dataset>.json` — raw model responses and per-sample grades

Useful flags to mention:
- `--base-output-dir <path>` — override the results directory
- `--max-samples <n>` — limit dataset size for quick tests (default 500; set ≤0 for the full dataset)
- `--replace-existing-output` — re-run and overwrite cached results

---

## CLI Reference

```
llm-behavior-eval <MODEL> <BEHAVIOR> [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--model-token` | None | HF token for gated model |
| `--judge-model` | `google/gemma-3-12b-it` | Judge model repo/path |
| `--judge-token` | same as model-token | HF token for judge |
| `--inference-engine` | None | `vllm` or `transformers` (overrides model+judge engine) |
| `--model-engine` | `transformers` | Engine for model inference only |
| `--judge-engine` | `transformers` | Engine for judge inference only |
| `--use-4bit` | False | 4-bit quantization for model (bitsandbytes) |
| `--use-4bit-judge` | False | 4-bit quantization for judge |
| `--max-samples` | 500 | Samples per dataset (≤0 = full dataset) |
| `--batch-size` | auto | Batch size for model inference |
| `--lora-path-or-repo-id` | None | LoRA adapter path/repo (vLLM only) |
| `--base-output-dir` | platform default | Results output directory |
| `--reasoning` | False | Enable reasoning in chat template |
| `--trust-remote-code` | auto | Trust remote model code |
| `--replace-existing-output` | False | Overwrite cached results |
| `--use-mlflow` | False | Enable MLflow tracking |
| `--mlflow-tracking-uri` | None | MLflow server URI |
| `--mlflow-experiment-name` | None | MLflow experiment name |
| `--vllm-gpu-memory-utilization` | 0.9 | GPU memory fraction for vLLM |
| `--vllm-max-model-len` | None | Max token context length for vLLM |

---

## Common Examples

```bash
# Evaluate gender bias (BBQ)
llm-behavior-eval meta-llama/Llama-3.1-8B-Instruct bias:gender

# Full BBQ bias sweep + hallucination, vLLM, 4-bit
llm-behavior-eval meta-llama/Llama-3.1-8B-Instruct bias:all,hallu \
  --inference-engine vllm --use-4bit

# Multiple bias datasets across BBQ and UNQOVER
llm-behavior-eval meta-llama/Llama-3.1-8B-Instruct \
  bias:gender,unqover:bias:nationality

# Evaluate a LoRA adapter (vLLM required)
llm-behavior-eval meta-llama/Llama-3.1-8B-Instruct prompt-injection \
  --model-engine vllm \
  --lora-path-or-repo-id my-org/my-lora-adapter

# Quick smoke test with 50 samples
llm-behavior-eval google/gemma-3-12b-it hallu --max-samples 50

# With MLflow tracking
llm-behavior-eval meta-llama/Llama-3.1-8B-Instruct bias:gender \
  --use-mlflow \
  --mlflow-tracking-uri http://localhost:5000 \
  --mlflow-experiment-name "llama-bias-eval"

# Gated model with custom output directory
llm-behavior-eval meta-llama/Llama-3.1-8B-Instruct hallu-med \
  --model-token $HF_TOKEN \
  --base-output-dir ./my-results
```

---

## Common Mistakes to Catch

- **`--inference-engine` + `--model-engine`/`--judge-engine`**: mutually exclusive — use one or the other
- **LoRA without vLLM**: LoRA adapters require `--model-engine vllm` or `--inference-engine vllm`
- **`unqover:unbias:*`**: UNQOVER has no `unbias` direction — only `unqover:bias:<type>` is valid
- **Negative `--max-samples`**: values ≤0 run the full dataset (not zero samples)
