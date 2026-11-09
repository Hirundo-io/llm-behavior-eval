# LLM Behavior Eval

Evaluate LLM behavior with native [Inspect](https://inspect.aisi.org.uk/) tasks,
independent answer and grader models, and Inspect `.eval` logs.

## Benchmarks

Task names below have the prefix `llm_behavior_eval/`. Bias families accept
`bias_type` and `kind`; see their package README for supported splits.

| Task | Hugging Face dataset | Primary metrics |
| --- | --- | --- |
| `bbq` | `hirundo-io/bbq-<type>-<kind>-free-text` | error, accuracy, stereotype bias |
| `bloom` | `hirundo-io/bloom-<type>-<kind>-free-text` (ambiguous uses `ambiguous-bias`) | error, accuracy, stereotype bias |
| `unqover` | `hirundo-io/unqover-<type>-bias-free-text` | error, accuracy, stereotype bias |
| `halueval` | `hirundo-io/halueval` | accuracy (%) |
| `medhallu` | `hirundo-io/medhallu` | accuracy (%) |
| `ccpc` | `hirundo-io/ccpc-500` | censorship rate, judged fraction |
| `xstest` | `hirundo-io/XSTest` | safe/unsafe refusal rates |
| `orbench` | `hirundo-io/or-bench` | safe/unsafe refusal rates |
| `prompt_injection` | `hirundo-io/prompt-injection-purple-llama` | attack success rate (%) |

Incomplete-response diagnostics are retained in logs. `inspect_template` is a
small wiring demonstration, not a tenth benchmark.

## Install

Python 3.12–3.13 is supported for the package and all reporting extras.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e '.[brief]'
```

For CPU development/reporting, use `uv sync --locked --extra brief --extra mlflow`.
Optional `[tokenizers]`, `[hf]`, and `[vllm]` extras enable local tokenization or
Inspect providers. The base and dev group do not install model runtimes.
Optional extras include `vllm`, `mlflow`, and `brief`. The devcontainer uses Python
3.12 and defaults to brief; select extras with
`LLM_BEHAVIOR_EVAL_INSTALL_EXTRAS=brief,mlflow bash .devcontainer/setup.sh`.

The historical Gemma 4 GPU stack pins vLLM 0.30 with Transformers 5.20.2;
do not downgrade Transformers to bypass its guarded per-layer configuration.
Install the locked stack together, with compatible NVIDIA drivers, CUDA toolkit,
and Python development headers for JIT compilation. This prior engine validation
is not a claim of native task GPU parity. Inspect model arguments replace old
engine flags; consult provider help and each task's README before loading adapters.

## Run

Run outside the checkout to ensure Inspect resolves installed registered names
rather than similarly named source directories. Production preprocessing requires
a matching explicit answer tokenizer and a native vLLM or HF answer provider:

```bash
mkdir -p results/native
cd results/native
inspect eval llm_behavior_eval/bbq -T bias_type=gender -T kind=bias \
  -T compatibility_tokenizer=/models/answer-snapshot \
  --model vllm//models/answer-snapshot --model-role grader=vllm//models/judge-snapshot \
  --log-dir logs
inspect view --log-dir logs
```

Use immutable local snapshots matching the deployments. The family applies native
thinking controls independently to both roles. API graders may instead use the
existing external-thinking assertions; their hidden tokenization remains
unqualified. HF configuration is shown in [the migration reference](INSPECT_MIGRATION.md#native-hf-preprocessing). Generic API answers remain unsupported.
See [production preprocessing and qualification](scripts/parity/README.md#production-preprocessing-contract-llm-169)
for task options, DSTA deployment scope, and separate answer/judge processes.
Local/offline datasets use `-T dataset_path=/absolute/path/to/saved-dataset`;
use Hugging Face `save_to_disk` format. Physical paths do not change task identity.
CCPC validates the full pinned 500-row snapshot, including its fingerprint,
before applying sample caps. `--limit 2` controls Inspect sample execution.

Thinking policies differ by benchmark; they are assertions about deployments,
not a universal switch. Bias and Purple Llama expose
`externally_configured_thinking`; HaluEval/Med-Hallu expose `thinking_disabled`
and `grader_thinking_disabled`; CCPC and refusal tasks expose
`answer_thinking_configured` and `judge_thinking_configured`. Refusal tasks also
have `thinking` (default true). HF model objects must receive the appropriate
`enable_thinking` and `do_sample` arguments independently for each role.
See each benchmark's README and [migration reference](INSPECT_MIGRATION.md).

[The preset example](examples/presets_customization.py) expands catalog mappings
into native tasks. `list_dataset_presets()` and `expand_dataset_preset()` retain
ordered asset identities; `preset.inspect_tasks` supplies task names and arguments.

## Migrating from the old CLI

`llm-behavior-eval <model> <preset>` is retired. Use Inspect CLI or Python `eval`.
Native outputs are `.eval` logs and optional brief CSV; old wide summary files,
response JSON directories, and old CLI engine flags are no longer the public flow.
Version 0.2.0 removes all legacy evaluator/configuration/engine imports. Mock tests verify wiring and scoring,
not real-model parity; tokenizer truncation, chat rendering, and generation
backend differences remain documented per task. See
[MLflow integration](MLFLOW_INTEGRATION.md) for the optional hook.

## Native Inspect summaries

For native tasks, `.eval` logs are the source of truth and
[inspect-brief](https://github.com/Hirundo-io/inspect-brief) replaces the legacy
`summary_brief.csv` / `summary_full.csv` reporting flow. Native tasks do not write per-dataset `metrics.csv`/`responses.json`.
The old evaluator and CSV writers are removed.

Use Python **3.12 or 3.13** for the package and optional brief integration.

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
and their thinking policies as described in each benchmark section. All nine benchmark tasks in the table above are integrated; `inspect_template`
is a wiring demonstration. Remove `INSPECT_BRIEF_TASKS` to include all tasks,
or set a comma-separated list (not a wildcard). Purple Llama ASR is already 0–100.

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
and are excluded from brief. The refusal scorer emits scalar sample scores;
its aggregate metric returns named rates and counts. Brief selects
`safe_refusal_rate`, `unsafe_refusal_rate`, and `incomplete_response_rate`.
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

## License

See [LICENSE](LICENSE).
