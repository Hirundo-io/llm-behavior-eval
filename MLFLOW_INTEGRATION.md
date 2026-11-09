# MLflow reporting for native Inspect

LLM-167 retains optional reporting because hirundo-core's
[`LLMBehaviorEvalExecutor`](https://github.com/Hirundo-io/hirundo-core/blob/ba96e66093f7ef6f5b304057216249653d1179f8/h_research/orchestration/evaluators.py)
passes a source run ID and checkpoint artifact subfolder. The platform's
inspect-mlflow integration is not proven to preserve that contract. Its reporting
queue stall is documented in [INFRA-321](https://linear.app/hirundo/issue/INFRA-321).
This implementation is only an Inspect hook, with no evaluation engine or model
loading. The legacy engine's own MLflow logging was removed with it in 0.2.0.

## Enable

Install with `uv pip install -e '.[mlflow]'` in your activated `.venv`.
The package's Inspect entry point (`llm_behavior_eval._registry`) imports
`llm_behavior_eval.hooks.mlflow` explicitly, because `hooks/` is not a benchmark
package and task discovery skips it. Inspect loads every entry point when an
eval starts, so the hook is present for any task, ours or not. Importing it does
not import MLflow, and its `enabled()` is false unless
`LLM_BEHAVIOR_EVAL_MLFLOW_ENABLED=1` (or `true`/`yes`), so Inspect never calls
it otherwise. Use native `inspect eval` or `inspect_ai.eval`; see
`examples/mlflow_example.py`. Do not enable both this hook and platform
inspect-mlflow for the same evaluation.

The hook reports; it does not configure evaluation. Inference settings for the
model under test and the grader are passed to Inspect per role (see
`INSPECT_MIGRATION.md`), and are recorded in the uploaded `.eval` logs.

The legacy CLI is retired. Configure the hook directly with these environment
settings; no CLI translation or flag forwarding is performed:

| Legacy setting | Native environment setting |
| --- | --- |
| `--use-mlflow` / `MlflowConfig` presence | `LLM_BEHAVIOR_EVAL_MLFLOW_ENABLED=1` |
| `--mlflow-tracking-uri` | `MLFLOW_TRACKING_URI` |
| `--mlflow-experiment-name` | `MLFLOW_EXPERIMENT_NAME` (default `llm-behavior-eval`) |
| `--mlflow-run-name` | `LLM_BEHAVIOR_EVAL_MLFLOW_RUN_NAME` (default `inspect-evaluation`) |
| `--mlflow-run-id` / `mlflow_run_id` | `LLM_BEHAVIOR_EVAL_MLFLOW_RUN_ID` |
| `--mlflow-artifact-path-subfolder` | `LLM_BEHAVIOR_EVAL_MLFLOW_ARTIFACT_SUBFOLDER` |
| Original LoRA reference | `LLM_BEHAVIOR_EVAL_MLFLOW_ADAPTER_REF` |
| Inferred checkpoint step override | `LLM_BEHAVIOR_EVAL_MLFLOW_STEP` |

Preserve the **original** adapter reference before resolving/downloading it for
Inspect's model provider. This hook neither downloads adapters nor changes model
arguments. Configuration is snapshotted at run start. Run precedence is explicit
ID, active MLflow run, inferred adapter run, then a new run. No child runs are
created. The active run must belong to the configured tracking store.

Inference accepts `mlflow://RUN/path`, `runs:/RUN/path`, and HTTP(S)
`TRACKING_ORIGIN/runs/RUN/path` (or legacy `/RUN`) only when scheme and authority
match the tracking URI. Arbitrary artifact-store URLs, UI fragment URLs, and
tracking servers mounted under path prefixes are not inferred; supply a run ID.
An exact `checkpoint-N` path segment supplies step N, otherwise step is zero.
An explicit nonnegative STEP overrides inference. One adapter/step applies to the
entire Inspect run; evaluate different checkpoints separately.

## Metrics and units

Metrics come directly from `EvalLog.results`, with no multiplication or inversion:

`TASK/IDENTITY/SCORER/REDUCER/METRIC`

TASK is the log's task name, made key-safe: characters other than letters,
digits, `_`, `.` and `-` become `_`, a component longer than 32 characters is
truncated, and any changed component gets a hash suffix, so
`llm_behavior_eval/bbq` becomes `llm_behavior_eval_bbq-<hash>`. IDENTITY is a
stable 16-character SHA-256 prefix over dataset specification, task arguments
and model identity. This separates task variants, datasets and models.
SCORER is Inspect's score name, REDUCER defaults to `default`. Unsafe/long name
components are sanitized and receive a hash suffix to avoid collisions. Raw task
arguments, adapter URLs and model parameters are not emitted as MLflow parameters.

| Old dashboard concept | Native metric mapping / unit |
| --- | --- |
| HaluEval, Med-Hallu accuracy | native `accuracy`, already 0–100 |
| Purple attack success | native `attack_success_rate`, already 0–100 |
| Bias accuracy/rates | native metric names, fractions |
| Safe/unsafe refusal rates | native metric names, fractions |
| CCPC censorship rate | native `censorship_rate`, fraction |
| Per-dataset sample count | `TASK/IDENTITY/num_samples_evaluated`, completed sample attempts (includes epochs) |
| `datasets_attached` | cumulative number of task logs attached, including failed tasks |

Migrate old `DATASET_metric` and unprefixed summary CSV dashboard keys to the
namespaced keys above. Scorer metric names remain Inspect's names. Nonfinite
metrics (empty denominators) are omitted. Repeated evaluation of the same
configuration at the same step creates metric history, as MLflow normally does.
The counter is best effort and requires sequential reporters on a shared run;
concurrent writers can lose counter increments. It is not a unique dataset count.

## Artifacts, lifecycle and failure constraints

Reporting runs at `on_run_end`, after Inspect finalizes its logs (verified against
installed Inspect 0.3.288). Local `.eval` files are uploaded under
`llm-behavior-eval/[SUBFOLDER/]INSPECT_RUN_ID/EVAL_ID/`. `timestamp` creates a UTC
subfolder. Remote Inspect log URIs are not uploaded by this hook. Inspect logs are
the authoritative record, including model outputs; apply Inspect's normal data
and credential hygiene to their contents.

There is deliberately no summary CSV parsing, unprefixed summary logging, or
whole-results-directory upload. inspect-brief owns CSV creation. If a pipeline
needs its CSV as an MLflow artifact, upload it **after** inspect-brief completes;
this hook never races another run-end hook's writer.

New MLflow runs are created during final reporting and terminated FINISHED or
FAILED according to evaluation/reporting outcome. Explicit, active and inferred
runs remain caller-owned: their status is never changed. No MLflow asynchronous
logging queue is used. Reporting executes in a subprocess with a wall-clock limit
(`LLM_BEHAVIOR_EVAL_MLFLOW_TIMEOUT`, default 30 seconds); timeout kills and reaps
it. MLflow errors are reduced to a generic warning without URLs, tokens, SDK
tracebacks or evaluation exception contents. Evaluation failure remains intact.
A timeout/crash can leave partial metrics or an owned run RUNNING; there is no
remote cleanup retry that could stall evaluation. Hard process termination before
Inspect emits run end cannot report. These are best-effort reporting semantics.
