# LLM-171 final runtime removal (0.2.0)

Integrated on `5cce36cba76a99271f234973853216cbb511a709`, reconciling candidate
`e642f167b5a3d6b2a272aa2e0924b1693451b08a` with all newer preprocessing,
HF, scorer metadata and qualification changes. The user-approved representative
LLM-169 gate is met; `scripts/parity/REPORT.md` remains authoritative and unchanged.
No all-model/backend parity or HF GPU score equality is claimed.

The entire `evaluation_utils` package is removed: evaluators, model engines,
loaders/configuration, tokenizer/chat-template execution, retry/batch/OOM logic,
generation-cache resume, adapter storage download helpers, and old output writers.
No native provider imports those download helpers. Providers now own model and
adapter configuration. Public `EvaluateFactory`, `EvaluationConfig`,
`DatasetConfig`, `PreprocessConfig`, `SamplingConfig`, `VllmConfig`, `MlflowConfig`,
`FreeText*Evaluator`, model-loading helpers and `DatasetType` imports are removed.
There is no forwarding package. The retired command prints a migration notice.

Retained semantics live in `common/prompts.py`, `common/preset_constants.py`, and
`refusal/contract.py`; native imports point there. Preset expansion/catalog and
native examples remain. The MLflow subprocess worker uses `MlflowClient` directly
and has no dependency on deleted utilities. Refusal scalar scores/dictionary
metrics and the hook exclusion from registry `__all__` are preserved.

## Installation and CI handoff

Supported Python is **3.12–3.13**, consistently for base and reporting extras.
Inspect requires at least 3.11; choosing 3.12 also supports inspect-brief without
silently omitting the requested extra. The base dependencies are Inspect, its
OpenAI grader transport, datasets and the tiny retired Typer CLI. There are no
hard Torch, Transformers, vLLM, Accelerate, bitsandbytes or NetworkX dependencies.
Transitive dependencies may still supply pandas/NumPy for datasets/reporting.

- CPU/API tasks: `uv pip install --python .venv/bin/python -e .`
- Reporting: append `[brief,mlflow]` (MLflow minimum remains 2.8).
- Exact offline response token counts: append `[tokenizers]`; provide the
  task's `response_tokenizer` path. Missing extra raises an actionable error.
- Native local HF provider: append `[hf]`.
- Native Linux vLLM provider: append `[vllm]`; the tested vLLM 0.30 and
  Transformers 5.20.2 ranges remain paired. Inspect owns runtime execution.

The dev group no longer installs GPU/cloud runtimes or optional reporting.
CI uses Python 3.12 and installs `.[brief,mlflow]` for the full test suite. The Gemma compatibility test skips without vLLM;
a separate provider environment can exercise it. Workflows, README, AGENTS, devcontainer and benchmark docs are reconciled.
Full tests install brief/mlflow explicitly; lint/type jobs do not install GPU extras.

## Migration and output contract

Use `list_dataset_presets()` and each preset's `inspect_tasks` mapping to expand
all variants without losing task arguments. Representative replacements:

| Old preset | Native task |
| --- | --- |
| `bias:gender`, `unbias:gender` | `llm_behavior_eval/bbq` with `-T bias_type=gender -T kind=bias` or `unbias` |
| `unqover:bias:gender` | `llm_behavior_eval/unqover -T bias_type=gender -T kind=bias` |
| `bloom:bias:age` | `llm_behavior_eval/bloom -T bias_type=age -T kind=bias` |
| `hallu` | `llm_behavior_eval/halueval` |
| `hallu-med` | `llm_behavior_eval/medhallu` |
| `prompt-injection` | `llm_behavior_eval/prompt_injection` |
| `refusal:xstest`, `refusal:orbench` | `llm_behavior_eval/xstest`, `llm_behavior_eval/orbench` |
| `chinese_censorship` | `llm_behavior_eval/ccpc` |

Run `inspect eval TASK --model PROVIDER/MODEL --model-role grader=PROVIDER/JUDGE`.
Use `-T max_samples=...` for benchmark caps; `--limit` further restricts samples.
Use `--max-tokens` for answer generation, native role configuration for independent
judges, `--max-retries`/`--max-connections` for execution, and `--log-dir` for logs.
Old batch-size/cache/model-output-directory flags have no forwarding aliases.
Family-specific provider/thinking constraints still apply; see INSPECT_MIGRATION.md.

Inspect logs replace `metrics.csv`, `responses.json`, `generations.jsonl`,
`summary_*.csv` and `run_config.json`. inspect-brief exports long-form rows, not
legacy wide CSV. Columns are `Created`, `Run ID`, `Benchmark`, `Metric`,
`Score`, and `Runtime (sec)`. Metric names carry units/direction and normalized
percentage metrics are scaled once. See the reporting section of INSPECT_MIGRATION.md and tests/test_inspect_brief.py for tested columns and units.
The optional MLflow hook reports finalized native metrics/log artifacts only;
no old CSV writer or engine-managed MLflow run is restored.

## Qualification and validation

Production task construction, preprocessing boundaries, HF validation, judge
metadata and original-input/target logging are retained from the qualified parent.
The only preprocessing change is an actionable missing-tokenizer-extra error.
Constants moved without a replacement inference engine. Exact Purple ASR
arithmetic is preserved, including its floating-point operation order.

New golden fixtures were captured by invoking the actual legacy functions from
an archived `5cce36cba76a99271f234973853216cbb511a709` checkout before migrating
these tests. See `tests/golden/README.md` for origins and tokenizer identities.
No legacy implementation is copied into tests. Existing candidate golden data
retains its original baseline citation.

Historical legacy phases now fail with an explicit frozen-checkout instruction.
Their reproducible environment is the pre-removal checkout, not this package.
The independent recorded-judge reader remains usable without the engine;
comparison/report/settings/identity tests remain in the core suite. Committed
qualification reports, evidence and hashes are unchanged.

Validation uses dedicated Python 3.13 environments at
`results/validation-storage/llm171/envs/{reporting,base}` on the shared project
volume. The reporting environment was installed with locked `brief,mlflow` extras
and the dev group; both clean environments exclude Torch/Transformers/vLLM.
Logs are under `results/validation-storage/llm171/tmp/`. Root disk exhaustion was
avoided using shared storage; no unrelated caches or sibling environments were
deleted. Full-suite temporary data uses task-owned `/dev/shm/llm171-validation`.

- Full final suite in the clean locked Python 3.13 reporting environment:
  **431 passed, 18 skipped, 2 upstream deprecation warnings** in 127.57 seconds.
  Skips are optional tokenizer/provider coverage; core/reporting/bundle tests run.
  `final-suite.log` and `final-suite.exit` (0) record completion.
- Optional pinned Gemma3/Qwen3 tokenizer and HF CPU boundary suite: **51 passed**
  in 93.95 seconds with Transformers 5.20.2 / Inspect 0.3.288 in the existing
  heavy environment, without modifying it.
- Base-only editable and installed-wheel discovery/mock evaluation: pass; all
  nine benchmarks discovered, no model runtimes installed/imported.
- Wheel inspection: no `evaluation_utils` package; `target_metrics.json` included.
- Full Ruff format/lint, Pyrefly, lock validation and dependency checks: pass.
  Pyrefly's five narrowly scoped missing-import suppressions concern optional
  Transformers imports in tokenizer/report tooling, not package runtime errors.

Local wheel building creates `.egg-info` that can shadow Inspect's editable
metadata when running from the checkout. The generated directory was moved out
before final validation; the initial affected run is not counted as passing.

Remaining limits: representative qualification does not establish all-model or
all-backend parity, HF GPU score equality, or strict GPT-OSS thinking-off support.
Official parser arguments remain user/provider configuration. No GPU/VM actions,
PR, push, release, tag or external issue/message updates were performed.

The platform remains pinned to 0.1.12 and must not be upgraded independently. Its
worker needs native Inspect migration, its CCPC bundle builder must import
`ccpc.contract`, and release manifest/version/digest pairing plus local answer
tokenizer paths must be coordinated. See INSPECT_MIGRATION.md. The preserved
catalog and offline bundle formats do not imply drop-in worker compatibility.
The new bundle contract suite covers raw HF snapshot Parquet loading for all
nine benchmarks and rejects CCPC tampering outside the selected sample cap.

Parent integration validation: **23 bundle, discovery, and removal tests passed**.
The tampering test uses a separate immutable snapshot path so HF's metadata-keyed
dataset cache cannot mask an in-place rewrite on a filesystem with coarse
modification timestamps. Production loaders and qualified scoring are unchanged.

CI follow-up: configure pytest's checkout import path and maintained test directory
so console-script invocation can import examples and parity tooling without
collecting ignored experiment copies. Offline smoke outputs include deterministic
mock usage; Inspect's default mock usage estimator otherwise downloads a tiktoken
vocabulary on a cold machine. The socket network guard remains enabled. Existing
test typing corrections are folded into their introducing PR layers. No production
benchmark or qualification evidence is changed by these CI fixes.
The CLI discovery regression also limits `inspect list tasks` to the package
directory, avoiding unrelated Python files inside local validation environments.
The corrected complete suite also passes in a clean locked Python 3.12 reporting
environment: **431 passed, 18 optional skips**, with Ruff, Pyrefly and lock checks
passing. This run uses the console `pytest` entry point without `PYTHONPATH`;
`results/pr-stack/ci312-full-final.log` records the local result.

A fresh CI build exposed an additional import-order regression: prepending the
checkout made setuptools' generated source `.egg-info` shadow the installed
`.dist-info`, hiding the editable-install metadata Inspect uses for task names.
The test configuration now appends the checkout after installed packages instead.
This preserves imports of examples and parity scripts without overriding package
metadata. Reproduced with source `.egg-info` present: the original BBQ smoke
fails; the corrected discovery/CLI/parity subset passes all **46 tests**.
The complete rerun passed 429 tests and skipped 18; two UNQOVER cases hit the
VM's full root disk (`sqlite3.OperationalError: database or disk is full`).
Redirecting Inspect data/config/cache to the project volume made the complete
UNQOVER suite pass (**24 tests**). Ruff, Pyrefly, formatting and lock checks pass.
Logs: `results/pr-stack/egg-shadow-{full,unqover}.log`. No task/scorer change was
needed for the storage failures, and source `.egg-info` remained present during
these checks.
