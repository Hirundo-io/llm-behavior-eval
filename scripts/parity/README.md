# LLM-169 parity: procedure, tools and evidence

Parity compares the legacy engine (`llm_behavior_eval/evaluation_utils/`, removed
by LLM-171 in #209) with the native Inspect tasks on a fixed setup. Under the
self-contained design (LLM-155 "Design rules"), benchmark code sets **no**
inference settings for either role. Parity settings live only in each
benchmark's `parity.yaml`; the answer model's provider args go on the command
line (`-M`); and each benchmark has an **opt-in** legacy text preprocessing,
turned on with `-T legacy_tokenizer`.

## Frozen pre-removal checkout

Version 0.2.0 contains no legacy engine. Everything that runs it needs a separate
checkout of the last commit with the engine,
**`d8c80154c1e6ae90e2896811685d7c1a9bf89e68`** (the LLM-169 qualification head,
#208; `FROZEN_COMMIT` in [historical.py](historical.py)), with that checkout's own
locked environment:

```sh
git worktree add ../llm-behavior-eval-legacy d8c80154c1e6ae90e2896811685d7c1a9bf89e68
cd ../llm-behavior-eval-legacy
uv venv --python 3.12 .venv && uv sync --locked --extra brief --extra mlflow --extra vllm
```

Run there, not in this checkout:

* the `legacy` phase of `qualify_matrix.py` (and `run_matrix.py --phases ... legacy`);
* `reuse_legacy.py`;
* `python -m scripts.parity.removal_baseline`, which recaptures the frozen legacy
  expectations in `tests/golden/` (copy `scripts/parity/historical.py` and
  `scripts/parity/removal_baseline/` into that checkout first; it refuses a
  modified `llm_behavior_eval/` and records the checkout's commit as provenance).

In a 0.2.0 checkout these exit with a message naming this commit. Do not copy
legacy modules into the installed package. Everything else here (native phases,
`compare.py`, the reports, settings and identity audits, recorded-judgment checks
and `replay_redesign.py`) runs in the current checkout. The committed evidence and
its numbers are unchanged by the removal.

## Evidence status

The representative GPU qualification in [REPORT.md](REPORT.md) (31 configurations
x 100 samples, 3,100 paired rows, 7,448 judge requests per side, all exact) ran
the **pre-redesign implementation** (source `f35fb814`, plus the Purple fix
`b9a5b28`). That implementation set the same inference settings in code and
wrapped every task in an always-on preprocessing step. The evidence files and
numbers are unchanged and remain a record of that run.

What still holds for the current code, checked on CPU from the raw archive
(`production-evidence-archive.json`, SHA-256 `c9791b0d...`):

* **Requests, settings and scores** ([redesign-replay.json](redesign-replay.json)).
  `replay_redesign` re-runs all 31 recorded cases through the current tasks,
  configured only by `parity.yaml` and `-T legacy_tokenizer`, with mock models
  that return the recorded response for a byte-identical recorded request. Every
  answer and judge request (messages, order and settings, ignoring only the
  runner's capture fields), every legacy prompt token sequence (equal to the IDs
  the vLLM server actually received), every legacy judge text, per-sample score
  and recorded metric is identical: 31/31 cases, 3,100 samples, 7,448 judge
  requests. The one metric not compared is CCPC's old `incomplete_response_rate`,
  which the CCPC package no longer reports.
* **Purple ASR arithmetic** ([redesign-purple-metric-replay.json](redesign-purple-metric-replay.json)).
  The current metric returns `48.684210526315795` on all 251 recorded legacy
  rows, equal to the legacy value bit for bit.

What a GPU re-run with `parity.yaml` must still confirm: that real models given
these identical requests produce the recorded outputs, that is, that the vLLM
server applies `parity.yaml`'s `extra_body` (`add_special_tokens`,
`truncate_prompt_tokens`, `truncation_side`, `chat_template_kwargs`) exactly as
the code-set values were applied, so the observed prompt token IDs, raw answers
and judge outputs match again. No GPU is available on the machine that produced
this layer and no remote GPU route is documented, so the re-run is pending.

| Benchmark | GPU evidence (pre-redesign) | CPU replay (current code) | Re-run with `parity.yaml` |
| --- | --- | --- | --- |
| BBQ | 12 configs, 1,200 rows, exact | exact | needed |
| UNQOVER | 4 configs, 400 rows, exact | exact | needed |
| BLOOM | 9 configs, 900 rows, exact | exact | needed |
| HaluEval | 1 config, 100 rows, exact; earlier 500-row eager run exact | exact | needed |
| Med-Hallu | 1 config, 100 rows, exact | exact | needed |
| Purple Llama | 1 config, 100 rows, exact (fixed-source rescore) | exact; 251-row ASR replay exact | needed |
| CCPC | 1 config, 100 rows, exact (Gemma 4 26B-A4B judge) | exact | needed |
| XSTest | 1 config, 100 rows, exact | exact | needed |
| OR-Bench | 1 config, 100 rows, exact | exact | needed |

Not covered by any evidence: the shipped `parity.yaml` graders use Inspect's HF
provider, but the qualification served both roles with vLLM; HF GPU score
equality is unqualified (CPU tests check only that the documented HF `-M` flags
feed the legacy token IDs). Full splits beyond the representative subsets, and
judges other than Gemma 3 12B (CCPC: Gemma 4 26B-A4B), are not qualified.
`google/gemma-4-12B-it`, the recommended judge, is not parity-qualified.

## How a parity run is configured

1. `--run-config llm_behavior_eval/<benchmark>/parity.yaml`: greedy decoding,
   the legacy token budgets, seed 42, the legacy thinking switch for both roles,
   the 500-row cap, and for vLLM/SGLang/OpenAI-compatible servers the legacy
   prompt tokenization (`add_special_tokens: true`; answer prompts
   `truncate_prompt_tokens: 1024`, `truncation_side: right`). These values are
   exactly the request settings recorded in every qualified run.
2. `-M` for the model under test, because the file does not name it. On HF:
   `-M do_sample=false -M enable_thinking=<legacy>` and
   `-M tokenizer_call_args='{truncation: true, max_length: 1024}'` (Inspect's HF
   provider ignores `extra_body`). On vLLM the file already carries these.
3. `-T legacy_tokenizer=<answer snapshot>` (plus
   `-T legacy_tokenizer_revision=<commit>` for a cached repository). This is the
   benchmark's opt-in legacy text preprocessing: the judge sees what the legacy
   judge saw (see each benchmark README). It is a task argument because Inspect's
   run config sets task arguments only inside a `task:` block that also names
   the task, which `parity.yaml` must not do (checked against inspect_ai 0.3.288,
   `RunConfigInput`/`TaskInput`).

Do not pass `--model-role grader=...` with a parity config: Inspect replaces the
whole grader entry, including its `config`. To serve the judge differently,
copy `parity.yaml` and change only the grader's `model` and `args`.

### Per-benchmark parity commands (HF, as shipped)

`$ANSWER` is the answer model's local snapshot directory.

```sh
inspect eval llm_behavior_eval/bbq -T bias_type=gender -T kind=bias -T legacy_tokenizer="$ANSWER" --model hf/local -M model_path="$ANSWER" -M do_sample=false -M enable_thinking=false -M tokenizer_call_args='{truncation: true, max_length: 1024}' --run-config llm_behavior_eval/bbq/parity.yaml
inspect eval llm_behavior_eval/unqover -T bias_type=gender -T legacy_tokenizer="$ANSWER" --model hf/local -M model_path="$ANSWER" -M do_sample=false -M enable_thinking=false -M tokenizer_call_args='{truncation: true, max_length: 1024}' --run-config llm_behavior_eval/unqover/parity.yaml
inspect eval llm_behavior_eval/bloom -T bias_type=gender -T kind=bias -T legacy_tokenizer="$ANSWER" --model hf/local -M model_path="$ANSWER" -M do_sample=false -M enable_thinking=false -M tokenizer_call_args='{truncation: true, max_length: 1024}' --run-config llm_behavior_eval/bloom/parity.yaml
inspect eval llm_behavior_eval/halueval -T legacy_tokenizer="$ANSWER" --model hf/local -M model_path="$ANSWER" -M do_sample=false -M enable_thinking=false -M tokenizer_call_args='{truncation: true, max_length: 1024}' --run-config llm_behavior_eval/halueval/parity.yaml
inspect eval llm_behavior_eval/medhallu -T legacy_tokenizer="$ANSWER" --model hf/local -M model_path="$ANSWER" -M do_sample=false -M enable_thinking=false -M tokenizer_call_args='{truncation: true, max_length: 1024}' --run-config llm_behavior_eval/medhallu/parity.yaml
inspect eval llm_behavior_eval/prompt_injection -T legacy_tokenizer="$ANSWER" --model hf/local -M model_path="$ANSWER" -M do_sample=false -M enable_thinking=false -M tokenizer_call_args='{truncation: true, max_length: 1024}' --run-config llm_behavior_eval/prompt_injection/parity.yaml
inspect eval llm_behavior_eval/ccpc -T legacy_tokenizer="$ANSWER" --model hf/local -M model_path="$ANSWER" -M do_sample=false -M enable_thinking=false -M tokenizer_call_args='{truncation: true, max_length: 1024}' --run-config llm_behavior_eval/ccpc/parity.yaml
inspect eval llm_behavior_eval/xstest -T legacy_tokenizer="$ANSWER" --model hf/local -M model_path="$ANSWER" -M do_sample=false -M enable_thinking=true -M tokenizer_call_args='{truncation: true, max_length: 1024}' --run-config llm_behavior_eval/xstest/parity.yaml
inspect eval llm_behavior_eval/orbench -T legacy_tokenizer="$ANSWER" --model hf/local -M model_path="$ANSWER" -M do_sample=false -M enable_thinking=true -M tokenizer_call_args='{truncation: true, max_length: 1024}' --run-config llm_behavior_eval/orbench/parity.yaml
```

Other BBQ/UNQOVER/BLOOM splits change only `bias_type`/`kind` (the 31 required
configurations are listed in `pending.json`).

### vLLM variant (the backend the qualification used)

Copy the benchmark's `parity.yaml`, set the grader's `model` to
`vllm/<judge snapshot>` and drop its HF `args`; keep its `config`. Then:

```sh
inspect eval llm_behavior_eval/halueval -T legacy_tokenizer="$ANSWER" \
  --model "vllm/$ANSWER" -M max_model_len=16384 -M dtype=bfloat16 \
  --run-config parity-vllm.yaml
```

The qualification also requested `return_token_ids` (prompt token evidence) and,
in batches, a per-case `cache_salt`; neither is a legacy setting. The runners
below add them and do everything else from `parity.yaml`.

## Running the qualification matrix (requires GPUs)

Freeze this checkout and the frozen pre-removal checkout (above), which has both
the legacy engine and the native tasks; run the matrix there when it includes the
`legacy` phase. Provision the pinned models with
`provision_models.py`, then for each case (or all 31) run, in separate
processes:

```sh
python -m scripts.parity.run_matrix --root "$ROOT" --answer "$ANSWER" \
  --judge "$JUDGE" --ccpc-judge "$CCPC_JUDGE" --provision provision.json \
  --samples 100 --phases prepare legacy answer judge report \
  --group-native --isolate-cache --ordered --compiled \
  --ccpc-judge-content-format string
python -m scripts.parity.sampled_report report --root "$ROOT" \
  --manifest sampled-manifest.json --output sampled-report.json
```

Native phases (`qualify_matrix`, `qualify_native_batch`, `qualify_native`) build
both roles from `parity.yaml` (`parity_models.py`), turn on
`legacy_tokenizer=<answer snapshot>`, select rows with Inspect's `limit`, and
refuse a judge snapshot that is not the repository `parity.yaml` names (Gemma 3
12B; CCPC Gemma 4 26B-A4B). vLLM server arguments (`max_model_len`, `dtype`,
`seed`, `max_num_seqs`, ...) are the runners' `-M` equivalent. `report_matrix`
compares the strict evidence, `settings.py` audits every recorded request against
`parity.yaml`, and `replay_redesign` reproduces the CPU replay above:

```sh
python -m scripts.parity.replay_redesign --evidence EXTRACTED_ARCHIVE \
  --tokenizer "$ANSWER" --output redesign-replay.json
```

## Tools

| Script | Purpose |
| --- | --- |
| `compare.py` | Strict, offline evidence comparator (protocol below). |
| `parity_models.py` | Both roles' settings from `parity.yaml`, capture fields, judge identity check. |
| `qualify_matrix.py` | Per-case prepare / legacy / answer / judge phases. |
| `qualify_native_batch.py` | Native answer/judge phases across cases on shared Inspect-owned servers. |
| `qualify_native.py` | Native answer/judge wrapper for one task with JSON task/provider args. |
| `run_matrix.py` | Runs the phases in isolated processes and keeps their logs. |
| `report_matrix.py`, `matrix_report.py`, `sampled_report.py` | Per-case strict reports and the 500-row and sampled summaries. |
| `settings.py` | Audits retained legacy configs and every native request against `parity.yaml`. |
| `prepare_frozen.py`, `prepare_edges.py` | Extend a frozen selection; synthetic long-boundary probes. |
| `reuse_legacy.py`, `rescore_followup.py` | Replay recorded legacy evidence; isolated scorer follow-ups. |
| `replay_redesign.py`, `replay_purple_metric.py` | CPU replays of recorded evidence through the current code. |
| `qualify_reasoning.py` | Bounded GPT-OSS parser/scorer integration probe (XSTest `parity.yaml` settings plus operator reasoning args). |
| `capture.py`, `ordered.py`, `tokenizer_probe.py`, `provision_models.py` | Response capture, ordered rescoring, an independent tokenizer oracle, model provisioning. |

Removed with this redesign, because they targeted deleted code paths and were
superseded by the matrix runners: `live_generation.py`, `tokenizer_audit.py` and
`generation_report.py` (the early eight-row generation diagnostic), and
`qualify_hallu.py` and `report_hallu.py` (the HaluEval-only experiment). Their
evidence (`gpu-summary.json`, `gemma-small-slice.json`,
`production-halueval-500.json`) is kept; the scripts remain in the history of
`t3/inspect-migration`.

CPU tests: `tests/test_parity_*.py`, `tests/test_inspect_preprocessing.py` and
`tests/test_inspect_hf_preprocessing.py`. The last two use a tiny offline
tokenizer and, when present, the pinned Gemma 3
(`093f9f388b31de276ce2de164bdc2081324b9767`, under `results/tokenizers/gemma3` or
`LLM169_TOKENIZER`) and Qwen3 (`c1899de289a04d12100db370d81485cdf75e47ca`, under
`results/tokenizers/qwen3` or `LLM169_QWEN3_TOKENIZER`) tokenizers; the
tokenizer JSON hashes are checked and missing fixtures skip.

## Evidence protocol v1 (`compare.py`)

Run from the repository root with `.venv/bin/python scripts/parity/compare.py manifest.json --output report.json`.
Exit 2 means incomplete evidence; exit 0 means comparisons were computed, **not acceptance**.
No tolerance is approved. In particular the issue's proposed 1 pp / one stderr
rule is not a statistical test or owner sign-off. No legacy engine is invoked.

A manifest has `expected_cases` (all split names) and `cases`. The committed
`pending.json` enumerates all 31 required configurations and intentionally has
no fabricated measurements. Each case specifies:

- `name`, `family`, `expected_samples`, `responses`, `identity`, `eval`, `metrics_csv`;
  paths resolve relative to the manifest. `eval` accepts native `.eval` or JSON.
- `runs.legacy` and `runs.native`, each containing nonempty `model_revision`,
  `judge_revision`, `tokenizer_revision`, `dataset_revision`, `backend`, `packages`,
  `config`. Use resolved immutable revisions, not branch names. Config must record
  seed, cap, template hash/kwargs, thinking, all token budgets, quantization,
  dtype, batch settings, device and generation arguments. Provenance differences
  are reported, never silently treated as equivalent. Values are attestations;
  the script cannot independently verify loaded weights.
- `fields`: mappings for `verdict`, `answer`, `judge_question`, `finish_reason`,
  `rendered_prompt`, plus any target/stereotype/token-ID fields. Each mapping has
  `legacy` and `native` dot-separated dictionary paths. Example: legacy `grade`
  maps to native `scores.halueval_scorer.metadata.grade`. Read the actual
  scorer keys in the log; do not assume registry prefixes. Missing fields remain
  incomplete even when aggregate metrics match. Explicit null skipped verdicts
  are allowed; absent verdict fields are not.
- `metrics`: explicit mappings with `name`, `unit`, `legacy_column`, `scale`,
  optional `offset`, `native_scorer`, `native_metric`. Conversion is
  `legacy * scale + offset`; the output delta is native minus converted legacy.
  HaluEval/Med-Hallu accuracy and Purple Llama ASR are percent; other rates are
  fractional, counts are `count`, stereotype bias is `index`. Inspect scorer and
  metric keys must match exactly. Legacy displayed accuracy CSV already uses
  percent: use scale 1 for hallu, 0.01 for bias accuracy; for Purple ASR from
  displayed accuracy use scale -1, offset 100. Do not use these conversions on
  internal fractional metrics. Rounded legacy CSV limits precision.
- Optional `generations` points to the original batched `generations.jsonl`;
  `generation_fields` lists batch columns to flatten with strict equal lengths.
  Access them as `generation.finish_reasons`, `generation.input_texts`, etc.
  Do not include absent stereotype arrays. The sidecar must also contain
  `generations_sha256`. Counts must match responses exactly.

Identity sidecar: `{"responses_sha256": "...", "ids": ["dataset:source-row", ...]}`.
Generate IDs at selection time from the pinned source snapshot, **before** seed-42
HF shuffle and cap; CCPC uses benchmark_id and fixed source order. Bind both raw
files by SHA256. Legacy row order alone is not identity; never reconstruct IDs
by matching generated text or by assigning native IDs to legacy positions without
independent source-selection evidence. A wrong but attested sidecar cannot be
detected by this tool: retain the source snapshot/order manifest for audit.

## Qualification procedure notes

The notes below describe how the recorded evidence was produced; the commands
still apply, with native phases now configured by `parity.yaml`.

### Extending an existing frozen selection

The historical full-source supplement recipe below is optional. The current
owner-approved migration gate uses representative subsets, described next.
Prepare any additional selection from exact Parquet bytes rather than resolving
dataset revisions again:

```bash
python -m scripts.parity.prepare_frozen \
  --primary-root results/matrix500 --root results/full-small --samples 1000 \
  --cases bbq:religion:bias bbq:religion:unbias bbq:physical:bias bbq:physical:unbias
```

This verifies the original source hashes and seed-42 order, checks the extended
selection through the production task, and records preparation provenance in a
fresh root. It performs no inference. Run `run_matrix` on that root with the same
`--samples`, cases, model arguments and execution settings, selecting phases
`legacy answer judge report` (omit `prepare`). Preserve the original reports.
Supplement reports describe their actual row counts; `matrix_report` retains its
separate 500-row acceptance gate and must not substitute a larger run implicitly.

`report_matrix` also checks retained run settings and every native model-event
configuration against the benchmark's `parity.yaml`: budgets, greedy decoding,
seed, concurrency, thinking flags, special tokens and the answer-only prompt cap. To audit existing artifacts
without rewriting their reports, run `python -m scripts.parity.settings CASE_ROOT
[CASE_ROOT ...] --output settings-audit.json`. The audit binds its checks to
source, retained-configuration and scored-log hashes. It reports zero judge
requests explicitly; settings validation alone does not establish score parity.

### Representative paired subsets

The reduced LLM-169 scope uses deterministic first-100 selections for all 31
configurations in `pending.json`. `prepare_frozen --samples 100` creates fresh
case directories with unchanged source bytes/revisions and independently checks
IDs against the production task. It preserves CCPC's original published order
and validated 500-row source fingerprint. Selection never depends on outputs.

When complete real legacy evidence already exists, `reuse_legacy --primary-root
OLD --root SAMPLED --cases CASE...` replays the selected actual generation records
through the retained grader's parsing and aggregation methods. Every replayed
judge prompt must encode to the original observed token IDs, and every resulting
response must equal the original prefix. This performs no model inference and
does not create simulated predictions. The original run configuration retains its
original cap; `legacy-reuse-provenance.json` records the distinct comparison
subset, source hashes and recomputed metrics. Cases without real evidence require
fresh legacy inference at the sampled cap.

Before native inference, pin the separate manifest:

```bash
python -m scripts.parity.sampled_report pin \
  --root results/sampled --manifest results/sampled-manifest.json
```

Run the existing `run_matrix` with `--samples 100 --phases answer judge report
--group-native --isolate-cache --ordered --compiled`, the pinned model/provision
arguments, and `--ccpc-judge-content-format string`. Answer and judge processes
run sequentially. Summarize with `sampled_report report --root results/sampled
--manifest results/sampled-manifest.json --output results/sampled-report.json`.
The sampled gate requires exact fields, metrics and observed judge outputs/order;
the historical `matrix_report` 500-row gate remains separate. Neither gate accepts
CPU/mock inference as real paired evidence. Sampled results establish scoped
implementation parity, not full-dataset equality or a statistical accuracy bound.

`rescore_followup` preserves a baseline report, verifies frozen production source
hashes, copies actual retained artifacts byte-for-byte, and reuses the immutable
native answer log for real scoring under an explicitly isolated fix. Its CLI
requires primary/follow-up roots, both source directories, source provenance and
model provision metadata. Pass an explicit case-root mapping to `sampled_report`
to select that follow-up; source IDs and generation identity must remain equal.

### Bounded upstream reasoning integration

Reasoning parser choice belongs to user/provider configuration. Supply official
parser arguments through Inspect (for example `-M reasoning_parser=qwen3`), or
use a verified upstream default. The package has no model-to-parser mapping or
custom model-output parser. Benchmark judge verdict parsing remains task-owned.
See [Inspect reasoning configuration](https://inspect.aisi.org.uk/reasoning#vllm--sglang)
and [vLLM 0.30 reasoning outputs](https://docs.vllm.ai/en/v0.30.0/features/reasoning_outputs/).

`qualify_reasoning` takes explicit `--reasoning-parser`, `--reasoning-effort` and
`--rejected-effort` arguments plus cached answer/judge paths, a local XSTest dataset
and fresh output root. Run `answer`, `judge`, then `report` in separate processes.
It selects one safe and one unsafe source case for production generation, adds
an explicitly labelled eight-token truncation diagnostic, and checks a rejected
effort request. Real raw responses, Inspect content blocks, stop reasons and the
exact text seen by the production scorer are retained. This bounded probe does
not claim old/new generation equality or certify the complete benchmark matrix
for another model.

Pinned vLLM 0.30 defaults GPT-OSS to `openai_gptoss`; its Harmony effort values are
`low`, `medium`, and `high`. The cached revision
`6cee5e81ee83917806bbde320786a8fb61efebee` ignores `enable_thinking` in its template.
Reasoning extraction and suppression are separate: **low is not off**. Strict
thinking-off benchmark modes remain unsupported for that combination; use a
model/provider with a verified off mode. Live integration results belong in the
qualification report, not an inferred claim from parser availability or CPU tests.
