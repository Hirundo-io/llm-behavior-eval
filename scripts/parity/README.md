# LLM-169 evidence protocol v1

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
  maps to native `scores.hallucination_scorer.metadata.grade`. Read the actual
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

## Reproducible qualification recipe (requires resources)

1. Freeze this source commit and the legacy source commit in separate existing
   checkouts/environments. The retired CLI must not be restored. Use the retained
   legacy evaluator API/factory from its frozen revision. Run its existing
   generation and grading calls; do not write another evaluation engine.
2. Provision the designated Gemma 3 4B answer, Gemma 3 12B independent judge,
   and CCPC Gemma 4 26B A4B judge. The follow-up below pins and downloads these
   models; no additional model-selection approval is required. Record actual
   weight/tokenizer revisions and hashes.
3. Freeze all 31 dataset snapshots, exact source order and stable IDs. Use
   seed 42 and cap 500 first, then full primary splits. Retain source hashes.
4. Run each native task via Inspect with independent `grader` role and the same
   backend as legacy, greedy settings and family budgets. Bias/hallu/injection:
   128/32, thinking off; refusal: 2048/128, answer thinking on/judge off;
   CCPC: 8192/4096, both off. HF requires explicit `enable_thinking` and
   `do_sample=False` per role; vLLM template kwargs must match each deployment.
   Use benchmark README task parameters for local dataset snapshots.
5. Capture legacy outputs and native `.eval` logs. Preserve provider-rendered
   prompts/token IDs at the boundary for a generation subset. A native message
   list is not a rendered prompt. Legacy decoded `input_texts` is also not the
   original token IDs. When those are absent, retain an incomplete generation
   comparison rather than inventing them. `tokenizer_probe.legacy_text_probe`
   reconstructs padded/truncated tokens from an already-rendered prompt using
   the exact local tokenizer; record template/revision provenance separately.
6. Fill field/metric mappings and hashed identity sidecars. Compare, review every
   mismatch, publish raw logs to the owner's selected shared artifact location,
   and obtain explicit per-divergence/tolerance acceptance before LLM-171.

This is a protocol, not a turnkey live runner: models, snapshots, acceptance and
boundary evidence are still missing. No network inference or download is needed
for `pytest tests/test_parity_evidence.py` with installed test dependencies.

### Concrete HaluEval invocation after provisioning

Run these in **separate processes**, on an approved idle GPU, with local pinned
paths. `DATASET_SOURCE` must be an HF `load_dataset`-compatible local source
(e.g. a directory with train.parquet), not a `save_to_disk` directory: the retained
legacy CustomDataset uses `load_dataset`. Native accepts both formats. Set
`HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` to prevent accidental downloads.
Capture source IDs/hashes before running; the commands below do not manufacture
the absent identity sidecar or provider-boundary token evidence.

```python
# Legacy process, retained API at the frozen source commit
import os
from pathlib import Path
from llm_behavior_eval.evaluation_utils.evaluate_factory import EvaluateFactory
from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig
from llm_behavior_eval.evaluation_utils.dataset_config import DatasetConfig
from llm_behavior_eval.evaluation_utils.enums import DatasetType

EvaluateFactory.create_evaluator(
    EvaluationConfig(
        model_path_or_repo_id=os.environ["ANSWER_SNAPSHOT"],
        judge_path_or_repo_id=os.environ["JUDGE_SNAPSHOT"],
        inference_engine="transformers",
        sample=False,
        sample_judge=False,
        enable_thinking=False,
        use_4bit=False,
        use_4bit_judge=False,
        max_samples=500,
        batch_size=1,
        judge_batch_size=1,
        max_answer_tokens=128,
        max_judge_tokens=32,
        results_dir=Path("results/parity/legacy"),
    ),
    DatasetConfig(
        file_path=os.environ["DATASET_SOURCE"],
        dataset_id="hirundo-io/halueval",
        dataset_type=DatasetType.BIAS,
        seed=42,
    ),
).evaluate()
```

```python
# Native process, same local weights and dataset; HF backend on both sides
import os
from inspect_ai import eval
from inspect_ai.model import get_model
from llm_behavior_eval.halueval.task import halueval

answer = get_model(
    "hf/local-answer",
    model_path=os.environ["ANSWER_SNAPSHOT"],
    tokenizer_path=os.environ["ANSWER_SNAPSHOT"],
    enable_thinking=False,
    do_sample=False,
)
grader = get_model(
    "hf/local-judge",
    model_path=os.environ["JUDGE_SNAPSHOT"],
    tokenizer_path=os.environ["JUDGE_SNAPSHOT"],
    enable_thinking=False,
    do_sample=False,
)
eval(
    halueval(dataset_path=os.environ["DATASET_SOURCE"], max_samples=500),
    model=answer,
    model_roles={"grader": grader},
    max_connections=1,
    log_dir="results/parity/native",
)
```

These are unexecuted resource-dependent recipes. Inspect's concurrency setting
is not proof of identical backend batching. Record actual dtype/device/batching
and token boundary evidence before accepting this comparison. Use the nine task
READMEs for remaining family parameters; use the manifest's 31 configurations,
then repeat without the cap. This example does not resolve prompt parity gaps.

### Bounded cached-model GPU diagnostic

`live_generation.py` invokes the retained evaluator's `generate()` and the native
Purple Llama task with scoring disabled. An explicitly unused mock grader role
satisfies task setup only; no mock verdict or benchmark score is produced. Run
`prepare`, `legacy`, and `native` as separate processes. The native model provider
owns its server lifecycle. The model and Arrow file must already exist; set
`HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`. The script caps diagnostics at 32 rows
(default 8), uses seed-42 source selection and vLLM's legacy engine seed 0,
greedy 128-token answers, max model length 8192, one sequence, eager execution,
and GPU memory utilization 0.5. This setup deliberately exposes the unchanged
native-versus-legacy input differences. It is not a replacement scoring engine.

```sh
python scripts/parity/live_generation.py prepare --root results/parity-gpu \
  --model "$ANSWER_SNAPSHOT" --arrow "$CACHED_ARROW" --samples 8
python scripts/parity/live_generation.py legacy --root results/parity-gpu \
  --model "$ANSWER_SNAPSHOT" --samples 8
python scripts/parity/live_generation.py native --root results/parity-gpu \
  --model "$ANSWER_SNAPSHOT" --samples 8
python scripts/parity/tokenizer_audit.py --root results/parity-gpu \
  --model "$ANSWER_SNAPSHOT"
python scripts/parity/generation_report.py results/parity-gpu \
  --output results/parity-gpu/generation-report.json
```

`tokenizer_audit.py` compares the actual local tokenizer's legacy round trips
with expected native templates, not captured server token IDs. The generation
report includes raw answers and structured native content to expose reasoning
separation; exact-completion disagreement need not mean different underlying
tokens. No judge agreement, ASR or other benchmark metric is inferred. Preserve
failed startup logs and all `.eval` files. A supported runtime, approved model
and independent real judge are still required for the main acceptance protocol.

### Pinned designated-model follow-up

`provision_models.py --root /absolute/task/root` downloads the three designated
Gemma snapshots at immutable revisions, reuses its cache, and records installed
runtime versions without recording authentication. Install this checkout with
`uv sync --locked --python 3.12` in a new task directory first. Check disk space
before provisioning; the three snapshots occupy about 79 GiB.

`qualify_hallu.py` is a small live HaluEval compatibility experiment using the
retained evaluator API and public Inspect eval/score APIs. Invoke as a module
from the repository root, with absolute pinned local snapshot paths:

```sh
python -m scripts.parity.qualify_hallu legacy --root "$EVIDENCE" --answer "$ANSWER_SNAPSHOT" --judge "$JUDGE_SNAPSHOT" --dataset "$DATASET_SOURCE" --samples 2
python -m scripts.parity.qualify_hallu answer --root "$EVIDENCE" --answer "$ANSWER_SNAPSHOT" --judge "$JUDGE_SNAPSHOT" --dataset "$DATASET_SOURCE" --samples 2
python -m scripts.parity.qualify_hallu judge --root "$EVIDENCE" --answer "$ANSWER_SNAPSHOT" --judge "$JUDGE_SNAPSHOT" --dataset "$DATASET_SOURCE" --samples 2
```

Run the `prepare` phase first to download the pinned HaluEval snapshot, export
Parquet, and bind pre-shuffle source IDs to its SHA256. Use the same arguments.
Do not reuse a populated legacy output directory for a captured-input rerun: cached
generations contain no new API observations. Preserve old output directories.

Wait for each process to exit and verify GPU release before the next. Native
answer and judge lifecycles belong to Inspect. The answer process retains the
original log. The judge phase adapts a copy's question and target using the
legacy tokenizer oracle before public rescoring; this is an explicit experiment,
not yet a production dataset migration. It requests actual vLLM prompt token IDs
in model events, adds tokenizer-default special tokens, and applies right-side
1024 truncation only to answer input. Legacy judge input has no such cap.
`return_token_ids` response evidence must be inspected before asserting parity;
the oracle alone is not sufficient. This runner currently covers HaluEval only.

`python -m scripts.parity.report_hallu --root "$EVIDENCE" --source "$EVIDENCE/source.json" --provision "$TASK_ROOT/provision.json"`
builds the manifest and invokes the strict comparator. The legacy runner observes
input IDs passed to the existing generation API; it does not replace the engine.
The report preserves raw files and hashes a derived generation file that attaches
those observations. Both rendered-prompt fields are decoded from captured input
IDs with special tokens retained. Missing observations fail closed. A compared
small slice is not qualification of the 31-config, 500-row matrix.

### Production preprocessing contract (LLM-169)

All nine native tasks now require `compatibility_tokenizer` for real answer
providers. It must name the **answer deployment's** local tokenizer snapshot, or
an already cached repository plus `compatibility_tokenizer_revision` (immutable
commit). No network download occurs inside evaluation. These are task args,
e.g. `-T compatibility_tokenizer=/models/gemma3/snapshots/093f9f388b31de276ce2de164bdc2081324b9767`.
An absent tokenizer fails before answer generation. Ordinary mock smoke tasks
remain usable without one; they do not establish tokenizer parity.

The supported answer boundary is Inspect native **vLLM** (including its remote
server configuration). For a DSTA deployment, use that provider with the matching
local tokenizer, and configure server/model args through Inspect. Requests send
ordinary messages, never decoded or pre-templated user text. The answer request
sets `add_special_tokens=true`, `return_token_ids=true`, and a 1024-token cap with
the local tokenizer's truncation direction. Keep the server tokenizer/template
identical to the selected snapshot. Arbitrary template arguments, older Gemma
system merging, and templates requiring legacy thinking-string surgery fail
explicitly; this is not a blanket claim of model-family equivalence.

Scorers read `metadata.compatibility`, containing decoded question, target and
stereotype plus expected input IDs/masks. Source `sample.input` and target are
unchanged. Targets/stereotypes use a 256-token round trip without special tokens;
Purple's independent question has its own 256-token cap. CCPC deliberately judges
the original source question, matching its retained evaluator. Native vLLM judge
requests add special tokens and request token IDs, but reject any prompt cap.
The existing family thinking controls remain independent for both roles.

API graders are usable with the existing per-role external-thinking assertions:
the judge **request text** is derived from the answer tokenizer identically.
Their server tokenization and hidden prompts are not qualified by that assertion.
HF answers currently fail explicitly: Inspect 0.3.288 tokenizes with
`padding=True, **tokenizer_call_args` (passing padding again collides), and its
chat-template path differs from legacy special thinking transformations. No
custom provider/engine is introduced to conceal this boundary limitation.

`python -m scripts.parity.qualify_hallu` now uses production metadata directly;
its judge phase does not rewrite the copied log. `report_hallu` compares expected
metadata IDs with actual `model` event `call.response.prompt_token_ids`. Re-run
both phases: old experimental logs lack production metadata.

For any family/configuration use `python -m scripts.parity.qualify_native answer`
then `judge`, with `--family`, `--root`, `--task-args task.json`, `--answer`, and
`--judge`. Optional `--answer-args`/`--judge-args` supply Inspect provider argument
JSON files. Task JSON selects dataset/split, cap, and tokenizer. Identity is
persisted and mismatched phase arguments are rejected. This wrapper produces
native evidence; retained legacy family invocation and strict comparator manifests
still need family-specific configuration. It does not invent paired evidence.

CPU tests use Gemma 3 revision `093f9f388b31de276ce2de164bdc2081324b9767` under
`results/tokenizers/gemma3` or `LLM169_TOKENIZER`; the tokenizer JSON hash is checked.
Download tokenizer/config/template files only. Missing fixtures explicitly skip
these tests. They exercise long input/target/stereotype/Purple questions, both
truncation directions, and all registered tasks with mock inference and real
logs. They are not new GPU qualification. The earlier two-row GPU evidence
remains the only live evidence described in REPORT.md.

The additional Qwen3 thinking test pins `Qwen/Qwen3-0.6B` at
`c1899de289a04d12100db370d81485cdf75e47ca`, tokenizer-only files under
`results/tokenizers/qwen3`. It checks thinking enabled/disabled against the
retained template helper and actual tokenizer IDs; no Qwen inference was run.

### Extending an existing frozen selection

After the required 500-row matrix, prepare full-source supplements from its exact
Parquet bytes rather than resolving dataset revisions again:

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
configuration: budgets, greedy decoding, seed, concurrency, thinking flags,
special tokens and the answer-only prompt cap. To audit existing artifacts
without rewriting their reports, run `python -m scripts.parity.settings CASE_ROOT
[CASE_ROOT ...] --output settings-audit.json`. The audit binds its checks to
source, retained-configuration and scored-log hashes. It reports zero judge
requests explicitly; settings validation alone does not establish score parity.
