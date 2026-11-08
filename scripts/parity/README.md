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
2. Provision already-authorized local weights for an owner-selected tested small
   instruct answer model (proposed Gemma 3 4B), Gemma 3 12B judge and CCPC Gemma 4
   26B A4B judge. Record actual weight/tokenizer revisions and hashes. No model
   revision has been selected or accepted in this report.
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
        model_path_or_repo_id=os.environ['ANSWER_SNAPSHOT'],
        judge_path_or_repo_id=os.environ['JUDGE_SNAPSHOT'],
        inference_engine='transformers',
        sample=False, sample_judge=False, enable_thinking=False,
        use_4bit=False, use_4bit_judge=False,
        max_samples=500, batch_size=1, judge_batch_size=1,
        max_answer_tokens=128, max_judge_tokens=32,
        results_dir=Path('results/parity/legacy'),
    ),
    DatasetConfig(file_path=os.environ['DATASET_SOURCE'],
                  dataset_id='hirundo-io/halueval',
                  dataset_type=DatasetType.BIAS, seed=42),
).evaluate()
```

```python
# Native process, same local weights and dataset; HF backend on both sides
import os
from inspect_ai import eval
from inspect_ai.model import get_model
from llm_behavior_eval.halueval.task import halueval

answer = get_model('hf/local-answer', model_path=os.environ['ANSWER_SNAPSHOT'],
                   tokenizer_path=os.environ['ANSWER_SNAPSHOT'],
                   enable_thinking=False, do_sample=False)
grader = get_model('hf/local-judge', model_path=os.environ['JUDGE_SNAPSHOT'],
                   tokenizer_path=os.environ['JUDGE_SNAPSHOT'],
                   enable_thinking=False, do_sample=False)
eval(halueval(dataset_path=os.environ['DATASET_SOURCE'], max_samples=500),
     model=answer, model_roles={'grader': grader},
     max_connections=1, log_dir='results/parity/native')
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
