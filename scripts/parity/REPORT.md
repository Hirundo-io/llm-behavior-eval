# LLM-169 qualification report — 2026-11-08

**Engine retirement remains blocked.** The designated-model follow-up below now records real old/new HaluEval answers and independent judge scores on two rows. The 31-configuration qualification matrix remains incomplete. The earlier eight-row generation-only diagnostic is historical evidence.
`pending-report.json` lists all 31 required split configurations as incomplete;
old, new, delta and verdict agreement are unavailable, not zero. No proposed
tolerance or divergence has owner acceptance. No legacy files were deleted.

| Benchmark | Deterministic semantic evidence reviewed | Remaining qualification gate |
|---|---|---|
| BBQ (12) | Classification precedence, stereotype formula, seed-42 selection | decoded question, 1024 input / 256 target+stereotype round trips; live output |
| UNQOVER (4) | Shared bias scorer; no unbias multiplier | same bias gaps; live output |
| BLOOM (9) | Shared bias scorer; paired and ambiguous identities | same bias gaps; live output |
| HaluEval | NOT_ATTEMPTED is not incorrect; all-row denominator; percent accuracy | decoded question/input/target; live output |
| Med-Hallu | Shared hallu formulas and source selection | same hallu gaps; live output |
| Purple Llama | Stop-only Yes/No, ASR percent; malformed means No | independent 256-token judge question, 1024 input; live output; Orr source still unlocated |
| CCPC | Frozen order/fingerprint, all answers judged, invalid verdict makes rate undefined | input round trip, actual pinned dataset, Gemma template support; live output |
| XSTest | stop/length judging, unknown skips, zero empty denominators | decoded question/input, structured reasoning; live output |
| OR-Bench | Same refusal denominators, bracket parser | decoded question/input, structured reasoning; live output |

“Reviewed” does not mean this environment passed each benchmark suite. All nine
READMEs and relevant current preprocessing, generation-record, grading and
selection source were inspected. XSTest already passes seed=42 in this checkout;
the OR-Bench README's claim that it omits the seed is stale. CCPC deliberately
retains source order. No sibling-owned documentation was rewritten.

## Compatibility decisions, version 1 (proposed, not approved)

Faithfully preserve the legacy seed/cap, finish eligibility, score denominators,
prompt text, thinking settings and token budgets. The new offline tokenizer probe
pins padding=max_length, tokenizer-default input special tokens, input cap 1024,
target/stereotype cap 256 with no special tokens, and skip-special-token decode.
Purple fresh-run judge questions independently use the 256 cap; only old caches
without judge_questions fall back to the decoded input.

The probe is evidence tooling, **not a wired native preprocessing fix**. Do not
feed decoded chat templates back as user messages: Inspect would template them
again. Matching judge text alone would hide differing answer-model inputs. A
complete compatibility path needs exact tokenizer/template revisions and a native
provider input hook that preserves token IDs/masks or a verified equivalent.
No such provider boundary is qualified here, and no custom model engine was added.
Model-specific Gemma system demotion, Nemotron /no_think, Muse Glimmer channel
rewriting, custom thinking argument names, left/right truncation and structured
reasoning need explicit fixtures against selected real tokenizers before wiring.

Proposed versioned divergences requiring owner decision: bias exceptional finish
reasons/missing choices now skip as empty+incorrect whereas legacy can exhaust
its judge iterator; native structured reasoning may be excluded differently;
Purple empty-run incomplete rate differs (native NaN versus legacy zero).
These are not silently covered by an aggregate tolerance. CCPC publication
contract differences remain separate from parity to this repository's legacy.
The candidate Purple README documents the already-exhausted source search; no
new Orr source was found in this checkout, and no port claim is made.

## Real H200 diagnostic

Parent started the explicitly authorized H200 after discovering all three VMs
were stopped. Initial hardware check: NVIDIA H200, 143,771 MiB total, 0 MiB used,
0% utilization, no compute jobs. Existing cached GPT-OSS-20B weights and Purple
Llama data were usable; the required Gemma judges and a tested small answer model
were not found in the inspected cache/model locations. No model was downloaded.
Only h200 was used. Existing experiments/environments were not modified.

An isolated `.venv` and output directory used existing dependency paths, with
small Inspect 0.3.288 and Accelerate 1.15.0 installs via uv. Actual runtime:
Python 3.12.16, torch 2.14.0, transformers 5.20.2, vLLM 0.33.0, datasets 5.1.0.
vLLM is outside the pinned 0.30.x contract. GPT-OSS-20B revision is
`6cee5e81ee83917806bbde320786a8fb61efebee`, confirmed by cached file metadata;
config/tokenizer/template hashes are committed in `gpu-summary.json`.
Dataset upstream commit is unresolved; cached Arrow builder fingerprint and
exported Parquet hash are retained. This model is outside the README tested list.

Eight rows were selected independently using source IDs and HF seed-42 shuffle.
Both real runs used greedy 128-token answers, engine seed 0 (the legacy vLLM
default), one sequence, eager execution, context limit 8192 and memory fraction
0.5. Legacy retained preprocessing caps at 1024; native candidate remains
unchanged. Thinking-off was requested; GPT-OSS still emitted reasoning. This is
not a qualified thinking-off setup. No judge executed. Native task scoring was
disabled; its explicit unused mock grader role supplied no verdicts or scores.

Observed generation diagnostics:

- Raw completion matches: **0/8**.
- Finish matches after `length` → `max_tokens`: **7/8**.
- Actual judge-question text matches: **8/8**.
- Independent tokenizer audit: expected rendered templates/input IDs match for
  **8/8**, with 88–161 input tokens. This does not exercise truncation and is not
  captured native-server boundary evidence.

Legacy output includes literal Harmony analysis/final channel text; Inspect
stores structured reasoning and final text separately. Row 84 has identical
reasoning text after removing the legacy analysis prefix, explaining its observed
representation difference. Rows 51 and 3 have matching final text but different
reasoning. Rows 124, 203, 70 and 222 differ in reasoning and/or whether final text
is reached within the budget. Row 120 is legacy `length` versus native `stop`:
this changes stop-only judge eligibility (legacy 2 stopped rows versus native 3).
Every row's observation is recorded in `gpu-summary.json`. The exact cause of
reasoning-sequence differences is **unresolved**; native server token boundaries,
provider rendering, generation defaults and numeric behavior need inspection.
No aggregate tolerance can certify these unexplained differences.

The first legacy startup failed because existing ninja/CUDA binaries were not on
PATH. Its log is preserved; rerunning with those existing tool paths completed.
Native startup warned about an unrelated inherited inspect-evals entrypoint
incompatibility, but the saved `.eval` status is success with all eight samples.
The real generation summary was recomputed locally and byte-compared with the
remote summary. Independent source/tokenizer reconstruction verifies legacy row
alignment before comparing output. Raw evidence totals under 1 MiB, kept at:

- Local: `results/llm169-h200/` (ignored experimental outputs).
- H200 persistent disk: `/mnt/filesystem-x0/llm169_parity_20261108/results/`.
- Committed compact observations, hashes and runtime: `gpu-summary.json`.

After both processes exited, `nvidia-smi --query-compute-apps` returned no jobs.
The parent independently confirmed the GPU was idle, restored the original
`VM_OFF_h200` watchdog flag, and confirmed the instance reached `STOPPED`.
The diagnostic artifacts are also copied to the integration checkout's ignored
`results/llm169-h200/` directory.

## Benchmark score gate

These fields are unavailable, not zero. The diagnostic above supplies no judge
agreement or benchmark score. All 31 split configurations remain incomplete in
`pending-report.json`.

| Benchmark | Old primary score | Native primary score | Delta | Judge verdict agreement | Gate |
|---|---|---|---|---|---|
| BBQ (12 splits) | — | — | — | — | Missing qualified runs |
| UNQOVER (4) | — | — | — | — | Missing qualified runs |
| BLOOM (9) | — | — | — | — | Missing qualified runs |
| HaluEval | — | — | — | — | Missing qualified runs |
| Med-Hallu | — | — | — | — | Missing qualified runs |
| Purple Llama | — | — | — | — | Eight real generation rows disagree; no judge run |
| CCPC | — | — | — | — | Missing qualified runs and pinned judge |
| XSTest | — | — | — | — | Missing qualified runs |
| OR-Bench | — | — | — | — | Missing qualified runs |

## Deterministic validation

The local `.venv` was created with uv. Local PyPI DNS failed; read-only cached
archives and existing pure-Python tooling supplied the fallback environment.
`runtime.json` records this local environment separately from the H200 runtime.

- `XDG_CACHE_HOME=/tmp/llm169-xdg-cache XDG_DATA_HOME=/tmp/llm169-xdg-data .venv/bin/python -m pytest tests/test_parity_evidence.py -q`: **18 passed**. Includes actual native mock compressed `.eval` write/read, all evidence unit conversions, and a direct regression against actual legacy preprocessing with a deterministic tokenizer.
- Same pytest invocation with all nine `tests/test_inspect_<benchmark>.py` files added: **107 passed, 100 failed, 11 errors**. Incompatible cached Pillow `_imaging` and sandbox endpoint permission failures mean this broad local run is not acceptance evidence. It must be rerun in a fully supported environment.
- Repository-wide Ruff format/check: pass. Focused Pyrefly over scripts/parity and the evidence tests: **0 diagnostics**.
- Pending comparison exits 2 with all 31 rows incomplete, as designed.

Parent integration validation (same day): all nine benchmark test modules,
foundation, registry smoke and evidence tests passed together: **254 passed**
in 75.45 seconds. This run used the existing OR-Bench worktree's complete
comparison environment with the integration checkout on `PYTHONPATH`, offline
HF flags, and execution outside sandbox socket restrictions. Repository Ruff
format/check and focused parity-script/test Pyrefly checks also passed after
narrowing the shuffled Arrow dataset type. These are deterministic/mock checks;
they do not qualify the live benchmark score gate.

Fixtures cover missing/duplicate/misaligned IDs, hash mismatch, failed samples
and runs, absent fields/provenance/primary metrics, undefined metrics, confusion
counts, percent/fraction conversion, batched generation evidence, and 1024/256
round trips. Existing benchmark formula tests remain distinct from live parity.
The sidecar/mapping provenance still requires an audit of the original source;
the comparator never invents missing IDs or model revisions.

## Remaining decisions and work

Inspect HF 0.3.288 supports
`tokenizer_call_args={"truncation": True, "max_length": 1024}`. Its provider also
supplies `padding=True` explicitly; passing `padding="max_length"` through that
same dictionary duplicates the keyword. This is a feasible native cap control
but not complete legacy token-boundary equality. No shared native compatibility
path was wired without provider-boundary verification; the probe is evidence
only. Long-input/target tests and special template behavior still gate retirement.

The original diagnostic required selected answer/judge revisions and a supported
pinned runtime; the follow-up below now supplies those resources. Remaining work
includes all pinned dataset snapshots, complete
provider-boundary evidence, full old/native scoring runs, explicit tolerances and
per-divergence sign-off. Existing H200 source copies contained no discoverable
Purple task filename; requested Orr source remains unlocated. LLM-171 stays gated.

Evidence remains local under the parent's instruction following the approval
rejection. The attempted detailed
Linear update was rejected by automatic approval review; no issue comment was
posted. No PR, push, merge, release, legacy deletion or downstream Slack update
occurred. Small JSON evidence files are explicitly tracked despite broad ignore
rules; model weights and experimental outputs remain untracked.


## Designated-model follow-up in checkout t3-585e80c8

This follow-up used the authorized existing H200 and provisioned all three issue-
designated model snapshots. No model-selection approval gate was introduced.
The GPU was initially idle and the 1 TB persistent disk had 467 GB free.
The isolated task root is `/mnt/filesystem-x0/llm169_585e80c8`; source was copied
from baseline `a35f01d`, with this checkout's changes overlaid. The previous task
root and environment were not modified. `uv sync --locked --python 3.12` completed.
Actual versions: Python 3.12.16, Torch 2.13.0, vLLM 0.30.0, Transformers 5.20.2, Inspect 0.3.288.

Immutable models, downloaded successfully (approximately 79 GiB total):

- Answer `google/gemma-3-4b-it`: `093f9f388b31de276ce2de164bdc2081324b9767`.
- Independent judge `google/gemma-3-12b-it`: `96b6f1eccf38110c56df3a15bffe176da04bfd80`.
- CCPC judge `google/gemma-4-26B-A4B-it`: `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` (downloaded, not yet qualified by inference).

`provision.json` records snapshot locations, package versions, and hashes of
configuration/tokenizer/template metadata. Existing HF credentials sufficed;
there was no gated-model access or disk-capacity blocker.

The first paired run uses `hirundo-io/halueval` at
`1516057d20afbd37202ce4d3f8c761c36c4d481c`, exported as Parquet. The source record
binds its SHA256 and independently selected seed-42 IDs `2342` and `554` from
3,000 original rows. Both sides use greedy BF16, engine seed 0, batch/sequence
limit 1, answer/judge budgets 128/32, context 8192, eager mode, no quantization,
and GPU memory utilization 0.5. Dataset shuffle seed remains 42.

Native provider controls explicitly request `return_token_ids`,
`add_special_tokens=true`, and answer-only right-side `truncate_prompt_tokens=1024`.
The production hallucination config now preserves these caller controls while
enforcing thinking-off. A regression verifies both roles, nested template kwargs,
and non-mutation of caller config. Native answer and judge run through separate
public Inspect eval/score processes. The original answer log is retained; a copy
is adapted for judging with the decoded legacy question and 256-token target
round trip. This adapter remains an experiment in the runner, not a completed
production preprocessing migration. Legacy judge prompt tokenization does **not**
apply the answer's 1024-token cap.

The first native and legacy run each produced both answers with `stop`; answers
match exactly, and the independent real 12B judges both returned `CORRECT` on
both rows. Old and native accuracy are 100%, delta 0 percentage points; verdict
agreement is 2/2. Native empty-response count and incomplete-response rate are 0.
These are measured values for two rows, not inferred values for missing runs.
Actual native model-event responses contain prompt token IDs. They match the
legacy oracle for both rows, including two leading BOS tokens. This is captured
server evidence, not a claim that independent template reconstruction alone
proves boundary equality.

A second retained-API run observes actual legacy generation input tensors and
preserves the original run separately. `report_hallu.py` binds raw response,
generation, and observed-boundary files by SHA256, attaches observations in a
derived file, and invokes the existing strict comparator. Rendered-prompt fields
on both sides are decoded from their actual input IDs with special tokens kept.
Missing capture rows cause failure rather than invented evidence.

Validation: the supported H200 environment passed all 18 HaluEval tests,
including the new provider-control regression. Repository Ruff check and format
checks pass. Focused runner type checking is recorded with the local evidence.

This is **partial implementation and live qualification**, not completion of
LLM-169. Other families' small paired runs, 500 rows across all 31 configurations,
full primary splits, long-input/target/stereotype boundary cases, and production
preprocessing integration remain unperformed. No resource/access blocker was
found that justifies declaring those unavailable. No tolerance or divergence was
accepted. The legacy engines remain; the separate LLM-171 removal candidate was
not integrated. No Linear, Slack, PR, push, or release action occurred.

The completed strict two-row report is committed as `gemma-small-slice.json`:
status `compared`, no errors, no mismatches, and acceptance explicitly pending.
Actual answer-input IDs and special-token-preserving decoded prompts match for
both rows. The local recomputation equals the remote small-slice result exactly.
The full comparator exits 2 because all 31 required full configuration pairs are
still absent; the separately named small slice does not masquerade as one.
Full raw evidence is retained locally at `results/llm169-gemma-h200/` and remotely
under the task root's `evidence/`. Focused Pyrefly finished with 0 errors.

Cleanup: all own inference and validation processes exited. H200 showed 0% GPU
utilization, 0 MiB used, no compute applications, and no containers (checked with
`sudo -n docker ps -q`). The original `VM_OFF_h200` contents were restored and the
unique backup removed before requesting stop. Box and rtx2 were untouched.
The final Nebius state query confirmed **STOPPED**.

## Production preprocessing follow-up (t3-33e12a3f)

The nine production task factories now wrap their existing Inspect solvers with
shared preprocessing. Scorers consume explicit compatibility metadata; source
input and target are not rewritten. The HaluEval qualification script no longer
imports a preprocessing oracle into its native phase or edits copied log inputs.
The generic native wrapper supports all nine tasks and arbitrary task JSON
configuration, with persisted phase identity. The existing thinking-control
merge helper remains unchanged.

CPU checks use the pinned Gemma 3 tokenizer above and Qwen3-0.6B at
`c1899de289a04d12100db370d81485cdf75e47ca`. Long 1024/256 boundaries match retained
preprocessing in both truncation directions; registered tasks exercise actual
preprocessing, scoring and saved logs with mock inference. CCPC retains its
original source judge question by design. Purple uses its independent capped
question and the production answer tokenizer for response token counts.

Supported production answer transport is native vLLM. Missing explicit tokenizer
identity, unsupported answer providers, custom template transformations and
judge truncation controls fail clearly. HF 0.3.288 source accepts
`tokenizer_call_args` but passes padding separately; HF compatibility is not
implemented or qualified here. API graders can use existing external thinking
assertions with identical judge request text, but that does not establish their
hidden tokenization. See README for deployment and runner commands.

No new GPU inference was performed, and no VM was started. The earlier two-row
live result remains experimental evidence from its original runner, not a live
validation of this production change. Re-run small slices using production task
options before the expensive matrix. Other-family retained invocation/comparator
manifests, the full matrix, HF answer support and model-specific template surgery
remain concrete follow-up work. No tolerance or divergence is accepted. Legacy
engines remain intact; no PR, push, merge, release or external issue update.

Validation environment: Python 3.13, Inspect 0.3.288, Transformers 5.20.2. The
checkout's editable install reused existing dependency files after a fresh
installation exhausted local disk capacity. Generated root egg-info was moved
out of the checkout because it shadowed editable installation metadata and
prevented registered discovery after changing directories. Tests ran outside the
sandbox after its async log reader stalled. These were local validation issues,
not model access/GPU blockers.

Validation: affected deterministic suite **235 passed, 1 skipped** (the optional
`inspect_brief` package is absent in the reused validation environment). Full Ruff
lint/format and focused production/tooling Pyrefly pass. No generated distribution
artifacts or tokenizer downloads are committed.

Parent integration recheck at `0393941`: **274 passed** in 80.48 seconds across
all nine benchmark modules, foundation, registry smoke, parity evidence, and
production preprocessing tests. The pinned Gemma/Qwen tokenizer fixtures were
present, so these boundary tests ran rather than skipped. Repository Ruff
format/check and focused preprocessing/runner Pyrefly checks passed. This remains
CPU/mock qualification; the production GPU matrix is still pending.

## Production GPU qualification in checkout t3-f8a66042 (in progress)

This run freezes production source at `f35fb814b0c99081707f81241b39ed72cd4ebcbf`.
The later HF support change is deliberately not mixed into these vLLM pairs.
The isolated H200 run root is `/mnt/filesystem-x0/llm169_f8a66042`, reusing the
supported environment and three immutable model snapshots from the previous run.
Runtime source-path verification confirms imports from the new source copy.
Actual packages: Python 3.12.16, Torch 2.13.0, vLLM 0.30.0, Transformers 5.20.2,
Inspect 0.3.288, Datasets 5.0.1.

`production-pinned-datasets.json` records all 31 immutable revisions, complete
export hashes, selected-ID hashes, and source sizes. The required capped matrix
contains 13,217 rows per implementation. Eleven configurations have fewer than
500 source rows; those runs use the full source. CCPC retains its exact revision,
ordered 500-row validation and fingerprint. XSTest exposes only `test`; the
runner was corrected to follow the production train-or-single-split rule.
The failed first preparation log is preserved separately.

New tooling invokes only retained EvaluateFactory and native Inspect eval/score
APIs. `qualify_matrix.py` prepares and executes individual phases;
`run_matrix.py --group-native` and `qualify_native_batch.py` reuse native servers
across tasks while keeping answer and judge processes separate. Legacy boundary
observers preserve actual answer/judge input tensors and unrounded metrics;
`report_matrix.py` compares those with native model-event token IDs and scorer
metadata. Original sample inputs and targets are checked unchanged. Derived
artifacts bind the raw files by hashes. `matrix_report.py` keeps absent rows and
scores missing and cannot substitute a small slice for a 500-row configuration.
Synthetic long-boundary fixtures are explicitly separate, and cannot modify CCPC.

The existing `qualify_native.py` separate scoring phase now forwards task metrics.
This matters for bias, whose scorer declares no metrics itself. A real Inspect
mock-log regression exercises answer then separate scoring; it is tooling evidence,
not live benchmark qualification. No production preprocessing/scorer code changed.

The first production HaluEval two-row pair completed: actual answer input token
IDs (including double BOS), rendered prompts, outputs, finish reasons, judge
questions and verdicts match. Both accuracy values are 100%. This early wiring
pair used native request seed 0 versus retained request seed 42; engine seed was
0 on both sides. The runner now uses request seed 42 for subsequent pairs. The
early pair must not be represented as matched decoding provenance or a completed
matrix row. Its legacy judge token boundary was not captured. Raw evidence lives
locally in ignored `results/llm169-production-h200/` and under the remote run root.

The nine-family small checks and subsequent full matrix are still in progress.
No tolerance, divergence, engine retirement, or HF GPU qualification is accepted.
The H200 was initially STOPPED and idle after starting; its original watchdog flag
is backed up at the unique `VM_OFF_h200.llm169-f8a66042-backup` sibling path.
GPU ownership remains with this run until all work finishes and original-state
cleanup is verified. Box and rtx2 are untouched. No PR, push, release, Linear or
Slack action has been taken.
