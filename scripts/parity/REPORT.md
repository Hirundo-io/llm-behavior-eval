# LLM-169 qualification report — 2026-11-08

**Engine retirement remains blocked.** No qualified old/new benchmark scores exist here. A real eight-row generation-only diagnostic is recorded below.
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

Required resources/decisions: owner-approved tested answer and independent judge
revisions, supported pinned runtime, all pinned dataset snapshots, complete
provider-boundary evidence, full old/native scoring runs, explicit tolerances and
per-divergence sign-off. Existing H200 source copies contained no discoverable
Purple task filename; requested Orr source remains unlocated. LLM-171 stays gated.

Evidence remains local under the parent's instruction following the approval
rejection. The attempted detailed
Linear update was rejected by automatic approval review; no issue comment was
posted. No PR, push, merge, release, legacy deletion or downstream Slack update
occurred. Small JSON evidence files are explicitly tracked despite broad ignore
rules; model weights and experimental outputs remain untracked.
