# LLM-171: legacy runtime removal (0.2.0)

Built on the LLM-169 qualification layer (#208) at
`d8c80154c1e6ae90e2896811685d7c1a9bf89e68`, the last commit with the legacy
engine and the frozen pre-removal checkout for every historical legacy run
(`scripts/parity/historical.py`, `scripts/parity/README.md`). This layer redoes
the earlier removal (694f110, on the pre-redesign stack) for the self-contained
benchmark design. `scripts/parity/REPORT.md` and all committed evidence are
unchanged. No all-model or all-backend parity is claimed.

## What was removed

* `llm_behavior_eval/evaluation_utils/` (22 modules, 6,044 lines): the free-text
  evaluators, the Transformers and vLLM engines, model and tokenizer loading,
  batch-size search and OOM backoff, the generation cache, the legacy dataset,
  evaluation, sampling and vLLM configs, the output writers and the engine's
  MLflow logging. There is no forwarding package.
* The deprecated lazy exports of `llm_behavior_eval/__init__.py`
  (`EvaluateFactory`, `EvaluationConfig`, ..., see `CHANGELOG.md`). The package
  exports the preset catalog only.
* Twelve test modules that tested only the engine (6,358 lines):
  `test_base_evaluator`, `test_bloom`, `test_collator`, `test_dataset`,
  `test_dataset_config`, `test_dataset_identity`, `test_eval_engines`,
  `test_free_text_censorship_evaluator`, `test_free_text_utils`,
  `test_mlflow_integration` (the engine's MLflow logging; the native hook keeps
  `test_inspect_mlflow`), `test_util_functions` and
  `test_legacy_entrypoint_contracts` (factory routing and `EvaluationConfig`
  LoRA, output-directory and credential validation). Their native assertions were
  kept: the BLOOM preset expansions and the catalog completeness check moved to
  `tests/test_preset_catalog.py`; the catalog dataset ids and the factory's
  dataset-to-family routing are frozen in `tests/test_preset_inspect_mapping.py`.
* Dependencies that only the engine used: Torch, Transformers, Accelerate,
  bitsandbytes, NumPy, pandas, pydantic, pydantic-settings, tqdm, NetworkX,
  protobuf and sentencepiece (base), and the `gcs`, `azure` and `gpt-oss` extras.
  `uv.lock` drops 28 packages and the Python < 3.12 resolution forks.

`common/preprocessing.py` (the opt-in `-T legacy_tokenizer` mechanics) and all
native code are kept. Where native code lazily imports Transformers (that module
and Purple Llama's `response_tokenizer`), a missing extra now raises an
`ImportError` naming `llm-behavior-eval[tokenizers]`. Package docstrings and
`parity.yaml` comments that cite legacy source files now name the frozen commit.

## Frozen legacy expectations

Tests that compared native code with the live engine now compare it with the
engine's outputs frozen in `tests/golden/*.json` (12 files). Each was captured by
`python -m scripts.parity.removal_baseline <name>` running the legacy code at
the frozen commit on the tests' own deterministic inputs, and records that commit,
the command and the legacy symbols read. Captures were run at both `b52404f`
and `d8c80154` (which differ only by two evidence JSON files): all values were
identical, and repeated captures are byte-identical. Where a test fed native-built
inputs into legacy code, the inputs are stored and asserted too. Every assertion
keeps its cases and equality; floats stay exact where they were exact.
`tests/golden/README.md` lists each file's tests and sources.

`tests/test_removal_contract.py` (ported from 694f110 and extended) checks:
discovery imports no model runtime; the engine and the removed exports are
gone; version 0.2.0 and Python 3.12–3.13 with the five base dependencies; the
historical scripts (`reuse_legacy`, the `legacy` phase of `qualify_matrix` and
`run_matrix`, and the baseline capture) exit with a message naming the frozen
commit and create nothing; no package or test module imports the engine or the
capture code; and every golden file cites the frozen commit and has a capture
module. `tests/test_offline_bundle_contract.py` (ported from 694f110; tasks no
longer take `max_samples`, so whole fixture splits are compared) loads raw HF
snapshot Parquet bundles for all nine benchmarks and rejects a tampered CCPC
bundle.

`reuse_legacy.py`'s recorded-judgment reader moved to
`scripts/parity/recorded_judgments.py`, so its tests run without the engine.

## Installation

Python 3.12 or 3.13. Base dependencies: Inspect, `agent-client-protocol<1`,
the OpenAI client, `datasets` and Typer. The `agent-client-protocol` bound is new:
Inspect 0.3.288 requires `>=0.12.0` without an upper bound, and the 1.0 release
removed `acp.helpers`, so every `inspect eval` in a fresh unlocked install failed
with `ModuleNotFoundError` (found by the clean-install check below; the lock
already held 0.12.1).

* CPU, hosted and `openai-api/` providers: the base package.
* Reporting: `[brief,mlflow]`.
* `-T legacy_tokenizer` and Purple Llama's `response_tokenizer`: `[tokenizers]`.
* Inspect's `hf/` provider: `[hf]`. Inspect's vLLM provider on Linux: `[vllm]`
  (the tested vLLM 0.30 / Transformers 5.20 pair).

The dev group installs `llm-behavior-eval[mlflow,hf,vllm,brief]`, so
`uv sync --locked --extra brief --extra mlflow` (what CI runs) keeps the HF,
tokenizer and Gemma 4 compatibility tests running instead of skipping them.

## Validation

All on Python 3.12, in a `uv sync --locked --extra brief --extra mlflow`
environment, with the pinned Gemma 3 / Qwen3 tokenizer snapshots present:

* `pytest` (plain console entry point): 914 passed, 0 skipped. Before removal,
  the same environment at the #208 head ran 1,013 passed; the difference is
  the twelve removed engine-only modules, offset by the ported and moved tests.
* The rewritten legacy-comparing test files also pass with a meta-path blocker
  that makes any `llm_behavior_eval.evaluation_utils` import fail, run before
  the engine was deleted.
* `ruff format --check --diff .`, `ruff check .`, `pyrefly check` (0 errors) and
  `uv lock --check`: pass.
* Wheel built from a copy of the tree: 0.2.0, `Requires-Python <3.14,>=3.12`,
  no `evaluation_utils`, `scripts` or `tests`, and the 30 per-package
  `parity.yaml`, README and `target_metrics.json` files.
* Clean install outside the repository (fresh Python 3.12 venv, wheel with
  `[brief]`): Torch, Transformers, vLLM, Accelerate and bitsandbytes absent;
  `from llm_behavior_eval import EvaluateFactory` raises `ImportError`;
  `inspect list tasks llm_behavior_eval` from site-packages lists all ten tasks;
  mock evals by registered name of `inspect_template`, and of `bbq` and
  `halueval` on local `save_to_disk` datasets with a mock grader, succeed with
  their report metrics; the retired `llm-behavior-eval` exits 2.

## Remaining limits

The representative qualification does not establish all-model or all-backend
parity, HF GPU score equality or `google/gemma-4-12B-it` judge parity; a GPU
re-run with `parity.yaml` is still pending (`scripts/parity/README.md`).

Do not upgrade the platform to 0.2.0 on its own: its worker still calls the
removed API, and its release manifest, version pairing and local answer-tokenizer
paths need a companion migration. The offline bundle tests do not establish
end-to-end platform readiness.
