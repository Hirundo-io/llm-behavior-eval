# Changelog

## 0.2.0 (unreleased)

Breaking release: llm-behavior-eval is now only a set of native
[Inspect](https://inspect.aisi.org.uk/) tasks (LLM-171). The 0.1.x legacy
evaluation engine is removed. See `INSPECT_MIGRATION.md` for running the
benchmarks with Inspect and `REMOVAL_VALIDATION.md` for how the removal was
checked.

### Removed: the legacy engine

`llm_behavior_eval/evaluation_utils/` is deleted with everything that existed only
for it: the free-text evaluators, the Transformers and vLLM engines and model
loading, batch-size search and OOM backoff, the generation cache and resume, the
legacy dataset and evaluation configs, the CSV/JSON output writers and the
engine's own MLflow logging. There is no forwarding package.

### Removed: the legacy Python API

`llm_behavior_eval` no longer exports `EvaluateFactory`, `EvaluationConfig`,
`DatasetConfig`, `PreprocessConfig`, `DatasetType`, `MlflowConfig`, `VllmConfig`,
`SamplingConfig`, `FreeTextBiasEvaluator`, `FreeTextRefusalEvaluator`,
`SYSTEM_PROMPT_DICT`, `load_tokenizer_with_transformers`,
`load_transformers_model_and_tokenizer`, `pick_best_dtype` or
`safe_apply_chat_template`; the evaluator and config classes had warned since the
Inspect migration. The package exports the preset catalog only:
`DatasetPreset`, `InspectTaskMapping`, `expand_dataset_preset`,
`get_dataset_preset` and `list_dataset_presets`.

### Removed: the legacy CLI

`llm-behavior-eval <model> <preset>` (and `python -m llm_behavior_eval`) runs
nothing. It prints how to run the preset with `inspect eval` and exits with status
2, as it has since the Inspect migration. No legacy flag is translated.

### Ended: built-in inference defaults

The legacy engine chose inference settings for you. None of them remain, for the
model under test or for the judge:

* answer and judge token budgets per family (bias, hallucination and prompt
  injection 128/32; censorship 8192/4096; refusal 2048/128), greedy decoding for
  both roles, and the thinking switch (on for refusal, off otherwise);
* batch sizes, 4-bit loading, the default judge model (`google/gemma-4-12B-it`)
  and the 1024-token prompt truncation.

Set inference options per role through Inspect (`--max-tokens`, `--temperature`,
`-M ...`, `--model-role grader=...`), or pass a benchmark's `parity.yaml` with
`--run-config` to reproduce the legacy settings exactly. Always pass a grader:
none is implied.

### Python and dependencies

* Requires Python 3.12 or 3.13 (was `>=3.10.12,<3.14`).
* Base dependencies are Inspect, its OpenAI-compatible client, Hugging Face
  `datasets` and Typer (for the retirement notice), plus
  `agent-client-protocol<1`: Inspect 0.3.288 does not bound it, and its 1.0
  release breaks every `inspect eval` in a fresh install. Torch, Transformers,
  Accelerate, bitsandbytes, NumPy, pandas, pydantic, pydantic-settings, tqdm,
  NetworkX, protobuf and sentencepiece are no longer required.
* New extras: `hf` (Inspect's `hf/` provider: Transformers, Torch, Accelerate),
  `tokenizers` (Transformers, sentencepiece, protobuf: the opt-in
  `-T legacy_tokenizer` parity preprocessing). `vllm` keeps the tested
  vLLM 0.30 / Transformers 5.20 pair. `brief` and `mlflow` are unchanged.
* Removed extras: `gcs`, `azure`, `gpt-oss` (legacy engine only).

### Historical legacy runs

Reproducing a legacy run (the `legacy` phase of `scripts/parity/qualify_matrix.py`,
`scripts/parity/reuse_legacy.py`, or recapturing `tests/golden/`) needs the frozen
pre-removal checkout `d8c80154c1e6ae90e2896811685d7c1a9bf89e68`; see
`scripts/parity/README.md`. Those scripts exit with that instruction when the
legacy engine is absent. Tests that compared native code with the legacy engine
now compare it with legacy outputs frozen in `tests/golden/` at that commit.

### Platform note

Do not upgrade the platform to 0.2.0 on its own: its worker still calls the
removed API. Migrate it to native Inspect tasks first.
