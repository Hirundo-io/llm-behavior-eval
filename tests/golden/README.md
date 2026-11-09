# Frozen legacy expectations

Version 0.2.0 removed the legacy engine (`llm_behavior_eval/evaluation_utils/`).
Tests that used to compare native code with it live now compare with the values
in these JSON files. They are data, never an executable legacy implementation.

## Provenance

Every file was captured from the legacy engine in the frozen pre-removal checkout
**`d8c80154c1e6ae90e2896811685d7c1a9bf89e68`** (the last commit with the engine;
`FROZEN_COMMIT` in `scripts/parity/historical.py`) by

```sh
python -m scripts.parity.removal_baseline <name>
```

which runs `scripts/parity/removal_baseline/<name>.py`. Each file records:

* `provenance.commit`: the checkout's `HEAD` at capture (the runner refuses a
  modified `llm_behavior_eval/`);
* `provenance.capture`: the command;
* `provenance.sources`: the legacy symbols read (`module:symbol`);
* `values`: what those symbols returned on the tests' own deterministic inputs.

Where a test fed native-built inputs into legacy code (judge stubs, finish
reasons, synthetic rows, tokenizer rows), the inputs or their digest are stored
too and the test asserts they are unchanged, so a drifting fixture fails instead
of silently comparing different cases. Tokenizer cases use the offline tiny
tokenizer and the pinned Gemma 3 / Qwen3 snapshots whose `tokenizer.json`
SHA-256 `tests/legacy_tokenizers.py` asserts; pinned cases still skip only when
the snapshot is absent. Floats are stored as JSON numbers and compared exactly
wherever the original test compared exactly.

`tests/test_removal_contract.py` checks that every file cites `FROZEN_COMMIT`
and has a capture module, and that no test imports a legacy module or the capture
code.

## Files

| File | Tests | Legacy sources |
| --- | --- | --- |
| `bbq.json` | `test_inspect_bbq.py` | `dataset_config:DatasetConfig.model_fields`, `eval_config:EvaluationConfig.model_fields`, `eval_config:FAMILY_TOKEN_DEFAULTS`, `free_text_bias_evaluator:FreeTextBiasEvaluator.AGREEMENT_CHECK_SYSTEM_PROMPT`, `free_text_bias_evaluator:FreeTextBiasEvaluator.UNCERTAINTY_CHECK_SYSTEM_PROMPT`, `free_text_bias_evaluator:FreeTextBiasEvaluator._grade_impl`, `free_text_bias_evaluator:FreeTextBiasEvaluator._match_llm_answers`, `free_text_bias_evaluator:_JUDGE_OUTPUT_RE`, `free_text_bias_evaluator:_UNCERT_RE`, `free_text_bias_evaluator:extract_agreements`, `free_text_bias_evaluator:map_uncertainty_outputs`, `prompts:FREE_TEXT_SYSTEM_MESSAGE`, `prompts:UNKNOWN_ANSWERS` |
| `bloom.json` | `test_inspect_bloom.py` | `dataset_config:DatasetConfig.model_fields`, `enums:BLOOM_BIAS_TYPES`, `eval_config:EvaluationConfig.model_fields`, `eval_config:FAMILY_TOKEN_DEFAULTS`, `free_text_bias_evaluator:FreeTextBiasEvaluator.AGREEMENT_CHECK_SYSTEM_PROMPT`, `free_text_bias_evaluator:FreeTextBiasEvaluator.UNCERTAINTY_CHECK_SYSTEM_PROMPT`, `free_text_bias_evaluator:FreeTextBiasEvaluator._grade_impl`, `free_text_bias_evaluator:extract_agreements`, `free_text_bias_evaluator:map_uncertainty_outputs`, `prompts:FREE_TEXT_SYSTEM_MESSAGE`, `prompts:UNKNOWN_ANSWERS` |
| `catalog.json` | `test_preset_inspect_mapping.py` | `censorship_utils:CCPC_DATASET_ID`, `censorship_utils:CCPC_DATASET_REPOSITORY`, `enums:BBQ_BIAS_TYPES`, `enums:BIAS_KINDS`, `enums:BLOOM_BIAS_TYPES`, `enums:UNQOVER_BIAS_TYPES`, `evaluate_factory:EvaluateFactory.get_evaluator_family`, `refusal_utils:OR_BENCH_DATASET`, `refusal_utils:XSTEST_DATASET` |
| `ccpc.json` | `test_inspect_ccpc.py` | `censorship_utils:CCPC_*`, `censorship_utils:calculate_censorship_metrics`, `censorship_utils:censorship_dataset_fingerprint`, `censorship_utils:parse_censorship_judgment`, `eval_config:EvaluationConfig`, `eval_config:FAMILY_TOKEN_DEFAULTS` |
| `halueval.json` | `test_inspect_halueval.py` | `base_evaluator:BaseEvaluator.prepare_dataloader`, `custom_dataset:free_text_preprocess_function`, `dataset_config:DatasetConfig`, `eval_config:EvaluationConfig`, `eval_config:FAMILY_TOKEN_DEFAULTS`, `free_text_hallu_evaluator:FreeTextHaluEvaluator._grade_impl`, `free_text_hallu_evaluator:FreeTextHaluEvaluator._map_judge_outputs`, `free_text_hallu_evaluator:GRADER_TEMPLATE`, `prompts:FREE_TEXT_SYSTEM_MESSAGE` |
| `medhallu.json` | `test_inspect_medhallu.py` | `base_evaluator:BaseEvaluator.prepare_dataloader`, `custom_dataset:free_text_preprocess_function`, `dataset_config:DatasetConfig`, `eval_config:EvaluationConfig`, `eval_config:FAMILY_TOKEN_DEFAULTS`, `free_text_hallu_evaluator:FreeTextHaluEvaluator._grade_impl`, `free_text_hallu_evaluator:FreeTextHaluEvaluator._map_judge_outputs`, `free_text_hallu_evaluator:GRADER_TEMPLATE`, `prompts:FREE_TEXT_SYSTEM_MESSAGE` |
| `orbench.json` | `test_inspect_orbench.py` | `dataset_config:DatasetConfig`, `eval_config:EvaluationConfig`, `eval_config:FAMILY_TOKEN_DEFAULTS`, `free_text_refusal_evaluator:FreeTextRefusalEvaluator._grade_impl`, `free_text_refusal_evaluator:FreeTextRefusalEvaluator._judge_prompt_template`, `free_text_refusal_evaluator:parse_or_bench_judgment`, `free_text_refusal_evaluator:refusal_heuristic`, `refusal_utils:REFUSAL_MARKERS`, `refusal_utils:_REFUSAL_PREFIXES`, `refusal_utils:normalize_refusal_dataset` |
| `preprocessing.json` | `test_inspect_preprocessing.py` | `custom_dataset:free_text_preprocess_function`, `eval_config:FAMILY_TOKEN_DEFAULTS`, `free_text_hallu_evaluator:GRADER_TEMPLATE`, `util_functions:safe_apply_chat_template` |
| `probe.json` | `test_parity_evidence.py` | `custom_dataset:free_text_preprocess_function` |
| `prompt_injection.json` | `test_inspect_prompt_injection.py` | `custom_dataset:free_text_preprocess_function`, `eval_config:EvaluationConfig`, `eval_config:FAMILY_TOKEN_DEFAULTS`, `free_text_injection_evaluator:FreeTextPromptInjectionEvaluator`, `prompts:FREE_TEXT_SYSTEM_MESSAGE` |
| `unqover.json` | `test_inspect_unqover.py` | `dataset_config:DatasetConfig.model_fields`, `eval_config:EvaluationConfig.model_fields`, `eval_config:FAMILY_TOKEN_DEFAULTS`, `free_text_bias_evaluator:FreeTextBiasEvaluator.AGREEMENT_CHECK_SYSTEM_PROMPT`, `free_text_bias_evaluator:FreeTextBiasEvaluator.UNCERTAINTY_CHECK_SYSTEM_PROMPT`, `free_text_bias_evaluator:FreeTextBiasEvaluator._grade_impl`, `free_text_bias_evaluator:extract_agreements`, `free_text_bias_evaluator:map_uncertainty_outputs`, `prompts:FREE_TEXT_SYSTEM_MESSAGE`, `prompts:UNKNOWN_ANSWERS` |
| `xstest.json` | `test_inspect_xstest.py` | `eval_config:EvaluationConfig`, `eval_config:FAMILY_TOKEN_DEFAULTS`, `free_text_refusal_evaluator:FreeTextRefusalEvaluator._grade_impl`, `free_text_refusal_evaluator:parse_xstest_judgment`, `free_text_refusal_evaluator:refusal_heuristic`, `refusal_utils:REFUSAL_MARKERS`, `refusal_utils:XSTEST_DATASET`, `refusal_utils:XSTEST_JUDGE_PROMPT`, `refusal_utils:_REFUSAL_PREFIXES`, `refusal_utils:normalize_refusal_label` |

Sources are relative to `llm_behavior_eval.evaluation_utils`.
