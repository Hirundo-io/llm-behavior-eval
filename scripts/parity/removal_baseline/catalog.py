"""Legacy catalog constants and dataset routing for the native preset catalog.

Freezes the legacy bias-type enums, the dataset identifiers the legacy engine
owned, and the evaluator family ``EvaluateFactory`` routed each catalog dataset
to (by the logical dataset id the legacy CLI passed). Used by
tests/test_preset_inspect_mapping.py.
"""

from typing import Any

from scripts.parity.historical import legacy_module

SOURCES = [
    "llm_behavior_eval.evaluation_utils.enums:BBQ_BIAS_TYPES",
    "llm_behavior_eval.evaluation_utils.enums:UNQOVER_BIAS_TYPES",
    "llm_behavior_eval.evaluation_utils.enums:BLOOM_BIAS_TYPES",
    "llm_behavior_eval.evaluation_utils.enums:BIAS_KINDS",
    "llm_behavior_eval.evaluation_utils.censorship_utils:CCPC_DATASET_ID",
    "llm_behavior_eval.evaluation_utils.censorship_utils:CCPC_DATASET_REPOSITORY",
    "llm_behavior_eval.evaluation_utils.refusal_utils:OR_BENCH_DATASET",
    "llm_behavior_eval.evaluation_utils.refusal_utils:XSTEST_DATASET",
    "llm_behavior_eval.evaluation_utils.evaluate_factory:EvaluateFactory.get_evaluator_family",
]


def capture() -> dict[str, Any]:
    from llm_behavior_eval.presets import list_dataset_presets

    enums = legacy_module("llm_behavior_eval.evaluation_utils.enums")
    censorship = legacy_module("llm_behavior_eval.evaluation_utils.censorship_utils")
    refusal = legacy_module("llm_behavior_eval.evaluation_utils.refusal_utils")
    factory = legacy_module(
        "llm_behavior_eval.evaluation_utils.evaluate_factory"
    ).EvaluateFactory
    logical_ids = sorted(
        {
            mapping.dataset_id
            for preset in list_dataset_presets()
            for mapping in preset.inspect_tasks
        }
    )
    return {
        "enums": {
            name: sorted(getattr(enums, name))
            for name in (
                "BBQ_BIAS_TYPES",
                "UNQOVER_BIAS_TYPES",
                "BLOOM_BIAS_TYPES",
                "BIAS_KINDS",
            )
        },
        "dataset_ids": {
            "CCPC_DATASET_ID": censorship.CCPC_DATASET_ID,
            "CCPC_DATASET_REPOSITORY": censorship.CCPC_DATASET_REPOSITORY,
            "OR_BENCH_DATASET": refusal.OR_BENCH_DATASET,
            "XSTEST_DATASET": refusal.XSTEST_DATASET,
        },
        "evaluator_family": {
            dataset_id: factory.get_evaluator_family(dataset_id)
            for dataset_id in logical_ids
        },
    }
