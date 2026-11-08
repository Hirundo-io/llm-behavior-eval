import importlib
import inspect

import pytest

from llm_behavior_eval import expand_dataset_preset, list_dataset_presets


@pytest.mark.parametrize("preset", list_dataset_presets(), ids=lambda p: p.name)
def test_every_expansion_has_exact_native_identity_and_arguments(preset):
    assert list(preset.dataset_ids) == expand_dataset_preset(preset.name.upper())
    assert len(preset.inspect_tasks) == len(preset.dataset_ids)
    for asset, mapping in zip(preset.dataset_ids, preset.inspect_tasks, strict=True):
        task = mapping.task_name.removeprefix("llm_behavior_eval/")
        args = dict(mapping.task_args)
        if task in {"bbq", "bloom", "unqover"}:
            kind = "ambiguous-bias" if args["kind"] == "ambiguous" else args["kind"]
            assert asset == f"hirundo-io/{task}-{args['bias_type']}-{kind}-free-text"
            assert mapping.dataset_id == asset
        else:
            expected = {
                "halueval": "hirundo-io/halueval",
                "medhallu": "hirundo-io/medhallu",
                "prompt_injection": "hirundo-io/prompt-injection-purple-llama",
                "ccpc": "hirundo-io/ccpc-500",
                "xstest": "hirundo-io/XSTest",
                "orbench": "hirundo-io/or-bench",
            }
            assert asset == expected[task]
            assert mapping.dataset_id == (
                "chinese_censorship" if task == "ccpc" else asset
            )
            assert args == {}
        if task != "prompt_injection":  # Purple Llama integration is pending.
            constructor = getattr(
                importlib.import_module(f"llm_behavior_eval.{task}.task"), task
            )
            inspect.signature(constructor).bind(**args)


def test_catalog_keeps_two_argument_constructor():
    from llm_behavior_eval import DatasetPreset

    preset = DatasetPreset("hallu", ("hirundo-io/halueval",))
    assert preset.inspect_tasks[0].task_name == "llm_behavior_eval/halueval"
