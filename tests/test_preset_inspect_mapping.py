"""Every legacy preset maps to registered Inspect tasks and their ``-T`` args."""

import importlib
import inspect
import subprocess
import sys

import pytest

from llm_behavior_eval import (
    DatasetPreset,
    expand_dataset_preset,
    get_dataset_preset,
    list_dataset_presets,
)
from llm_behavior_eval import presets as catalog

SINGLE_TASK_ASSETS = {
    "halueval": "hirundo-io/halueval",
    "medhallu": "hirundo-io/medhallu",
    "prompt_injection": "hirundo-io/prompt-injection-purple-llama",
    "ccpc": "hirundo-io/ccpc-500",
    "xstest": "hirundo-io/XSTest",
    "orbench": "hirundo-io/or-bench",
}
# Generation settings belong to the user (per role) or to parity.yaml.
INFERENCE_SETTINGS = {
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "max_answer_tokens",
    "max_judge_tokens",
    "reasoning_effort",
    "reasoning_tokens",
    "enable_thinking",
    "thinking",
    "do_sample",
    "seed",
}


@pytest.mark.parametrize("preset", list_dataset_presets(), ids=lambda p: p.name)
def test_every_expansion_has_exact_native_identity_and_arguments(preset):
    assert list(preset.dataset_ids) == expand_dataset_preset(preset.name.upper())
    assert len(preset.inspect_tasks) == len(preset.dataset_ids)
    for asset, mapping in zip(preset.dataset_ids, preset.inspect_tasks, strict=True):
        task = mapping.task_name.removeprefix("llm_behavior_eval/")
        args = dict(mapping.task_args)
        assert not INFERENCE_SETTINGS & args.keys()
        if task in {"bbq", "bloom", "unqover"}:
            kind = "ambiguous-bias" if args["kind"] == "ambiguous" else args["kind"]
            assert asset == f"hirundo-io/{task}-{args['bias_type']}-{kind}-free-text"
            assert mapping.dataset_id == asset
        else:
            assert asset == SINGLE_TASK_ASSETS[task]
            assert mapping.dataset_id == (
                "chinese_censorship" if task == "ccpc" else asset
            )
            assert args == {}
        assert mapping.cli_args == tuple(
            part for key, value in args.items() for part in ("-T", f"{key}={value}")
        )
        constructor = getattr(
            importlib.import_module(f"llm_behavior_eval.{task}.task"), task
        )
        inspect.signature(constructor).bind(**args)


@pytest.mark.parametrize("preset", list_dataset_presets(), ids=lambda p: p.name)
def test_mapped_split_is_the_one_the_task_loads(preset, monkeypatch):
    """The task, given the mapped arguments, asks for exactly the preset's asset."""
    for asset, mapping in zip(preset.dataset_ids, preset.inspect_tasks, strict=True):
        task = mapping.task_name.removeprefix("llm_behavior_eval/")
        if task not in {"bbq", "bloom", "unqover"}:
            continue
        module = importlib.import_module(f"llm_behavior_eval.{task}.dataset")
        requested = []

        def load(identity, _requested=requested, **kwargs):
            _requested.append(identity)
            raise LookupError("stop after the identity is chosen")

        monkeypatch.setattr(module, "load_hf_dataset", load)
        with pytest.raises(LookupError):
            module.dataset(**dict(mapping.task_args))
        assert requested == [asset]


@pytest.mark.parametrize(
    "package,types",
    [
        ("bbq", catalog.BBQ_BIAS_TYPES),
        ("unqover", catalog.UNQOVER_BIAS_TYPES),
        ("bloom", catalog.BLOOM_BIAS_TYPES),
    ],
)
def test_catalog_types_match_each_benchmark_package(package, types):
    module = importlib.import_module(f"llm_behavior_eval.{package}.dataset")
    assert set(module.BIAS_TYPES) == types


def test_catalog_matches_the_legacy_catalog_while_it_exists():
    from llm_behavior_eval.evaluation_utils import enums

    assert catalog.BBQ_BIAS_TYPES == enums.BBQ_BIAS_TYPES
    assert catalog.UNQOVER_BIAS_TYPES == enums.UNQOVER_BIAS_TYPES
    assert catalog.BLOOM_BIAS_TYPES == enums.BLOOM_BIAS_TYPES
    assert catalog.BIAS_KINDS == enums.BIAS_KINDS


@pytest.mark.parametrize(
    "spelling,name",
    [
        (" BLOOM : UNBIAS : ALL ", "bloom:unbias:all"),
        ("Refusal:XSTest", "refusal:xstest"),
        ("hallucination", "hallu"),
        ("hallucination-med", "hallu-med"),
        ("CHINESE_CENSORSHIP", "chinese_censorship"),
    ],
)
def test_lookup_normalizes_legacy_spellings(spelling, name):
    assert get_dataset_preset(spelling).name == name


@pytest.mark.parametrize("name", ["bloom:neutral:age", "unqover:unbias:gender", ""])
def test_unknown_presets_are_rejected(name):
    with pytest.raises(ValueError, match="Unknown behavior preset"):
        get_dataset_preset(name)


def test_catalog_keeps_two_argument_constructor():
    preset = DatasetPreset("hallu", ("hirundo-io/halueval",))
    assert preset.inspect_tasks[0].task_name == "llm_behavior_eval/halueval"


def test_catalog_import_is_lightweight():
    script = (
        "import sys; import llm_behavior_eval.presets as p; p.list_dataset_presets(); "
        "heavy = {'datasets', 'torch', 'transformers', 'inspect_ai'} & set(sys.modules); "
        "assert not heavy, heavy; "
        "assert not any(m.startswith('llm_behavior_eval.evaluation_utils') "
        "for m in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
