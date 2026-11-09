"""Opt-in legacy text preprocessing: legacy equivalence, opt-in wiring, parity configs.

CPU only. The tiny tokenizer runs everywhere; the pinned Gemma 3 and Qwen3
snapshots are optional and skip when absent. None of this is GPU evidence.
"""

import importlib
import inspect
import json
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import GenerateConfig, ModelOutput, get_model

from llm_behavior_eval import _registry
from llm_behavior_eval.common.parity import parity_config_path
from llm_behavior_eval.common.preprocessing import (
    LEGACY_PROMPT_MAX_LENGTH,
    LEGACY_TEXT_MAX_LENGTH,
    METADATA_KEY,
    load_tokenizer,
    render_chat,
    roundtrip,
)
from tests.legacy_tokenizers import pinned_tokenizer, tiny_tokenizer
from tests.native_fixtures import local_task_dataset

BENCHMARKS = {
    # name: legacy FAMILY_TOKEN_DEFAULTS key
    "bbq": "bias",
    "unqover": "bias",
    "bloom": "bias",
    "halueval": "hallucination",
    "medhallu": "hallucination",
    "prompt_injection": "prompt-injection",
    "ccpc": "censorship",
    "xstest": "refusal",
    "orbench": "refusal",
}
GRADER_REPLY = (
    "A Yes gt_agree: yes candidate_uncertain: no True 2_full_refusal [[direct_refusal]]"
)


@pytest.fixture(params=["tiny", "gemma3", "qwen3"])
def tokenizer_path(request, tmp_path) -> Path:
    if request.param == "tiny":
        return tiny_tokenizer(tmp_path)
    return pinned_tokenizer(request.param)


@pytest.fixture
def tiny(tmp_path) -> Path:
    return tiny_tokenizer(tmp_path)


def legacy_preprocess(tokenizer: Any, thinking: bool, **row: str) -> dict[str, Any]:
    from llm_behavior_eval.evaluation_utils.custom_dataset import (
        free_text_preprocess_function,
    )

    return free_text_preprocess_function(
        {key: [value] for key, value in row.items()},
        tokenizer,
        LEGACY_PROMPT_MAX_LENGTH,
        LEGACY_TEXT_MAX_LENGTH,
        "stereotyped_answer" in row,
        enable_thinking=thinking,
    )


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("long", [False, True])
def test_views_equal_legacy_preprocessing(tokenizer_path, side, thinking, long):
    tokenizer = load_tokenizer(str(tokenizer_path))
    previous = tokenizer.truncation_side
    tokenizer.truncation_side = side
    try:
        repeat = 1400 if long else 1
        row = dict(
            question=" First  question\n" + "long input " * repeat,
            answer=" target " * (500 if long else 1),
            stereotyped_answer=" stereotype " * (500 if long else 1),
            system_prompt="You are helpful.\n",
            judge_question=" Independent question " * (500 if long else 1),
        )
        legacy = legacy_preprocess(tokenizer, thinking, **row)
        rendered = render_chat(
            tokenizer,
            [
                {"role": "system", "content": row["system_prompt"]},
                {"role": "user", "content": row["question"] + "\n"},
            ],
            legacy_thinking=thinking,
        )
        prompt = roundtrip(
            tokenizer,
            rendered,
            max_length=LEGACY_PROMPT_MAX_LENGTH,
            add_special_tokens=True,
        )
        pad = tokenizer.pad_token_id
        expected = [int(t) for t in legacy["test_input_ids"][0] if int(t) != pad]
        assert prompt["input_ids"] == expected
        assert prompt["decoded"] == tokenizer.decode(
            legacy["test_input_ids"][0], skip_special_tokens=True
        )
        if long:
            assert len(prompt["input_ids"]) == LEGACY_PROMPT_MAX_LENGTH
        for key, legacy_key in [
            ("answer", "gt_answers"),
            ("stereotyped_answer", "stereotyped_answers"),
            ("judge_question", "judge_questions"),
        ]:
            view = roundtrip(
                tokenizer,
                row[key],
                max_length=LEGACY_TEXT_MAX_LENGTH,
                add_special_tokens=False,
            )
            assert view["decoded"] == tokenizer.decode(
                legacy[legacy_key][0], skip_special_tokens=True
            )
    finally:
        tokenizer.truncation_side = previous


@pytest.mark.parametrize(
    "marker",
    ["/no_think", "to=self", "reasoning_strength", "System role not supported"],
)
def test_unsupported_legacy_templates_fail_before_rendering(marker):
    class Tokenizer:
        def get_chat_template(self):
            return marker

        def apply_chat_template(self, *args, **kwargs):
            pytest.fail("Unsupported template reached rendering")

    with pytest.raises(ValueError, match="Unsupported legacy chat template"):
        render_chat(
            Tokenizer(), [{"role": "user", "content": "q"}], legacy_thinking=False
        )


def test_open_think_with_thinking_off_is_rejected():
    class Tokenizer:
        def get_chat_template(self):
            return "template"

        def apply_chat_template(self, *args, **kwargs):
            return "<bos>user<think>\n"

    with pytest.raises(ValueError, match="open <think>"):
        render_chat(
            Tokenizer(), [{"role": "user", "content": "q"}], legacy_thinking=False
        )


def test_tokenizer_must_be_local_or_pinned(tmp_path):
    with pytest.raises(ValueError, match="local snapshot"):
        load_tokenizer("google/gemma-3-4b-it")
    with pytest.raises(ValueError, match="immutable"):
        load_tokenizer("google/gemma-3-4b-it", "main")


@pytest.mark.parametrize("name", list(BENCHMARKS))
def test_opt_in_is_a_task_argument_and_off_by_default(name):
    parameters = inspect.signature(getattr(_registry, name)).parameters
    assert parameters["legacy_tokenizer"].default is None
    assert parameters["legacy_tokenizer_revision"].default is None


@pytest.mark.parametrize("name", list(BENCHMARKS))
def test_default_runs_record_no_legacy_views(name, tmp_path, monkeypatch):
    monkeypatch.setenv("INSPECT_BRIEF_ENABLED", "false")
    dataset = local_task_dataset(name, tmp_path, monkeypatch)
    log = inspect_eval(
        f"llm_behavior_eval/{name}",
        task_args={"dataset_path": str(dataset)},
        model="mockllm/answer",
        model_roles={
            "grader": get_model(
                "mockllm/grader",
                custom_outputs=lambda *a: ModelOutput.from_content("m", GRADER_REPLY),
            )
        },
        limit=1,
        display="none",
        log_dir=str(tmp_path / "logs"),
    )[0]
    assert log.status == "success", log.error
    assert log.samples is not None
    assert METADATA_KEY not in log.samples[0].metadata


@pytest.mark.parametrize("name", list(BENCHMARKS))
def test_opt_in_records_views_without_settings_or_input_changes(
    name, tiny, tmp_path, monkeypatch
):
    monkeypatch.setenv("INSPECT_BRIEF_ENABLED", "false")
    dataset = local_task_dataset(name, tmp_path, monkeypatch)
    calls: list[list[str]] = []

    def grade(messages, tools, choice, config):
        assert config == GenerateConfig()
        calls.append([message.text for message in messages])
        return ModelOutput.from_content("mockllm/grader", GRADER_REPLY)

    def answer(messages, tools, choice, config):
        # The opt-in step sets no inference setting for the model under test.
        assert config == GenerateConfig()
        return ModelOutput.from_content("mockllm/answer", "answer")

    log = inspect_eval(
        f"llm_behavior_eval/{name}",
        task_args={"dataset_path": str(dataset), "legacy_tokenizer": str(tiny)},
        model=get_model("mockllm/answer", custom_outputs=answer),
        model_roles={"grader": get_model("mockllm/grader", custom_outputs=grade)},
        limit=1,
        display="none",
        log_dir=str(tmp_path / "logs"),
    )[0]
    assert log.status == "success", log.error
    saved = read_eval_log(log.location)
    assert saved.samples is not None
    sample = saved.samples[0]
    views = sample.metadata[METADATA_KEY]
    assert views["schema"] == "legacy-text-v1"
    assert views["legacy_thinking"] is (BENCHMARKS[name] == "refusal")
    assert views["tokenizer"]["truncation_side"] == "right"
    original = next(
        row
        for row in getattr(_registry, name)(dataset_path=str(dataset)).dataset
        if row.id == sample.id
    )
    assert sample.target == original.target
    if isinstance(sample.input, list):
        assert [(m.role, m.text) for m in sample.input] == [
            (m.role, m.text) for m in original.input
        ]
    else:
        assert sample.input == original.input
    judged = "\n".join(text for call in calls for text in call)
    if name == "ccpc":
        # CCPC's legacy judge saw the original question.
        assert isinstance(sample.input, str)
        assert sample.input in judged
        assert views["prompt"]["decoded"] not in judged
    elif name == "prompt_injection":
        assert views["texts"]["judge_question"]["decoded"] in judged
    else:
        assert views["prompt"]["decoded"] in judged
        assert views["prompt"]["decoded"] != sample.input


def test_hallucination_judge_prompt_matches_legacy(tiny, tmp_path, monkeypatch):
    from llm_behavior_eval.evaluation_utils.free_text_hallu_evaluator import (
        GRADER_TEMPLATE,
    )

    monkeypatch.setenv("INSPECT_BRIEF_ENABLED", "false")
    dataset = local_task_dataset("halueval", tmp_path, monkeypatch)
    calls = []

    def grade(messages, tools, choice, config):
        calls.append(messages[0].text)
        return ModelOutput.from_content("mockllm/grader", "A")

    log = inspect_eval(
        "llm_behavior_eval/halueval",
        task_args={"dataset_path": str(dataset), "legacy_tokenizer": str(tiny)},
        model="mockllm/answer",
        model_roles={"grader": get_model("mockllm/grader", custom_outputs=grade)},
        limit=1,
        display="none",
        log_dir=str(tmp_path / "logs"),
    )[0]
    assert log.status == "success", log.error
    assert log.samples is not None
    views = log.samples[0].metadata[METADATA_KEY]
    completion = log.samples[0].output.completion
    assert calls == [
        GRADER_TEMPLATE.format(
            question=views["prompt"]["decoded"],
            target=views["texts"]["target"]["decoded"],
            predicted_answer=completion,
        )
    ]


@pytest.mark.parametrize("name", list(BENCHMARKS))
def test_parity_config_matches_package_and_legacy_thinking(name):
    from llm_behavior_eval.evaluation_utils.eval_config import FAMILY_TOKEN_DEFAULTS

    path = parity_config_path(f"llm_behavior_eval.{name}")
    config = yaml.safe_load(path.read_text())
    thinking = config["generate_config"]["extra_body"]["chat_template_kwargs"][
        "enable_thinking"
    ]
    solver = importlib.import_module(f"llm_behavior_eval.{name}.solver")
    assert thinking is solver.LEGACY_THINKING
    family = cast("Any", BENCHMARKS[name])
    assert thinking is FAMILY_TOKEN_DEFAULTS[family]["enable_thinking"]
    header = path.read_text().split("\n\n")[0]
    # One parity command: run config, -M provider args and the -T opt-in.
    assert f"--run-config llm_behavior_eval/{name}/parity.yaml" in header
    assert "-T legacy_tokenizer=" in header
    assert f"-M enable_thinking={str(thinking).lower()}" in header
    assert "-M do_sample=false" in header
    assert "tokenizer_call_args='{truncation: true, max_length: 1024}'" in header


def test_run_config_cannot_carry_task_args_without_naming_the_task(tmp_path):
    """Why parity.yaml cannot turn the preprocessing on: -T is required."""
    from inspect_ai._cli.eval import RunConfigInput
    from pydantic import ValidationError

    from llm_behavior_eval.common.parity import read_parity_config

    with pytest.raises(ValidationError):
        RunConfigInput.model_validate({"task": {"args": {"legacy_tokenizer": "x"}}})
    named = tmp_path / "parity.yaml"
    named.write_text(
        json.dumps({"task": {"task": "llm_behavior_eval/bbq", "args": {}}})
    )
    with pytest.raises(ValueError, match="task"):
        read_parity_config(named)


@pytest.mark.parametrize("thinking", [False, True])
def test_qwen3_thinking_render_matches_legacy(thinking):
    from llm_behavior_eval.evaluation_utils.util_functions import (
        safe_apply_chat_template,
    )

    tokenizer = load_tokenizer(str(pinned_tokenizer("qwen3")))
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "question\n"},
    ]
    legacy = safe_apply_chat_template(
        tokenizer, [dict(m) for m in messages], enable_thinking=thinking
    )
    assert render_chat(tokenizer, messages, legacy_thinking=thinking) == legacy
