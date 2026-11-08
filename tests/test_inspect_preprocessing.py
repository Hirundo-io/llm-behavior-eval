"""CPU production boundary tests; never substitute log input before scoring."""

import hashlib
import os
from pathlib import Path

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput, get_model

from llm_behavior_eval import _registry
from llm_behavior_eval.common.preprocessing import load_tokenizer, preprocess
from llm_behavior_eval.evaluation_utils.custom_dataset import (
    free_text_preprocess_function,
)
from tests.native_fixtures import local_task_dataset


@pytest.fixture
def tokenizer():
    path = Path(
        os.environ.get("LLM169_TOKENIZER", "results/tokenizers/gemma3")
    ).resolve()
    if not path.is_dir():
        pytest.skip(
            "Set LLM169_TOKENIZER to pinned Gemma 3 snapshot (see parity README)"
        )
    assert (
        hashlib.sha256((path / "tokenizer.json").read_bytes()).hexdigest()
        == "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795"
    )
    return load_tokenizer(str(path), None)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("thinking", [False, True])
def test_long_legacy_roundtrips(tokenizer, side, thinking):
    previous = tokenizer.truncation_side
    tokenizer.truncation_side = side
    try:
        question = " First  question\n" + "long input " * 1400
        target = " target " * 500
        stereo = " stereotype " * 500
        independent = " Independent question " * 500
        system = "You are helpful.\n"
        legacy = free_text_preprocess_function(
            dict(
                question=[question],
                answer=[target],
                stereotyped_answer=[stereo],
                system_prompt=[system],
                judge_question=[independent],
            ),
            tokenizer,
            1024,
            256,
            True,
            enable_thinking=thinking,
        )
        result = preprocess(
            tokenizer,
            [
                dict(role="system", content=system),
                dict(role="user", content=question + "\n"),
            ],
            target,
            stereotype=stereo,
            question=independent,
            thinking=thinking,
        )
        for key, old in [
            ("prompt", "test_input_ids"),
            ("target", "gt_answers"),
            ("stereotype", "stereotyped_answers"),
        ]:
            assert result[key]["input_ids"] == legacy[old][0].tolist()
            assert result[key]["decoded"] == tokenizer.decode(
                legacy[old][0], skip_special_tokens=True
            )
        assert result["judge_question"] == tokenizer.decode(
            legacy["judge_questions"][0], skip_special_tokens=True
        )
    finally:
        tokenizer.truncation_side = previous


@pytest.mark.parametrize(
    "name",
    [
        "bbq",
        "bloom",
        "unqover",
        "halueval",
        "medhallu",
        "orbench",
        "xstest",
        "ccpc",
        "prompt_injection",
    ],
)
def test_registered_production_metadata(name, tmp_path, monkeypatch, tokenizer):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("INSPECT_BRIEF_ENABLED", "false")
    dataset = local_task_dataset(name, tmp_path, monkeypatch)
    calls = []

    def grade(messages, tools, choice, config):
        calls.append(messages)
        return ModelOutput.from_content(
            "mockllm/grader",
            "A Yes gt_agree: yes candidate_uncertain: no True 2_full_refusal [[direct_refusal]]",
        )

    def answer(messages, tools, choice, config):
        assert set(config.extra_body) == {
            "truncate_prompt_tokens",
            "truncation_side",
            "add_special_tokens",
            "return_token_ids",
        }
        assert config.extra_body["truncate_prompt_tokens"] == 1024
        assert config.extra_body["truncation_side"] == tokenizer.truncation_side
        assert config.extra_body["add_special_tokens"] is True
        assert config.extra_body["return_token_ids"] is True
        assert not messages[-1].text.startswith("<bos>")
        return ModelOutput.from_content("mockllm/answer", "answer")

    log = inspect_eval(
        f"llm_behavior_eval/{name}",
        task_args=dict(
            dataset_path=str(dataset), compatibility_tokenizer=tokenizer.name_or_path
        ),
        model=get_model(
            "mockllm/answer",
            custom_outputs=answer,
        ),
        model_roles={"grader": get_model("mockllm/grader", custom_outputs=grade)},
        limit=1,
        display="none",
        log_dir=str(tmp_path / "logs"),
    )[0]
    assert log.status == "success", log.error
    saved = read_eval_log(log.location)
    sample = saved.samples[0]
    meta = sample.metadata["compatibility"]
    assert meta["schema"] == "legacy-text-v1"
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
    assert any(
        meta["judge_question"] in message.text for call in calls for message in call
    )
    if isinstance(sample.input, str) and name != "ccpc":
        assert sample.input != meta["judge_question"]
    assert saved.results.total_samples == 1

    if name in {"halueval", "medhallu"}:
        from llm_behavior_eval.evaluation_utils.free_text_hallu_evaluator import (
            GRADER_TEMPLATE,
        )

        assert calls[0][0].text == GRADER_TEMPLATE.format(
            question=meta["judge_question"],
            target=meta["target"]["decoded"],
            predicted_answer="answer",
        )


@pytest.mark.parametrize("thinking", [False, True])
def test_qwen_thinking_roundtrip(thinking):
    from llm_behavior_eval.evaluation_utils.util_functions import (
        safe_apply_chat_template,
    )

    path = Path("results/tokenizers/qwen3").resolve()
    if not path.is_dir():
        pytest.skip("Pinned Qwen3 tokenizer fixture absent; see parity README")
    assert (
        hashlib.sha256((path / "tokenizer.json").read_bytes()).hexdigest()
        == "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
    )
    tokenizer = load_tokenizer(str(path), None)
    messages = [
        dict(role="system", content="Be helpful."),
        dict(role="user", content="question\n"),
    ]
    rendered = safe_apply_chat_template(tokenizer, messages, enable_thinking=thinking)
    expected = tokenizer(
        [rendered], truncation=True, padding="max_length", max_length=1024
    )
    actual = preprocess(tokenizer, messages, "target", thinking=thinking)
    assert actual["prompt"]["input_ids"] == expected["input_ids"][0]
    assert actual["judge_question"] == tokenizer.decode(
        expected["input_ids"][0], skip_special_tokens=True
    )


def test_judge_controls_never_apply_answer_cap(monkeypatch):
    from types import SimpleNamespace

    from inspect_ai.model import GenerateConfig

    import llm_behavior_eval.common.preprocessing as module

    class Vllm:
        def __str__(self):
            return "vllm/fixture"

    monkeypatch.setattr(module, "get_model", lambda **kwargs: Vllm())
    state = SimpleNamespace(metadata={"compatibility": {}})
    original = GenerateConfig(
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "custom": 1}
    )
    config = module.judge_config(state, original)
    assert config.extra_body == {
        "chat_template_kwargs": {"enable_thinking": False},
        "custom": 1,
        "add_special_tokens": True,
        "return_token_ids": True,
    }
    assert original.extra_body == {
        "chat_template_kwargs": {"enable_thinking": False},
        "custom": 1,
    }
    with pytest.raises(ValueError, match="must not"):
        module.judge_config(
            state, GenerateConfig(extra_body={"truncate_prompt_tokens": 1024})
        )


def test_gemma_system_role_demotion_is_template_equivalent(tokenizer):
    messages = [
        dict(role="system", content="Policy\n"),
        dict(role="user", content="Question\n"),
    ]
    demoted = [dict(role="user", content="Policy\n\n\nQuestion\n")]
    assert tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) == tokenizer.apply_chat_template(
        demoted, tokenize=False, add_generation_prompt=True
    )
