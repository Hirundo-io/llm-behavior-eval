"""Real native HF provider with CPU weight stub: boundary evidence, not GPU parity."""

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ModelName,
    get_model,
)
from inspect_ai.solver import TaskState
from transformers import AutoModelForCausalLM

from llm_behavior_eval.common.preprocessing import (
    load_tokenizer,
    preprocess,
    validate_hf,
)

CAP = dict(truncation=True, max_length=1024, add_special_tokens=True)


@pytest.fixture(
    params=[
        ("gemma3", "4667f2089529e8e7657cfb6d1c19910ae71ff5f28aa7ab2ff2763330affad795"),
        ("qwen3", "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    ]
)
def hf(request, monkeypatch):
    name, digest = request.param
    path = Path("results/tokenizers") / name
    if not path.is_dir():
        pytest.skip("Pinned tokenizer fixture absent")
    assert hashlib.sha256((path / "tokenizer.json").read_bytes()).hexdigest() == digest
    captured = []

    class Weights:
        device = "cpu"
        config = SimpleNamespace(model_type=name)

        def generate(self, input_ids, attention_mask, **kwargs):
            captured.append((input_ids.tolist(), attention_mask.tolist()))
            return SimpleNamespace(
                sequences=torch.cat([input_ids, input_ids[:, -1:]], dim=1),
                logits=None,
                hidden_states=None,
            )

    monkeypatch.setattr(
        AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: Weights()
    )
    model = get_model(
        "hf/local",
        model_path=str(path.resolve()),
        device="cpu",
        enable_thinking=False,
        do_sample=False,
        tokenizer_call_args=dict(CAP),
        memoize=False,
    )
    return model, load_tokenizer(str(path.resolve()), None), captured


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("long", [False, True])
def test_real_hf_generation_boundary(hf, monkeypatch, side, thinking, long):
    model, tokenizer, captured = hf
    monkeypatch.setattr(tokenizer, "truncation_side", side)
    monkeypatch.setattr(model.api.tokenizer, "truncation_side", side)
    monkeypatch.setattr(model.api, "enable_thinking", thinking)
    validate_hf(model.api, tokenizer)
    messages = [
        ChatMessageSystem(content="Be helpful."),
        ChatMessageUser(content="question " * (1600 if long else 1) + "\n"),
    ]
    expected = preprocess(
        tokenizer,
        [dict(role=m.role, content=m.text) for m in messages],
        "target",
        thinking=thinking,
    )
    asyncio.run(
        model.api.generate(
            messages, [], "none", GenerateConfig(max_tokens=1, max_connections=1)
        )
    )
    ids, masks = captured[0]
    active = [i for i, m in zip(ids[0], masks[0], strict=True) if m]
    oracle = expected["prompt"]
    assert active == [
        i
        for i, m in zip(oracle["input_ids"], oracle["attention_mask"], strict=True)
        if m
    ]
    assert (
        tokenizer.decode(ids[0], skip_special_tokens=True) == expected["judge_question"]
    )
    assert len(ids[0]) == 1024 if long else len(ids[0]) < 1024


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"tokenizer_call_args": {}}, "requires tokenizer_call_args"),
        ({"tokenizer_call_args": dict(CAP, padding="max_length")}, "omit padding"),
        ({"use_chat_template": False}, "use_chat_template"),
        ({"chat_template": "different"}, "override differs"),
        ({"enable_thinking": None}, "explicit enable_thinking"),
    ],
)
def test_reject_incompatible_provider_settings(hf, monkeypatch, changes, match):
    model, tokenizer, _ = hf
    for key, value in changes.items():
        monkeypatch.setattr(model.api, key, value)
    with pytest.raises(ValueError, match=match):
        validate_hf(model.api, tokenizer)


def test_identity_uses_content_not_path(hf):
    model, tokenizer, _ = hf
    model.api.tokenizer.add_tokens(["unexpected-extra-token"])
    with pytest.raises(ValueError, match="tokenizer differs"):
        validate_hf(model.api, tokenizer)


def test_judge_has_no_answer_cap(hf, monkeypatch):
    model, _, _ = hf
    with pytest.raises(ValueError, match="judge must not"):
        validate_hf(model.api)
    monkeypatch.setattr(model.api, "tokenizer_call_args", {})
    validate_hf(model.api)
    assert model.api.tokenizer_call_args == {}
    import llm_behavior_eval.common.preprocessing as module

    monkeypatch.setattr(module, "get_model", lambda **kwargs: model)
    config = GenerateConfig(max_tokens=32, extra_body={"caller": "preserved"})
    state = TaskState(
        model=ModelName("hf/local"),
        sample_id=1,
        epoch=1,
        input="question",
        messages=[],
        metadata={"compatibility": {}},
    )
    assert module.judge_config(state, config) is config
    assert model.api.enable_thinking is False


def test_refusal_hf_metadata_and_original_log(hf, tmp_path, monkeypatch):
    from inspect_ai import eval as inspect_eval

    from llm_behavior_eval.xstest.task import xstest
    from tests.native_fixtures import local_task_dataset

    model, tokenizer, captured = hf
    monkeypatch.setattr(model.api, "enable_thinking", True)
    monkeypatch.setenv("INSPECT_BRIEF_ENABLED", "false")
    dataset = local_task_dataset("xstest", tmp_path, monkeypatch)
    task = xstest(
        dataset_path=str(dataset), compatibility_tokenizer=tokenizer.name_or_path
    )
    original = task.dataset[0]
    log = inspect_eval(
        task,
        model=model,
        model_roles={"grader": "mockllm/grader"},
        limit=1,
        display="none",
        log_dir=str(tmp_path / "logs"),
    )[0]
    assert log.status == "success", log.error
    assert captured
    assert log.samples is not None
    sample = log.samples[0]
    assert sample.input == original.input
    assert sample.target == original.target
    assert sample.metadata["compatibility"]["thinking"] is True


def test_mixed_batch_padding_and_legacy_targets(hf):
    from llm_behavior_eval.evaluation_utils.custom_dataset import (
        free_text_preprocess_function,
    )

    model, tokenizer, captured = hf
    questions = ["short", "long " * 1600]
    targets: list[str] = ["target " * 500] * 2
    stereotypes: list[str] = ["stereotype " * 500] * 2
    independent: list[str] = ["independent " * 500] * 2
    legacy = free_text_preprocess_function(
        dict(
            question=questions,
            answer=targets,
            stereotyped_answer=stereotypes,
            system_prompt=["Be helpful."] * 2,
            judge_question=independent,
        ),
        tokenizer,
        1024,
        256,
        True,
        enable_thinking=False,
    )
    messages = [
        [ChatMessageSystem(content="Be helpful."), ChatMessageUser(content=q + "\n")]
        for q in questions
    ]

    async def run():
        await asyncio.gather(
            *(
                model.api.generate(
                    m, [], "none", GenerateConfig(max_tokens=1, max_connections=2)
                )
                for m in messages
            )
        )

    asyncio.run(run())
    assert len(captured) == 1
    ids, masks = captured[0]
    for index, message in enumerate(messages):
        result = preprocess(
            tokenizer,
            [dict(role=m.role, content=m.text) for m in message],
            targets[index],
            stereotype=stereotypes[index],
            question=independent[index],
        )
        for key, old in [
            ("prompt", "test_input_ids"),
            ("target", "gt_answers"),
            ("stereotype", "stereotyped_answers"),
        ]:
            assert result[key]["input_ids"] == legacy[old][index].tolist()
        assert result["judge_question"] == tokenizer.decode(
            legacy["judge_questions"][index], skip_special_tokens=True
        )
        active = [i for i, mask in zip(ids[index], masks[index], strict=True) if mask]
        prompt = result["prompt"]
        assert active == [
            i
            for i, mask in zip(
                prompt["input_ids"], prompt["attention_mask"], strict=True
            )
            if mask
        ]
    assert masks[0][0] == 0
    assert ids[0][0] == model.api.tokenizer.eos_token_id
    assert masks[1][0] == 1
