import json
import math

import pytest

from llm_behavior_eval.inspect_tasks.system_one_censorship import systemone_shim as shim
from llm_behavior_eval.inspect_tasks.system_one_censorship.protocol import (
    GATE_QUESTIONS,
    build_request,
    build_state,
    parse_answers,
)


def test_jev_prompt_matches_model_card_template():
    prompt = shim.jev_prompt(
        "noul", "Parcel arrived damaged.", "Is a refund requested?", ["false", "true"]
    )
    assert prompt == (
        "[kind] noul\n[state] Parcel arrived damaged.\n[question] Is a refund requested?\n"
        "[options]\nfalse\ntrue\n[decision]:"
    )


def test_jev_choice_prompt_labels_options():
    prompt = shim.jev_prompt("choice", "s", "q", ["low", "high"])
    assert prompt.endswith("[options]\nA) low\nB) high\n[decision]:")


def test_jev_state_text_renders_json_unsorted():
    state = {"message": "你好", "channel": "chat"}
    assert shim.jev_state_text(state) == json.dumps(state, ensure_ascii=False)
    assert shim.jev_state_text("plain") == "plain"


@pytest.mark.parametrize(
    ("definition", "kind", "options"),
    [
        ({"type": "noul", "instructions": "q"}, "noul", ["false", "true"]),
        (
            {"type": "score", "criteria": ["None", "Some", "Lots"]},
            "choice",
            ["None", "Some", "Lots"],
        ),
        (
            {"type": "choice", "criteria": {"a": "first", "b": None}},
            "choice",
            ["a: first", "b"],
        ),
    ],
)
def test_jev_plan(definition, kind, options):
    assert shim.jev_plan(definition) == (kind, options)


def test_jev_plan_rejects_unsupported_type():
    with pytest.raises(ValueError, match="unsupported"):
        shim.jev_plan({"type": "text"})


def test_jev_answers_parse_as_system_one():
    probabilities = {
        "noul": [0.3, 0.7],
        "score": [0.1, 0.2, 0.3, 0.2, 0.2],
    }
    answers = {}
    for question in GATE_QUESTIONS:
        definition = question.to_request()
        kind, _ = shim.jev_plan(definition)
        answers[question.id] = shim.jev_answer(
            definition, kind, probabilities[question.kind]
        )
    values = parse_answers({"answers": answers}, list(GATE_QUESTIONS))
    assert values["refuse"] == pytest.approx(0.7)
    assert values["risk"] == pytest.approx(0.2 + 0.6 + 0.6 + 0.8)


def test_jev_backend_applies_bias_and_temperature(tmp_path, monkeypatch):
    (tmp_path / "adapter_vllm").mkdir()
    head = {
        "slots": {"ranges": {"noul": [0, 2], "score": [2, 8], "choice": [8, 24]}},
        "verbalizer_ids": list(range(100, 124)),
        "bias": [0.5, -0.5] + [0.0] * 22,
    }
    (tmp_path / "adapter_vllm" / "decision_head.json").write_text(json.dumps(head))
    calibration = {"per_kind": {"noul": 2.0, "score": 1.0, "choice": 1.0}}
    (tmp_path / "calibration.json").write_text(json.dumps(calibration))
    backend = shim.JevVllmBackend(str(tmp_path), "http://127.0.0.1:1")
    sent = []

    class Response:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps(self.body).encode()

    def fake_urlopen(request, timeout):
        sent.append(json.loads(request.data))
        top = {"token_id:100": math.log(0.25), "token_id:101": math.log(0.75)}
        return Response({"choices": [{"logprobs": {"top_logprobs": [top]}}]})

    monkeypatch.setattr(shim.urllib.request, "urlopen", fake_urlopen)
    request = build_request(build_state("moderation", "hello"), [GATE_QUESTIONS[0]])
    result = backend(request)
    false_logit = (math.log(0.25) + 0.5) / 2.0
    true_logit = (math.log(0.75) - 0.5) / 2.0
    expected = 1 / (1 + math.exp(false_logit - true_logit))
    assert result["answers"]["refuse"]["noul"] == pytest.approx(expected)
    assert sent[0]["model"] == shim.JEV_LORA
    assert sent[0]["allowed_token_ids"] == [100, 101]
    assert sent[0]["prompt"].startswith("[kind] noul\n[state] {")


def test_jev_backend_rejects_non_http_upstream(tmp_path):
    with pytest.raises(ValueError, match="http"):
        shim.JevVllmBackend(str(tmp_path), "file:///x")


def test_agentjev_request_mirrors_client_defaults():
    request = build_request(build_state("moderation", "hi"), list(GATE_QUESTIONS))
    body = shim.agentjev_request(request)
    assert body["state"] == json.dumps(request["state"], ensure_ascii=False, indent=2)
    by_id = {question["id"]: question for question in body["questions"]}
    assert by_id["refuse"]["type"] == "boolean"
    assert by_id["refuse"]["criteria"] == shim.AGENTJEV_BOOLEAN_CRITERIA
    assert by_id["risk"]["levels"] == list(GATE_QUESTIONS[-1].levels)


def test_agentjev_answers_parse_as_system_one():
    result = {
        "results": [
            {
                "answers": [
                    {
                        "id": "refuse",
                        "type": "boolean",
                        "distribution": {"true": 0.8, "false": 0.2},
                    },
                    {
                        "id": "risk",
                        "type": "score",
                        "score": 2.5,
                        "distribution": {"0": 1.0},
                    },
                ]
            }
        ]
    }
    answers = shim.agentjev_answers(result)
    questions = [GATE_QUESTIONS[0], GATE_QUESTIONS[-1]]
    values = parse_answers({"answers": answers}, questions)
    assert values == {"refuse": 0.8, "risk": 2.5}
