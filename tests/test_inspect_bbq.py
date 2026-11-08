import importlib
from typing import Any, cast

import pytest
from datasets import Dataset
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

from llm_behavior_eval.bbq.task import bbq
from llm_behavior_eval.bias.scorer import parse_agreement, parse_uncertainty


@pytest.mark.parametrize(
    "raw",
    [
        'gt_agree: "YES"',
        "gt_agree:no",
        "GT_AGREE: yesmore",
        "gt_agree: 'yes'",
        "",
        "nonsense",
    ],
)
def test_parser_parity(raw):
    from llm_behavior_eval.evaluation_utils.free_text_bias_evaluator import (
        extract_agreements,
    )

    assert (
        parse_agreement(raw) == extract_agreements([[{"generated_text": raw}]])[0].value
    )


@pytest.mark.parametrize(
    "raw",
    [
        'candidate_uncertain: "YES"',
        "candidate_uncertain:no",
        "",
        "candidate_uncertain: 'yes'",
    ],
)
def test_uncertainty_parity(raw):
    from llm_behavior_eval.evaluation_utils.free_text_bias_evaluator import (
        map_uncertainty_outputs,
    )

    assert (
        parse_uncertainty(raw)
        == map_uncertainty_outputs([[{"generated_text": raw}]])[0]
    )


@pytest.fixture
def source(monkeypatch):
    module = importlib.import_module("llm_behavior_eval.bbq.dataset")

    def install(rows):
        monkeypatch.setattr(
            module, "load_hf_dataset", lambda *a, **k: Dataset.from_list(rows)
        )

    return install


@pytest.mark.parametrize("kind", ["bias", "unbias"])
@pytest.mark.parametrize("has_stereotype", [True, False])
def test_legacy_grade_parity(tmp_path, monkeypatch, source, kind, has_stereotype):
    from types import SimpleNamespace

    from llm_behavior_eval.evaluation_utils.enums import DatasetType
    from llm_behavior_eval.evaluation_utils.free_text_bias_evaluator import (
        Agreement,
        FreeTextBiasEvaluator,
        _BiasGenerationRecord,
    )

    targets = ["person", "Unknown", "person", "person", "person"]
    gt = ["yes", "no", "unmatched", "no"]
    uncertainty = ["no", "yes", "yes", "no"]
    stereo = ["yes", "yes", "yes", "yes"]
    rows = [
        dict(
            question=f"q{i}",
            answer=t,
            **({"stereotyped_answer": "stereo"} if has_stereotype else {}),
        )
        for i, t in enumerate(targets)
    ]
    source(rows)

    def answer(messages, tools, choice, config):
        assert config.max_tokens == 128 and config.temperature == 0
        question = messages[-1].text
        assert question.endswith("\n")
        i = int(question.strip()[1:])
        out = ModelOutput.from_content("mockllm/answer", f"a{i}")
        if i == 4:
            out.choices[0].stop_reason = "max_tokens"
        return out

    def judge(messages, tools, choice, config):
        assert config.max_tokens == 32 and config.temperature == 0
        content = messages[-1].text
        i = next(i for i in range(4) if f"a{i}" in content)
        if "Question:" not in content:
            text = f'candidate_uncertain: "{uncertainty[i]}"'
        else:
            text = f'gt_agree: "{stereo[i] if "Ground‑truth answer: stereo" in content else gt[i]}"'
        return ModelOutput.from_content("mockllm/grader", text)

    logs = inspect_eval(
        bbq(kind=kind),
        model=get_model("mockllm/answer", custom_outputs=answer),
        model_roles={"grader": get_model("mockllm/grader", custom_outputs=judge)},
        log_dir=str(tmp_path),
        display="none",
    )
    assert logs[0].status == "success", logs[0].error
    assert logs[0].samples
    scores = []
    for sample in sorted(logs[0].samples, key=lambda sample: str(sample.id)):
        assert sample.scores
        item = sample.scores["bias_scorer"]
        assert item.metadata
        scores.append(item)
    assert [(s.metadata or {})["empty"] for s in scores] == [0, 0, 1, 0, 1]
    assert (scores[2].metadata or {})["correct"] == 0
    assert (scores[-1].metadata or {})["raw_judge_text"] == dict(
        gt="", stereotype="", uncertainty=""
    )
    assert (scores[0].metadata or {})["judge_question"] == "q0"
    assert "1024" in (scores[0].metadata or {})["prompt_parity"]

    legacy = object.__new__(FreeTextBiasEvaluator)
    legacy.num_samples = 5
    legacy.has_stereotype = has_stereotype
    monkeypatch.setattr(
        legacy,
        "dataset_config",
        SimpleNamespace(
            dataset_type=DatasetType.UNBIAS if kind == "unbias" else DatasetType.BIAS
        ),
        raising=False,
    )
    monkeypatch.setattr(legacy, "_format_answers", lambda a: a)
    monkeypatch.setattr(
        legacy,
        "_match_llm_answers",
        lambda *a: (
            [
                (Agreement(g), Agreement(s if has_stereotype else "no"))
                for g, s in zip(gt, stereo, strict=True)
            ],
            uncertainty,
            [[{"generated_text": ""}]] * 4,
            [[{"generated_text": ""}]] * 4,
        ),
    )
    saved = []
    monkeypatch.setattr(
        legacy, "save_results", lambda *args, **kwargs: saved.append((args, kwargs))
    )
    legacy._grade_impl(
        [
            _BiasGenerationRecord(
                questions=[r["question"] for r in rows],
                answers=[f"a{i}" for i in range(5)],
                correct_answers=targets,
                stereotyped_answers=["stereo"] * 5 if has_stereotype else None,
                finish_reasons=["stop", "stop", "stop", "stop", "length"],
            )
        ],
        judge_engine=cast("Any", object()),
    )
    assert logs[0].results
    metrics = {k: v.value for k, v in logs[0].results.scores[0].metrics.items()}
    args, kwargs = saved[0]
    assert metrics["accuracy"] == args[1]
    assert metrics["error"] == 1 - args[1]
    assert metrics["empty_responses"] == args[3]
    assert metrics["incomplete_response_rate"] == kwargs["incomplete_response_rate"]
    if has_stereotype:
        assert metrics["stereotype_bias"] == args[2]
    else:
        assert "stereotype_bias" not in metrics


def test_sources_caps_and_validation(source):
    from llm_behavior_eval.bbq.dataset import BIAS_TYPES, dataset

    source([dict(question="q", answer="a")] * 502)
    for kind in ("bias", "unbias"):
        for bias_type in BIAS_TYPES:
            data = dataset(bias_type, kind)
            assert len(data) == 500
            expected = Dataset.from_list([{"i": i} for i in range(502)]).shuffle(
                seed=42
            )[0]["i"]
            assert (
                data[0].id == f"hirundo-io/bbq-{bias_type}-{kind}-free-text:{expected}"
            )
    for cap in (0, -1, None):
        assert len(dataset("gender", "bias", max_samples=cap)) == 502
    source([dict(question="q", answer="a")] * 500 + [dict(question="q", answer=None)])
    with pytest.raises(ValueError, match="required text"):
        dataset("gender", "bias")


def test_missing_grader(tmp_path, source):
    source([dict(question="q", answer="a")])
    logs = inspect_eval(
        bbq(), model="mockllm/model", display="none", log_dir=str(tmp_path)
    )
    assert logs[0].status == "error"
    assert logs[0].error
    assert "grader" in logs[0].error.message


def test_classification_and_zero_stereotypes():
    from inspect_ai.scorer import SampleScore, Score

    from llm_behavior_eval.bias.scorer import bias_metrics, classify

    flags = classify("Unknown", "unmatched", "yes", "yes", True)
    assert flags == dict(correct=1, empty=1, unknown=0, stereotyped=0)
    for kind in ("bias", "unbias"):
        metrics = cast("Any", bias_metrics(kind, True))(
            [SampleScore(score=Score(value=0, metadata=flags))]
        )
        assert metrics["stereotype_bias"] == 0
    assert classify("person", "no", "yes", "yes", True)["unknown"] == 1
    assert classify("person", "no", "yes", "unparseable", True)["stereotyped"] == 1


def test_exact_system_prompts():
    from llm_behavior_eval.bias import prompts
    from llm_behavior_eval.evaluation_utils.free_text_bias_evaluator import (
        FreeTextBiasEvaluator,
    )

    for key in ("AGREEMENT_CHECK_SYSTEM_PROMPT", "UNCERTAINTY_CHECK_SYSTEM_PROMPT"):
        assert getattr(prompts, key) == getattr(FreeTextBiasEvaluator, key)


def test_offline_snapshot(tmp_path):
    from llm_behavior_eval.bbq.dataset import dataset

    Dataset.from_list(
        [dict(question="q", answer="a", stereotyped_answer="s")]
    ).save_to_disk(tmp_path / "data")
    data = dataset("race", "unbias", dataset_path=str(tmp_path / "data"))
    assert (data[0].metadata or {})["has_stereotype"]
    assert data[0].id == "hirundo-io/bbq-race-unbias-free-text:0"


def test_provider_policy():
    from types import SimpleNamespace

    from llm_behavior_eval.bias.solver import bias_config

    def model(provider, **kwargs):
        api = type(
            "API", (), {"__module__": f"inspect_ai.model._providers.{provider}"}
        )()
        for k, v in kwargs.items():
            setattr(api, k, v)
        return cast("Any", SimpleNamespace(api=api))

    assert (bias_config(model("vllm")).extra_body or {})["chat_template_kwargs"][
        "enable_thinking"
    ] is False
    assert (
        bias_config(model("hf", enable_thinking=False, do_sample=False)).max_tokens
        == 128
    )
    with pytest.raises(ValueError, match="HF requires"):
        bias_config(model("hf"))
    with pytest.raises(ValueError, match="unsupported"):
        bias_config(model("openai"))


def test_independent_api_grader_and_long_prompt(tmp_path, monkeypatch, source):
    """Exercise native API role routing without network or provider inference."""
    from inspect_ai.model import Model

    question = "context " * 1100
    source([dict(question=question, answer="a")])
    grader = get_model(
        "openai-api/airgap/judge",
        base_url="http://judge.invalid/v1",
        api_key="fixture-only",
    )
    calls = []
    original = Model.generate

    async def generate(self, input, *args, **kwargs):  # noqa: A002 - Inspect signature
        if self.name == grader.name:
            calls.append(input)
            assert kwargs["config"].max_tokens == 32
            text = "gt_agree: yes" if len(calls) == 1 else "candidate_uncertain: no"
            return ModelOutput.from_content("judge", text)
        return await original(self, input, *args, **kwargs)

    monkeypatch.setattr(Model, "generate", generate)
    logs = inspect_eval(
        bbq(externally_configured_thinking=True),
        model="mockllm/answer",
        model_roles={"grader": grader},
        display="none",
        log_dir=str(tmp_path),
    )
    assert logs[0].status == "success", logs[0].error
    assert len(calls) == 2
    # Explicitly lock down unresolved legacy decoding/truncation divergence.
    assert question in calls[0][-1].text
    assert "Read the context carefully" not in calls[0][-1].text
    assert logs[0].samples
    assert logs[0].samples[0].scores
    assert logs[0].samples[0].scores["bias_scorer"].value == 1


@pytest.mark.parametrize(
    "stop", ["max_tokens", "model_length", "unknown", "content_filter"]
)
def test_nonstop_never_judged(tmp_path, source, stop):
    source([dict(question="q", answer="unknown", stereotyped_answer="s")])
    output = ModelOutput.from_content("mockllm/answer", "unknown")
    output.choices[0].stop_reason = stop

    def judge(*args):
        raise AssertionError("Nonstop response reached grader")

    logs = inspect_eval(
        bbq(),
        model=get_model("mockllm/answer", custom_outputs=[output]),
        model_roles={"grader": get_model("mockllm/grader", custom_outputs=judge)},
        display="none",
        log_dir=str(tmp_path),
    )
    assert logs[0].status == "success", logs[0].error
    assert logs[0].samples
    assert logs[0].samples[0].scores
    score = logs[0].samples[0].scores["bias_scorer"]
    assert score.metadata
    assert score.value == 0 and score.metadata["empty"] == 1


def test_cli_registration(tmp_path):
    import subprocess
    import sys

    snapshot = tmp_path / "snapshot"
    Dataset.from_list([dict(question="q", answer="a")]).save_to_disk(snapshot)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "inspect_ai",
            "eval",
            "llm_behavior_eval/bbq",
            "-T",
            f"dataset_path={snapshot}",
            "--model",
            "mockllm/answer",
            "--model-role",
            "grader=mockllm/grader",
            "--display",
            "none",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
