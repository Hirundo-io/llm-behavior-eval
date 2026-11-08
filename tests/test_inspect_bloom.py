import importlib
from typing import Any, cast

import pytest
from datasets import Dataset
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

from llm_behavior_eval.bloom.task import bloom


@pytest.fixture
def source(monkeypatch):
    module = importlib.import_module("llm_behavior_eval.bloom.dataset")

    def install(rows):
        monkeypatch.setattr(
            module, "load_hf_dataset", lambda *a, **k: Dataset.from_list(rows)
        )

    return install


@pytest.mark.parametrize("kind", ["bias", "unbias", "ambiguous"])
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
        bloom(kind=kind),
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
    from llm_behavior_eval.bloom.dataset import BIAS_TYPES, dataset

    source([dict(question="q", answer="a")] * 502)
    for kind in ("bias", "unbias"):
        for bias_type in BIAS_TYPES:
            data = dataset(bias_type, kind)
            assert len(data) == 500
            expected = Dataset.from_list([{"i": i} for i in range(502)]).shuffle(
                seed=42
            )[0]["i"]
            assert (
                data[0].id
                == f"hirundo-io/bloom-{bias_type}-{kind}-free-text:{expected}"
            )
    for cap in (0, -1, None):
        assert len(dataset("gender", "bias", max_samples=cap)) == 502
    full = dataset("age", "ambiguous", max_samples=0)
    capped = dataset("age", "ambiguous", max_samples=3)
    assert [s.id for s in capped] == [s.id for s in full[:3]]
    assert [s.id for s in full] != [
        s.id for s in dataset("age", "ambiguous", max_samples=0, seed=43)
    ]
    source([dict(question="q", answer="a")] * 500 + [dict(question="q", answer=None)])
    with pytest.raises(ValueError, match="required text"):
        dataset("gender", "bias")


def test_missing_grader(tmp_path, source):
    source([dict(question="q", answer="a")])
    logs = inspect_eval(
        bloom(), model="mockllm/model", display="none", log_dir=str(tmp_path)
    )
    assert logs[0].status == "error"
    assert logs[0].error
    assert "grader" in logs[0].error.message


def test_offline_snapshot(tmp_path):
    from llm_behavior_eval.bloom.dataset import dataset

    Dataset.from_list(
        [dict(question="q", answer="a", stereotyped_answer="s")]
    ).save_to_disk(tmp_path / "data")
    data = dataset("race", "unbias", dataset_path=str(tmp_path / "data"))
    assert (data[0].metadata or {})["has_stereotype"]
    assert data[0].id == "hirundo-io/bloom-race-unbias-free-text:0"


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
        bloom(externally_configured_thinking=True),
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
        bloom(),
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
            "llm_behavior_eval/bloom",
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


@pytest.mark.parametrize("bias_type", ["age", "gender", "race"])
@pytest.mark.parametrize("kind", ["bias", "unbias", "ambiguous"])
def test_all_split_identities_match_current_presets(monkeypatch, bias_type, kind):
    from llm_behavior_eval.bloom.dataset import dataset
    from llm_behavior_eval.presets import expand_dataset_preset

    preset = (
        f"bloom:bias:{bias_type}:ambiguous"
        if kind == "ambiguous"
        else f"bloom:{kind}:{bias_type}"
    )
    expected = expand_dataset_preset(preset)[0]
    calls = []

    def load(identity, **kwargs):
        calls.append((identity, kwargs))
        return Dataset.from_list([dict(question="q", answer="a")])

    monkeypatch.setattr("llm_behavior_eval.bloom.dataset.load_hf_dataset", load)
    data = dataset(bias_type, kind, dataset_path="offline.json", dataset_token="dummy")
    assert calls == [
        (expected, dict(dataset_path="offline.json", dataset_token="dummy"))
    ]
    assert data.name == expected
    assert data[0].id == f"{expected}:0"
    assert data[0].metadata == dict(
        dataset_id=expected,
        bias_type=bias_type,
        kind=kind,
        has_stereotype=False,
        stereotyped_answer=None,
    )


@pytest.mark.parametrize("kind", ["bias", "unbias"])
def test_all_expansion_matches_legacy(source, kind):
    from llm_behavior_eval.bloom.dataset import BIAS_TYPES, dataset
    from llm_behavior_eval.evaluation_utils.enums import BLOOM_BIAS_TYPES
    from llm_behavior_eval.presets import expand_dataset_preset

    source([dict(question="q", answer="a")])
    assert set(BIAS_TYPES) == BLOOM_BIAS_TYPES
    assert [dataset(t, kind).name for t in BIAS_TYPES] == expand_dataset_preset(
        f"BLOOM:{kind.upper()}:ALL"
    )


@pytest.mark.parametrize(
    "bias_type,kind",
    [
        ("religion", "bias"),
        ("age", "neutral"),
        ("all", "bias"),
        ("gender", "ambiguous-unbias"),
        ("gender", "unbias:ambiguous"),
        ("all", "ambiguous"),
    ],
)
def test_invalid_parameters_fail_before_loading(monkeypatch, bias_type, kind):
    def load(*args, **kwargs):
        raise AssertionError("Invalid selection loaded a dataset")

    monkeypatch.setattr("llm_behavior_eval.bloom.dataset.load_hf_dataset", load)
    with pytest.raises(ValueError, match="BLOOM"):
        bloom(bias_type=bias_type, kind=kind)


@pytest.mark.parametrize("column", ["question", "answer", "stereotyped_answer"])
def test_validation_before_cap(source, column):
    rows: list[dict[str, str | None]] = [
        dict(question="q", answer="a", stereotyped_answer="s") for _ in range(501)
    ]
    rows[-1][column] = None
    source(rows)
    with pytest.raises(ValueError, match="required text"):
        bloom()
