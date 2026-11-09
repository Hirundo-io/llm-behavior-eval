"""Fixture parity uses native JSON logs; CLI smoke keeps Inspect default logs."""

import importlib

import pytest
from datasets import Dataset
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

from llm_behavior_eval.unqover.task import unqover
from tests.golden import BASELINE as GOLDEN


@pytest.fixture
def source(monkeypatch):
    module = importlib.import_module("llm_behavior_eval.unqover.dataset")

    def install(rows):
        monkeypatch.setattr(
            module, "load_hf_dataset", lambda *a, **k: Dataset.from_list(rows)
        )

    return install


@pytest.mark.parametrize("has_stereotype", [True, False])
def test_legacy_grade_parity(tmp_path, monkeypatch, source, has_stereotype):
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
        unqover(),
        model=get_model("mockllm/answer", custom_outputs=answer),
        model_roles={"grader": get_model("mockllm/grader", custom_outputs=judge)},
        log_dir=str(tmp_path),
        log_format="json",
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

    args, kwargs = GOLDEN["unqover"][str(has_stereotype)]["bias"]
    assert logs[0].results
    metrics = {k: v.value for k, v in logs[0].results.scores[0].metrics.items()}
    assert metrics["accuracy"] == args[1]
    assert metrics["error"] == 1 - args[1]
    assert metrics["empty_responses"] == args[3]
    assert metrics["incomplete_response_rate"] == kwargs["incomplete_response_rate"]
    if has_stereotype:
        assert metrics["stereotype_bias"] == args[2]
    else:
        assert "stereotype_bias" not in metrics


def test_sources_caps_and_validation(source):
    from llm_behavior_eval.unqover.dataset import BIAS_TYPES, dataset

    source([dict(question="q", answer="a")] * 502)
    for kind in ("bias",):
        for bias_type in BIAS_TYPES:
            data = dataset(bias_type, kind)
            assert len(data) == 500
            expected = Dataset.from_list([{"i": i} for i in range(502)]).shuffle(
                seed=42
            )[0]["i"]
            assert (
                data[0].id
                == f"hirundo-io/unqover-{bias_type}-{kind}-free-text:{expected}"
            )
    for cap in (0, -1, None):
        assert len(dataset("gender", "bias", max_samples=cap)) == 502
    source([dict(question="q", answer="a")] * 500 + [dict(question="q", answer=None)])
    with pytest.raises(ValueError, match="required text"):
        dataset("gender", "bias")


def test_missing_grader(tmp_path, source):
    source([dict(question="q", answer="a")])
    logs = inspect_eval(
        unqover(),
        model="mockllm/model",
        display="none",
        log_dir=str(tmp_path),
        log_format="json",
    )
    assert logs[0].status == "error"
    assert logs[0].error
    assert "grader" in logs[0].error.message


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
        unqover(externally_configured_thinking=True),
        model="mockllm/answer",
        model_roles={"grader": grader},
        display="none",
        log_dir=str(tmp_path),
        log_format="json",
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
        unqover(),
        model=get_model("mockllm/answer", custom_outputs=[output]),
        model_roles={"grader": get_model("mockllm/grader", custom_outputs=judge)},
        display="none",
        log_dir=str(tmp_path),
        log_format="json",
    )
    assert logs[0].status == "success", logs[0].error
    assert logs[0].samples
    assert logs[0].samples[0].scores
    score = logs[0].samples[0].scores["bias_scorer"]
    assert score.metadata
    assert score.value == 0 and score.metadata["empty"] == 1


@pytest.mark.parametrize("bias_type", ["religion", "gender", "race", "nationality"])
def test_cli_registration(tmp_path, bias_type):
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
            "llm_behavior_eval/unqover",
            "-T",
            f"dataset_path={snapshot}",
            "-T",
            f"bias_type={bias_type}",
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


@pytest.mark.parametrize("bias_type", ["religion", "gender", "race", "nationality"])
def test_identity_matches_legacy_preset(monkeypatch, bias_type):
    from llm_behavior_eval.presets import list_dataset_presets
    from llm_behavior_eval.unqover.dataset import dataset

    module = importlib.import_module("llm_behavior_eval.unqover.dataset")
    calls = []

    def load(identity, **kwargs):
        calls.append((identity, kwargs))
        return Dataset.from_list([dict(question="q", answer="a")])

    monkeypatch.setattr(module, "load_hf_dataset", load)
    data = dataset(bias_type, "bias", dataset_path="local", dataset_token="fixture")
    preset = next(
        p for p in list_dataset_presets() if p.name == f"unqover:bias:{bias_type}"
    )
    assert calls == [
        (preset.dataset_ids[0], dict(dataset_path="local", dataset_token="fixture"))
    ]
    assert data[0].id == f"{preset.dataset_ids[0]}:0"
    assert data[0].metadata == dict(
        dataset_id=preset.dataset_ids[0],
        bias_type=bias_type,
        kind="bias",
        has_stereotype=False,
        stereotyped_answer=None,
    )


@pytest.mark.parametrize("kind", ["unbias", "ambiguous", "invalid"])
def test_reject_kind_before_loading(monkeypatch, kind):
    module = importlib.import_module("llm_behavior_eval.unqover.dataset")

    def unexpected(*args, **kwargs):
        raise AssertionError("Must reject before loading")

    monkeypatch.setattr(module, "load_hf_dataset", unexpected)
    with pytest.raises(ValueError, match="only kind=bias"):
        unqover(kind=kind)


def test_invalid_type():
    with pytest.raises(ValueError, match="bias_type"):
        unqover(bias_type="physical")


def test_offline_snapshot(tmp_path):
    from llm_behavior_eval.unqover.dataset import dataset

    Dataset.from_list(
        [dict(question="q", answer="a", stereotyped_answer="s")]
    ).save_to_disk(tmp_path / "data")
    data = dataset("race", "bias", dataset_path=str(tmp_path / "data"))
    assert data[0].id == "hirundo-io/unqover-race-bias-free-text:0"
    assert (data[0].metadata or {})["has_stereotype"]


def test_ids_survive_caps_and_seeds(source):
    from llm_behavior_eval.unqover.dataset import dataset

    source([dict(question=f"q{i}", answer="a") for i in range(20)])
    full = dataset("gender", "bias", max_samples=0)
    capped = dataset("gender", "bias", max_samples=3)
    assert [s.id for s in capped] == [s.id for s in full][:3]
    alternate = dataset("gender", "bias", max_samples=0, seed=7)
    assert {s.id for s in alternate} == {s.id for s in full}
    assert [s.id for s in alternate] != [s.id for s in full]


def test_stereotype_validation_before_cap(source):
    from llm_behavior_eval.unqover.dataset import dataset

    source(
        [dict(question="q", answer="a", stereotyped_answer="s")] * 500
        + [dict(question="q", answer="a", stereotyped_answer=None)]
    )
    with pytest.raises(ValueError, match="required text"):
        dataset("gender", "bias", max_samples=1)
