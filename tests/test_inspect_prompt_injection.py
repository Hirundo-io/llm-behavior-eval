import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from datasets import Dataset
from inspect_ai import eval as _inspect_eval
from inspect_ai.model import ModelOutput, get_model

from llm_behavior_eval.prompt_injection.dataset import DATASET_ID, dataset
from llm_behavior_eval.prompt_injection.scorer import parse_yes_no
from llm_behavior_eval.prompt_injection.task import prompt_injection
from tests.golden import BASELINE as GOLDEN


def inspect_eval(*args, **kwargs) -> Any:
    # JSON avoids lazy binary-log loading hangs in this sandbox.
    return _inspect_eval(*args, log_format="json", **kwargs)


@pytest.fixture
def source(monkeypatch):
    module = importlib.import_module("llm_behavior_eval.prompt_injection.dataset")

    def install(rows):
        monkeypatch.setattr(
            module, "load_hf_dataset", lambda *a, **k: Dataset.from_list(rows)
        )

    return install


@pytest.mark.parametrize(
    "raw", ["Yes", "NO!", "yesterday", "", "No then yes", "YES/no"]
)
def test_parser_parity(raw):
    expected = {
        "Yes": "Yes",
        "NO!": "No",
        "yesterday": "No",
        "": "No",
        "No then yes": "No",
        "YES/no": "Yes",
    }
    assert parse_yes_no(raw) == expected[raw]


@pytest.mark.parametrize("yes,total", [(1, 5), (2, 5), (1, 10), (3, 7), (0, 5), (5, 5)])
def test_asr_preserves_exact_legacy_arithmetic(monkeypatch, yes, total):
    from inspect_ai.scorer import SampleScore, Score

    from llm_behavior_eval.prompt_injection.scorer import attack_success_rate

    labels = ["Yes"] * yes + ["No"] * (total - yes)
    golden = json.loads(
        (Path(__file__).parent / "golden/purple_baseline.json").read_text()
    )
    scores = [SampleScore(score=Score(value=int(label == "Yes"))) for label in labels]
    scores.append(SampleScore(score=Score(value=0, metadata={"skipped": True})))
    actual = cast("Any", attack_success_rate())(scores)
    assert actual == golden[f"{yes}/{total}"]


def test_dataset(source):
    rows = [
        dict(
            question=f"q{i}",
            answer="a",
            system_prompt="override" if i == 0 else "",
            judge_question="j" if i == 0 else None,
        )
        for i in range(503)
    ]
    source(rows)
    data: Any = dataset()
    assert len(data) == 500
    expected = (
        Dataset.from_list(rows).add_column("idx", list(range(503))).shuffle(seed=42)
    )
    assert [s.id for s in data] == [f"{DATASET_ID}:{i}" for i in expected["idx"][:500]]
    assert len(dataset(max_samples=0)) == len(dataset(max_samples=-1)) == 503
    first: Any = next(
        s for s in dataset(max_samples=0) if (s.metadata or {})["source_index"] == 0
    )
    assert first.input[0].text == "override"
    assert first.input[1].text == "q0\n"
    assert first.metadata["judge_question"] == "j"
    assert data[0].metadata["judge_question"] == data[0].metadata["question"]
    source(rows + [dict(question="", answer="a")])
    with pytest.raises(ValueError):
        dataset(max_samples=1)


def test_native_legacy_grade_parity(source, tmp_path, monkeypatch):
    raw = ["Yes", "malformed", "No", "unused"]
    answers = ["<think>secret</think>a0", "a1", "a2", "a3"]
    source([dict(question=f"q{i}", answer="a") for i in range(4)])

    def answer(messages, tools, choice, config):
        assert config.max_tokens == 128 and config.temperature == 0
        i = int(messages[-1].text.strip()[1:])
        output = ModelOutput.from_content("mockllm/answer", answers[i])
        if i == 3:
            output.choices[0].stop_reason = "max_tokens"
        return output

    def judge(messages, tools, choice, config):
        assert config.max_tokens == 32 and config.temperature == 0
        text = messages[-1].text
        assert "secret" not in text
        i = next(i for i in range(3) if f'"a{i}"' in text)
        assert f"Question: q{i}" in text
        return ModelOutput.from_content("mockllm/judge", raw[i])

    log = inspect_eval(
        prompt_injection(),
        model=get_model("mockllm/answer", custom_outputs=answer),
        model_roles={"grader": get_model("mockllm/judge", custom_outputs=judge)},
        log_dir=str(tmp_path),
        display="none",
    )[0]
    assert log.status == "success", log.error
    saved = GOLDEN["injection_aggregate"]
    metrics = log.results.scores[0].metrics
    assert metrics["attack_success_rate"].value == 100 * (1 - saved["accuracy"])
    assert (
        metrics["incomplete_response_rate"].value
        == saved["incomplete_response_rate"]
        == 0.25
    )
    scores = [next(iter(s.scores.values())) for s in log.samples]
    assert sum(bool(s.metadata.get("skipped")) for s in scores) == 1
    assert sorted(s.metadata["raw_judge_text"] for s in scores) == [
        "",
        "No",
        "Yes",
        "malformed",
    ]
    assert all(s.metadata["response_tokens"] is None for s in scores)
    assert all("1024-token" in s.metadata["prompt_parity"] for s in scores)


def test_requires_grader(source, tmp_path):
    source([dict(question="q", answer="a")])
    log = inspect_eval(
        prompt_injection(),
        model="mockllm/answer",
        log_dir=str(tmp_path),
        display="none",
    )[0]
    assert log.status == "error"
    assert "grader" in log.error.message


def test_prompt_matches_legacy():
    from llm_behavior_eval.prompt_injection.prompts import PROMPT_JUDGE_RESPONSE

    assert PROMPT_JUDGE_RESPONSE == GOLDEN["prompts"]["PROMPT_JUDGE_RESPONSE"]


def test_policy_and_budget(source):
    from llm_behavior_eval.prompt_injection.task import model_config

    source([dict(question="q", answer="a")])
    assert prompt_injection(max_tokens=512).config.max_tokens == 512
    with pytest.raises(ValueError):
        prompt_injection(max_tokens=0)
    fake: Any = SimpleNamespace(api=SimpleNamespace())
    with pytest.raises(ValueError, match="thinking disabled"):
        model_config(fake)
    assert model_config(fake, external=True).max_tokens == 128
    assert model_config(fake, external=True, judge=True).max_tokens == 32


@pytest.mark.parametrize(
    "stop", ["max_tokens", "model_length", "unknown", "content_filter"]
)
def test_token_count_and_all_skipped(source, tmp_path, monkeypatch, stop):
    from math import isnan

    module = importlib.import_module("llm_behavior_eval.prompt_injection.scorer")

    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert text == "formatted" and add_special_tokens is False
            return [1, 2, 3]

    monkeypatch.setattr(
        module,
        "import_module",
        lambda name: SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())
        ),
    )
    source([dict(question="q", answer="a")])
    output = ModelOutput.from_content(
        "mockllm/answer", "<think>reason</think>formatted"
    )
    output.choices[0].stop_reason = stop

    def no_judge(*args):
        pytest.fail("truncated output must not call grader")

    log = inspect_eval(
        prompt_injection(response_tokenizer="/offline/tokenizer"),
        model=get_model("mockllm/answer", custom_outputs=[output]),
        model_roles={"grader": get_model("mockllm/judge", custom_outputs=no_judge)},
        log_dir=str(tmp_path),
        display="none",
    )[0]
    assert log.status == "success", log.error
    score = next(iter(log.samples[0].scores.values()))
    assert score.metadata["response_tokens"] == 3
    assert isnan(log.results.scores[0].metrics["attack_success_rate"].value)
    assert log.results.scores[0].metrics["incomplete_response_rate"].value == int(
        stop == "max_tokens"
    )


def test_offline_snapshot(tmp_path):
    Dataset.from_list([dict(question="offline", answer="target")]).save_to_disk(
        str(tmp_path / "data")
    )
    loaded: Any = dataset(dataset_path=str(tmp_path / "data"))
    assert loaded[0].id == DATASET_ID + ":0"
    assert loaded[0].metadata["question"] == "offline"


def test_current_preprocessor_and_explicit_truncation_gap(source, monkeypatch):
    question = "q" * 1100
    source([dict(question=question, answer="a")])
    native: Any = dataset()[0]
    assert native.metadata["judge_question"] == question
    # Baseline fake-tokenizer widths: input=1024, judge_question=128.
    assert (
        len(question)
        > GOLDEN["preprocessor_widths"]["prompt_injection"]["judge_question"]
    )
    assert (
        len(native.input[1].text)
        > GOLDEN["preprocessor_widths"]["prompt_injection"]["input"]
    )
    assert len(native.input[1].text) == 1101


def test_independent_api_role(tmp_path, monkeypatch, source):
    """Role routing fixture, not HTTP/provider inference parity."""
    from inspect_ai.model import Model

    source([dict(question="q", answer="a", judge_question="override?")])
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
            return ModelOutput.from_content("judge", "Yes")
        return await original(self, input, *args, **kwargs)

    monkeypatch.setattr(Model, "generate", generate)
    log = inspect_eval(
        prompt_injection(externally_configured_thinking=True),
        model="mockllm/answer",
        model_roles={"grader": grader},
        display="none",
        log_dir=str(tmp_path),
    )[0]
    assert log.status == "success", log.error
    assert len(calls) == 1 and "Question: override?" in calls[0]
    assert log.results.scores[0].metrics["attack_success_rate"].value == 100
