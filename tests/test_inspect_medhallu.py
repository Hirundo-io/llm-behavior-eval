"""Med-Hallu mapping and native wiring; deterministic fixtures, not live parity."""

import importlib
from typing import Any, cast

import pytest
from datasets import Dataset
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

from llm_behavior_eval.medhallu.dataset import DATASET_ID, PARITY_GAPS, dataset
from llm_behavior_eval.medhallu.prompts import SYSTEM_MESSAGE
from llm_behavior_eval.medhallu.task import medhallu
from tests.golden import BASELINE as GOLDEN


def test_dataset(tmp_path):
    path = tmp_path / "snapshot"
    Dataset.from_list(
        [{"question": f"q{i}", "answer": "a"} for i in range(502)]
    ).save_to_disk(path)
    limited = dataset(dataset_path=str(path))
    full = dataset(0, dataset_path=str(path))
    assert len(limited) == 500
    assert len(full) == len(dataset(-1, dataset_path=str(path))) == 502
    expected = Dataset.from_dict({"index": list(range(502))}).shuffle(seed=42)
    order = list(expected["index"])
    assert [s.id for s in limited] == [f"{DATASET_ID}:{i}" for i in order[:500]]
    assert [s.id for s in full] == [f"{DATASET_ID}:{i}" for i in order]
    assert limited.name == DATASET_ID
    assert limited[0].input == f"q{order[0]}"
    assert limited[0].metadata is not None
    assert limited[0].metadata["parity_gaps"] == PARITY_GAPS
    Dataset.from_list(
        [{"question": "q", "answer": "a"}, {"question": "", "answer": "a"}]
    ).save_to_disk(tmp_path / "invalid")
    with pytest.raises(ValueError):
        dataset(1, dataset_path=str(tmp_path / "invalid"))


def test_native_task(tmp_path):
    path = tmp_path / "snapshot"
    Dataset.from_list([{"question": "question", "answer": "gold"}]).save_to_disk(path)
    calls = []

    def answer(messages, tools, choice, config):
        calls.append(messages)
        assert config.max_tokens == 128 and config.temperature == 0
        return ModelOutput.from_content("mockllm/model", "answer")

    logs = inspect_eval(
        medhallu(dataset_path=str(path)),
        model=get_model("mockllm/model", custom_outputs=answer),
        model_roles={
            "grader": get_model(
                "mockllm/grader",
                custom_outputs=lambda *args: ModelOutput.from_content(
                    "mockllm/grader", "B"
                ),
            )
        },
        log_dir=str(tmp_path / "logs"),
        display="none",
        log_format="json",
    )
    assert logs[0].status == "success", logs[0].error
    assert calls[0][0].text == SYSTEM_MESSAGE
    assert calls[0][-1].text == "question\n"
    assert logs[0].results is not None
    metrics = logs[0].results.scores[0].metrics
    assert metrics["accuracy"].value == 0
    assert metrics["empty_responses"].value == 0
    import json
    from pathlib import Path

    saved_log = json.loads(Path(logs[0].location).read_text())
    saved_score = saved_log["samples"][0]["scores"]["hallucination_scorer"]
    assert saved_score["value"] == 0
    assert saved_score["explanation"] == "B"
    metadata = saved_score["metadata"]
    assert metadata["raw_judge_text"] == "B"
    assert metadata["dataset_id"] == DATASET_ID
    assert metadata["source_index"] == 0
    assert metadata["parity_gaps"] == PARITY_GAPS
    assert metadata["stop_reason"] == "stop"
    assert metadata["grade"] == "INCORRECT"
    assert not metadata["skipped"] and not metadata["parse_error"]
    missing = inspect_eval(
        medhallu(dataset_path=str(path)),
        model="mockllm/model",
        log_dir=str(tmp_path / "missing"),
        display="none",
        log_format="json",
    )
    assert missing[0].status == "error"


def test_documented_tokenizer_parity_gaps(monkeypatch, tmp_path):
    # Baseline fake-tokenizer widths were 1024/256; native retains raw text.
    question, target = "q" * 1200, "a" * 300
    path = tmp_path / "long"
    Dataset.from_list([{"question": question, "answer": target}]).save_to_disk(path)
    sample = dataset(dataset_path=str(path))[0]
    assert sample.input == question and sample.target == target
    assert len(question) > GOLDEN["preprocessor_widths"]["medhallu"]["input"]
    assert len(target) > GOLDEN["preprocessor_widths"]["medhallu"]["target"]
    assert sample.metadata is not None
    assert len(sample.metadata["parity_gaps"]) == 3


def test_independent_api_grader(tmp_path, monkeypatch):
    """Check independent API role routing with deterministic generation, not HTTP parity."""
    path = tmp_path / "snapshot"
    Dataset.from_list([{"question": "q", "answer": "a"}]).save_to_disk(path)
    grader = get_model(
        "openai-api/airgap/judge",
        base_url="http://judge.invalid/v1",
        api_key="fixture-not-secret",
    )
    requests = []

    async def generate(prompt, *, config):
        requests.append((prompt, config))
        return ModelOutput.from_content("judge", "B")

    monkeypatch.setattr(grader, "generate", generate)
    logs = inspect_eval(
        medhallu(dataset_path=str(path), grader_thinking_disabled=True),
        model="mockllm/model",
        model_roles={"grader": grader},
        display="none",
        log_format="json",
        log_dir=str(tmp_path / "logs"),
    )
    assert logs[0].status == "success", logs[0].error
    assert str(cast("Any", grader.api).client.base_url) == "http://judge.invalid/v1/"
    assert len(requests) == 1
    assert requests[0][1].max_tokens == 32
    assert logs[0].results is not None
    assert logs[0].results.scores[0].metrics["accuracy"].value == 0


def test_cli_registration(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "snapshot"
    Dataset.from_list([{"question": "q", "answer": "a"}]).save_to_disk(path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "inspect_ai",
            "eval",
            "llm_behavior_eval/medhallu",
            "--model",
            "mockllm/model",
            "--model-role",
            "grader=mockllm/grader",
            "-T",
            f"dataset_path={path}",
            "--log-dir",
            str(tmp_path / "logs"),
            "--log-format",
            "json",
            "--display",
            "none",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr


def test_loader_identity_and_token(monkeypatch):
    module = importlib.import_module("llm_behavior_eval.medhallu.dataset")
    calls = []

    def load(identity, **kwargs):
        calls.append((identity, kwargs))
        return Dataset.from_list([{"question": "medical question", "answer": "answer"}])

    monkeypatch.setattr(module, "load_hf_dataset", load)
    sample = dataset(None, dataset_path="offline/source", dataset_token="fixture")[0]
    assert calls == [
        (DATASET_ID, {"dataset_path": "offline/source", "dataset_token": "fixture"})
    ]
    assert sample.id == "hirundo-io/medhallu:0"
    assert sample.target == "answer"


def test_selection_matches_legacy(monkeypatch):
    source = Dataset.from_list(
        [
            {"question": f"q{i}", "answer": f"a{i}", "source_index": i}
            for i in range(502)
        ]
    )
    module = importlib.import_module("llm_behavior_eval.medhallu.dataset")
    monkeypatch.setattr(module, "load_hf_dataset", lambda *args, **kwargs: source)
    expected = GOLDEN["medhallu_order"]
    actual = dataset()
    assert [s.id for s in actual] == [f"{DATASET_ID}:{i}" for i in expected]
    assert [s.input for s in actual] == [f"q{i}" for i in expected]
    assert [s.target for s in actual] == [f"a{i}" for i in expected]
