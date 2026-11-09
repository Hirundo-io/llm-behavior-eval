"""Deterministic fixtures only; no live/GPU or publication parity claim."""

import asyncio
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from datasets import Dataset, DatasetDict
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalSpec, list_eval_logs, read_eval_log
from inspect_ai.model import GenerateConfig, ModelName, ModelOutput, get_model
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER, SampleScore, Score, Target
from inspect_ai.solver import TaskState

from llm_behavior_eval.ccpc import contract, dataset
from llm_behavior_eval.ccpc.metrics import (
    ccpc_metrics,
    censorship_rate,
    censorship_rate_stderr,
    judged_fraction,
    parse_verdict,
)
from llm_behavior_eval.ccpc.scorer import _judge_prompt, ccpc_scorer
from llm_behavior_eval.ccpc.solver import PROMPT_PARITY
from llm_behavior_eval.ccpc.task import ccpc
from llm_behavior_eval.common.parity import parity_config_path, read_parity_config
from llm_behavior_eval.common.target_metrics import (
    TARGET_METRICS_FILE,
    load_target_metrics,
    missing_target_metrics,
    read_target_metrics_file,
)
from llm_behavior_eval.evaluation_utils import censorship_utils as legacy
from llm_behavior_eval.evaluation_utils.eval_config import (
    FAMILY_TOKEN_DEFAULTS,
    EvaluationConfig,
)

CCPC = "llm_behavior_eval.ccpc"


@pytest.fixture(autouse=True)
def writable_inspect_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


@pytest.fixture
def rows(monkeypatch):
    data = Dataset.from_dict(
        {
            "question": [f"问题 {i}" for i in range(500)],
            "benchmark_id": [f"id-{i}" for i in range(500)],
        }
    )
    # Synthetic rows: patch only the expected fingerprint, explicitly. Production
    # validation (schema, count, IDs, fingerprint check) still runs unchanged.
    fingerprint = legacy.censorship_dataset_fingerprint(data)
    monkeypatch.setattr(dataset, "CCPC_DATASET_FINGERPRINT", fingerprint)
    monkeypatch.setattr(dataset, "load_dataset", lambda *a, **kw: data)
    return data


def test_contract():
    for key in vars(contract):
        if key.startswith("CCPC_"):
            assert getattr(contract, key) == getattr(legacy, key)


@pytest.mark.parametrize(
    "text",
    ["True", " FALSE\n", "true.", "yes", "", "<think>x</think>True", "True\nFalse"],
)
def test_parser(text):
    expected = legacy.parse_censorship_judgment(text)
    assert parse_verdict(text) == (
        NOANSWER if expected is None else CORRECT if expected else INCORRECT
    )


@pytest.mark.parametrize("raw", [[], ["True"], ["True", "False"], ["True", "oops"]])
def test_metrics(raw):
    expected = legacy.calculate_censorship_metrics(
        [legacy.parse_censorship_judgment(x) for x in raw]
    )
    scores = [
        SampleScore(score=Score(value=0, metadata={"verdict": parse_verdict(x)}))
        for x in raw
    ]
    reported = cast("Any", ccpc_metrics())(scores)
    assert list(reported) == ["censorship_rate", "stderr", "judged_fraction"]
    for compute, name, old in zip(
        [censorship_rate, censorship_rate_stderr, judged_fraction],
        reported,
        expected,
        strict=True,
    ):
        actual = compute(scores)
        both_nan = math.isnan(reported[name]) and math.isnan(actual)
        assert both_nan or reported[name] == actual
        if old is None or math.isnan(old):
            assert math.isnan(actual)
        else:
            assert actual == old


def test_pinned_load_and_order(rows, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        dataset, "load_dataset", lambda *a, **kw: calls.append((a, kw)) or rows
    )
    assert dataset.load_ccpc_dataset()[0].id == "id-0"
    assert calls[0] == (
        (contract.CCPC_DATASET_REPOSITORY,),
        dict(
            name="default",
            split="test",
            revision=contract.CCPC_DATASET_REVISION,
            token=None,
        ),
    )
    for source in (rows, DatasetDict({"test": rows, "train": rows.select([1])})):
        path = tmp_path / str(len(list(tmp_path.iterdir())))
        source.save_to_disk(str(path))
        result = dataset.load_ccpc_dataset(file_path=str(path))
        assert [r.id for r in result] == [f"id-{i}" for i in range(500)]
        assert result.name == "chinese_censorship"
        assert dataset.load_ccpc_dataset(dataset_path=str(path))[0].id == "id-0"


@pytest.mark.parametrize(
    "mutation", ["reorder", "last", "duplicate", "columns", "count", "blank"]
)
def test_validation_before_conversion(rows, monkeypatch, mutation):
    data = rows.to_dict()
    if mutation == "reorder":
        changed = rows.select(list(reversed(range(500))))
    elif mutation == "columns":
        changed = rows.add_column("extra", [0] * 500)
    elif mutation == "count":
        changed = rows.select(range(499))
    else:
        if mutation == "last":
            data["question"][-1] = "changed"
        elif mutation == "blank":
            data["question"][-1] = " "
        else:
            data["benchmark_id"][-1] = data["benchmark_id"][0]
        changed = Dataset.from_dict(data)
    monkeypatch.setattr(dataset, "load_dataset", lambda *a, **kw: changed)
    with pytest.raises(ValueError):
        dataset.load_ccpc_dataset()


def test_production_fingerprint_rejects_unpinned_rows(monkeypatch):
    # No fingerprint patch: the shipped contract value guards the real source.
    data = Dataset.from_dict(
        {
            "question": [f"q{i}" for i in range(500)],
            "benchmark_id": [f"id-{i}" for i in range(500)],
        }
    )
    monkeypatch.setattr(dataset, "load_dataset", lambda *a, **kw: data)
    assert dataset.CCPC_DATASET_FINGERPRINT == legacy.CCPC_DATASET_FINGERPRINT
    with pytest.raises(ValueError, match=legacy.CCPC_DATASET_FINGERPRINT):
        dataset.load_ccpc_dataset()


def test_limit_does_not_skip_validation(rows, monkeypatch, tmp_path):
    # Inspect's limit applies after the task validated the complete source.
    data = rows.to_dict()
    data["question"][-1] = "changed"
    changed = Dataset.from_dict(data)
    monkeypatch.setattr(dataset, "load_dataset", lambda *a, **kw: changed)
    with pytest.raises(ValueError, match="fingerprint"):
        inspect_eval(
            ccpc(),
            model="mockllm/answer",
            model_roles={"grader": "mockllm/judge"},
            limit=1,
            log_dir=str(tmp_path),
            display="none",
        )


@pytest.mark.parametrize("stop", ["stop", "max_tokens", "unknown", None])
def test_native_judge_all_finishes(monkeypatch, stop):
    calls = []

    class Judge:
        def __str__(self):
            return "mockllm/judge"

        async def generate(self, messages, **kwargs):
            calls.append((messages, kwargs))
            result = ModelOutput.from_content("judge", " TRUE ")
            result.choices[0].stop_reason = "max_tokens"
            return result

    monkeypatch.setattr("llm_behavior_eval.ccpc.scorer.get_model", lambda **kw: Judge())
    output = ModelOutput.from_content("model", "<think>hidden</think>答案")
    if stop is None:
        output.choices = []
    else:
        output.choices[0].stop_reason = stop
    state = TaskState(
        model=ModelName("mockllm/model"),
        sample_id="a",
        epoch=1,
        input="问题",
        messages=[],
        output=output,
    )
    score = asyncio.run(ccpc_scorer()(state, Target("")))
    assert score is not None and score.metadata is not None
    assert score.metadata["verdict"] == CORRECT
    assert score.metadata["raw_judge_text"] == " TRUE "
    assert score.metadata["stop_reason"] == stop
    messages, kwargs = calls[0]
    # No GenerateConfig: the grader uses only its role's own config.
    assert kwargs == {}
    assert messages[0].role == "system"
    assert messages[0].text == legacy.CCPC_JUDGE_SYSTEM_PROMPT
    assert messages[1].text == legacy.CCPC_JUDGE_PROMPT_TEMPLATE.format(
        serialized_judge_input=json.dumps(
            {"question": "问题", "response": "答案"},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


@pytest.mark.parametrize("raw", ["False", "unparseable"])
def test_eval_serialization(rows, tmp_path, raw):
    log = inspect_eval(
        ccpc(),
        model=get_model(
            "mockllm/answer",
            custom_outputs=[
                ModelOutput.from_content("answer", "text") for _ in range(2)
            ],
        ),
        model_roles={
            "grader": get_model(
                "mockllm/judge",
                custom_outputs=[
                    ModelOutput.from_content("judge", "True"),
                    ModelOutput.from_content("judge", raw),
                ],
            )
        },
        limit=2,
        log_dir=str(tmp_path),
        display="none",
    )[0]
    assert log.status == "success", log.error
    with open(log.location, "rb") as stream:
        restored = read_eval_log(stream)
    assert restored.results is not None and restored.samples is not None
    metrics = next(iter(restored.results.scores)).metrics
    rate = next(m.value for m in metrics.values() if m.name.endswith("censorship_rate"))
    assert math.isnan(rate) if raw == "unparseable" else rate == 0.5
    from llm_behavior_eval.ccpc.metrics import reported_rate

    assert reported_rate(rate, percentage=True) == (
        None if raw == "unparseable" else 50.0
    )
    assert json.loads(
        json.dumps({"censorship_rate": reported_rate(rate)}, allow_nan=False)
    )["censorship_rate"] == (None if raw == "unparseable" else 0.5)
    assert restored.samples[0].messages[0].role == "user"
    assert restored.samples[0].messages[0].text == "问题 0\n"
    assert restored.samples[0].metadata["prompt_parity"] == PROMPT_PARITY
    assert "1024" in PROMPT_PARITY
    assert len(restored.samples) == 2


def test_required_grader(rows, tmp_path):
    log = inspect_eval(
        ccpc(),
        model="mockllm/model",
        limit=1,
        log_dir=str(tmp_path),
        display="none",
    )[0]
    assert log.status == "error"
    assert log.error is not None
    assert "grader" in log.error.message


def test_no_prompt_truncation_hidden():
    question = "long " * 2000
    assert (
        json.loads(_judge_prompt(question, "answer").split("\n", 1)[1])["question"]
        == question
    )
    assert "unresolved" in PROMPT_PARITY


def test_independent_endpoint(rows, monkeypatch, tmp_path):
    from inspect_ai.model import Model

    grader = get_model(
        "openai-api/airgap/judge",
        base_url="http://judge.invalid/v1",
        api_key="fixture-only",
    )
    original = Model.generate
    calls = []
    answer_configs = []

    async def generate(self, input, *args, **kwargs):  # noqa: A002
        if self.name == grader.name:
            calls.append(input)
            # The scorer passes no GenerateConfig to the grader.
            assert not args and "config" not in kwargs
            return ModelOutput.from_content("judge", "True")
        answer_configs.append(kwargs.get("config"))
        return await original(self, input, *args, **kwargs)

    monkeypatch.setattr(Model, "generate", generate)
    log = inspect_eval(
        ccpc(),
        model="mockllm/answer",
        limit=1,
        model_roles={"grader": grader},
        display="none",
        log_dir=str(tmp_path),
    )[0]
    assert log.status == "success", log.error
    assert len(calls) == 1
    assert calls[0][0].role == "system"
    # The solver adds nothing to the (empty) eval-level config.
    assert answer_configs == [GenerateConfig()]


def test_task_sets_no_inference_settings(rows):
    task = ccpc()
    assert task.config == GenerateConfig()
    assert not task.model_roles


def test_registry_discovery():
    from llm_behavior_eval import _registry

    assert vars(_registry)["ccpc"] is ccpc
    assert "ccpc" in _registry.__all__
    listed = subprocess.run(
        [sys.executable, "-m", "inspect_ai", "list", "tasks", "llm_behavior_eval"],
        capture_output=True,
        text=True,
    )
    assert listed.returncode == 0, listed.stderr
    assert "ccpc" in listed.stdout


# parity.yaml request bodies (vLLM/OpenAI-compatible): the legacy thinking
# setting, and the legacy prompt tokenization (special tokens added; answer
# prompts truncated to 1024 tokens on the right).
ANSWER_BODY = {
    "chat_template_kwargs": {"enable_thinking": False},
    "add_special_tokens": True,
    "truncate_prompt_tokens": 1024,
    "truncation_side": "right",
}
GRADER_BODY = {
    "chat_template_kwargs": {"enable_thinking": False},
    "add_special_tokens": True,
}


def test_shipped_parity_config_matches_legacy():
    path = parity_config_path(CCPC)
    assert path == Path(__file__).parents[1] / "llm_behavior_eval" / "ccpc" / (
        "parity.yaml"
    )
    config = yaml.safe_load(path.read_text())
    assert set(config) == {"generate_config", "eval_config", "model_roles"}
    legacy_defaults = FAMILY_TOKEN_DEFAULTS["censorship"]
    assert legacy_defaults["enable_thinking"] is False
    assert legacy_defaults["sample_judge"] is False
    assert EvaluationConfig.model_fields["sample"].default is False

    answer = GenerateConfig.model_validate(config["generate_config"])
    assert answer.temperature == 0.0
    assert answer.max_tokens == legacy_defaults["max_answer_tokens"] == 8192
    assert answer.seed == 42
    assert answer.extra_body == ANSWER_BODY
    assert config["eval_config"] == {
        "limit": EvaluationConfig.model_fields["max_samples"].default
    }
    assert config["eval_config"]["limit"] == contract.CCPC_EXPECTED_ROWS

    grader = config["model_roles"]["grader"]
    assert set(grader) == {"model", "config", "args"}
    assert grader["model"] == "hf/google/gemma-4-26B-A4B-it"
    judge = GenerateConfig.model_validate(grader["config"])
    assert judge.temperature == 0.0
    assert judge.max_tokens == legacy_defaults["max_judge_tokens"] == 4096
    assert judge.seed == 42
    assert judge.extra_body == GRADER_BODY
    # Inspect's HF provider samples and leaves thinking to the template unless
    # told otherwise.
    assert grader["args"] == {"do_sample": False, "enable_thinking": False}


@pytest.fixture
def mock_parity_config(tmp_path) -> Path:
    """The shipped parity config with only the grader model swapped for a mock."""
    config = yaml.safe_load(parity_config_path(CCPC).read_text())
    config["model_roles"]["grader"]["model"] = "mockllm/judge"
    config["model_roles"]["grader"]["args"] = {}
    path = tmp_path / "parity.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def assert_parity_recorded(spec: EvalSpec) -> None:
    generate = spec.model_generate_config
    assert (generate.temperature, generate.max_tokens) == (0.0, 8192)
    assert spec.config.limit == 500
    assert spec.model_roles
    grader = spec.model_roles["grader"]
    assert not isinstance(grader, list)
    assert (grader.config.temperature, grader.config.max_tokens) == (0.0, 4096)


def test_parity_config_python(rows, tmp_path, mock_parity_config):
    log = inspect_eval(
        ccpc(),
        model="mockllm/answer",
        log_dir=str(tmp_path),
        display="none",
        **read_parity_config(mock_parity_config),
    )[0]
    assert log.status == "success", log.error
    assert_parity_recorded(log.eval)
    assert log.samples is not None
    assert [sample.id for sample in log.samples][:2] == ["id-0", "id-1"]
    assert len(log.samples) == 500


def test_parity_config_cli(tmp_path, mock_parity_config):
    # A local save_to_disk snapshot of synthetic rows; the subprocess patches
    # only the expected fingerprint, explicitly, before running Inspect's CLI.
    data = Dataset.from_dict(
        {
            "question": [f"q{i}" for i in range(500)],
            "benchmark_id": [f"id-{i}" for i in range(500)],
        }
    )
    snapshot = tmp_path / "snapshot"
    data.save_to_disk(str(snapshot))
    script = (
        "import sys\n"
        "from llm_behavior_eval.ccpc import dataset\n"
        f"dataset.CCPC_DATASET_FINGERPRINT = {legacy.censorship_dataset_fingerprint(data)!r}\n"
        "from inspect_ai._cli.main import main\n"
        "sys.argv = ['inspect', *sys.argv[1:]]\n"
        "main()\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "eval",
            "llm_behavior_eval/ccpc",
            "--model",
            "mockllm/answer",
            "-T",
            f"dataset_path={snapshot}",
            "--run-config",
            str(mock_parity_config),
            "--log-dir",
            str(tmp_path / "logs"),
            "--display",
            "none",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    logs = list_eval_logs(str(tmp_path / "logs"))
    assert len(logs) == 1
    log = read_eval_log(logs[0])
    assert log.status == "success", log.error
    assert_parity_recorded(log.eval)


def test_target_metrics_match_eval_metric_names(rows, tmp_path):
    path = parity_config_path(CCPC).with_name(TARGET_METRICS_FILE)
    package_targets = read_target_metrics_file(path)
    assert list(package_targets) == ["llm_behavior_eval/ccpc"]
    targets = package_targets["llm_behavior_eval/ccpc"]
    assert load_target_metrics()["llm_behavior_eval/ccpc"] == targets
    assert targets == [
        {
            "name": "censorship_rate",
            "is_percentage": True,
            "is_higher_better": False,
            "is_normalized": True,
        }
    ]
    log = inspect_eval(
        ccpc(),
        model="mockllm/answer",
        model_roles={"grader": "mockllm/judge"},
        limit=1,
        log_dir=str(tmp_path),
        display="none",
    )[0]
    assert log.status == "success", log.error
    assert log.results is not None
    names = {name for score in log.results.scores for name in score.metrics}
    assert names == {"censorship_rate", "stderr", "judged_fraction"}
    # inspect-brief finds every target in the written log.
    assert missing_target_metrics(log.location) == []


def test_registry_names_are_package_specific():
    # A generic name (`stderr`, `censorship_rate`) defined by another benchmark
    # would replace CCPC's in Inspect's registry, and rescoring a log would run
    # that benchmark's code; see _registry.check_unique_registry_names.
    from inspect_ai.util import registry_info

    from llm_behavior_eval import _registry

    names = {
        name
        for (_, name), modules in _registry.registry_definitions().items()
        if any(module.startswith(f"{CCPC}.") for module in modules)
    }
    assert names == {
        "llm_behavior_eval/ccpc",
        "llm_behavior_eval/ccpc_metrics",
        "llm_behavior_eval/ccpc_prompt",
        "llm_behavior_eval/ccpc_scorer",
    }
    assert registry_info(ccpc_metrics()).name == "llm_behavior_eval/ccpc_metrics"


@pytest.mark.parametrize(
    ("judge", "rate", "judged"),
    [(["False", "True"], 0.5, 1.0), (["True", "oops"], None, 0.5)],
)
def test_epochs_keep_every_verdict(rows, tmp_path, judge, rate, judged):
    # Inspect's epoch reducers keep only the first epoch's metadata; the CCPC
    # metrics read unreduced scores so every epoch's verdict counts.
    log = inspect_eval(
        ccpc(),
        model=get_model(
            "mockllm/answer",
            custom_outputs=[ModelOutput.from_content("answer", "a")] * 2,
        ),
        model_roles={
            "grader": get_model(
                "mockllm/judge",
                custom_outputs=[ModelOutput.from_content("judge", j) for j in judge],
            )
        },
        limit=1,
        epochs=2,
        log_dir=str(tmp_path),
        display="none",
    )[0]
    assert log.status == "success", log.error
    assert log.results is not None
    metrics = {
        name: m.value
        for score in log.results.scores
        for name, m in score.metrics.items()
    }
    assert set(metrics) == {"censorship_rate", "stderr", "judged_fraction"}
    if rate is None:
        assert math.isnan(metrics["censorship_rate"])
        assert math.isnan(metrics["stderr"])
    else:
        assert metrics["censorship_rate"] == rate
        assert metrics["stderr"] == 0.0
    assert metrics["judged_fraction"] == judged


def test_upstream_license_ships_in_package_readme():
    readme = Path(dataset.__file__).with_name("README.md").read_text()
    assert "Copyright (c) 2026 Hirundo" in readme
    assert "Permission is hereby granted, free of charge" in readme
