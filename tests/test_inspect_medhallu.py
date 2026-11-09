"""Med-Hallu: deterministic legacy parity fixtures, not live provider/GPU parity."""

import ast
import asyncio
import importlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml
from datasets import Dataset
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalSpec, list_eval_logs, read_eval_log, recompute_metrics
from inspect_ai.model import (
    ChatMessageUser,
    GenerateConfig,
    ModelName,
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import Metric, SampleScore, Score, Target
from inspect_ai.solver import TaskState

from llm_behavior_eval.common.metrics import incomplete_response_rate
from llm_behavior_eval.common.parity import parity_config_path, read_parity_config
from llm_behavior_eval.common.target_metrics import (
    TARGET_METRICS_FILE,
    load_target_metrics,
    missing_target_metrics,
    read_target_metrics_file,
)
from llm_behavior_eval.medhallu.dataset import DATASET_ID, PARITY_GAPS, dataset
from llm_behavior_eval.medhallu.metrics import (
    medhallu_accuracy,
    medhallu_empty_responses,
)
from llm_behavior_eval.medhallu.prompts import GRADER_TEMPLATE, SYSTEM_MESSAGE
from llm_behavior_eval.medhallu.scorer import medhallu_scorer, parse_grade
from llm_behavior_eval.medhallu.task import medhallu

PACKAGE = "llm_behavior_eval.medhallu"
PACKAGE_DIR = Path(__file__).parents[1] / "llm_behavior_eval" / "medhallu"
METRICS = ["accuracy", "empty_responses", "incomplete_response_rate"]


def rate(metric: Metric) -> Callable[[list[SampleScore]], float]:
    return cast("Callable[[list[SampleScore]], float]", metric)


def mapping(metric: Metric) -> Callable[[list[SampleScore]], dict[str, float]]:
    return cast("Callable[[list[SampleScore]], dict[str, float]]", metric)


def snapshot(tmp_path: Path, rows: list[dict[str, str]], name="snapshot") -> str:
    path = tmp_path / name
    Dataset.from_list(rows).save_to_disk(path)
    return str(path)


def mock_grader(reply: str = "B"):
    return get_model(
        "mockllm/grader",
        custom_outputs=lambda *args: ModelOutput.from_content("mockllm/grader", reply),
    )


def test_package_is_self_contained():
    for path in PACKAGE_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = node.names[0].name
            else:
                continue
            if module.startswith("llm_behavior_eval"):
                assert module.startswith(
                    ("llm_behavior_eval.common.", "llm_behavior_eval.medhallu.")
                ), (path.name, module)


def test_prompts_match_legacy():
    from llm_behavior_eval.evaluation_utils.free_text_hallu_evaluator import (
        GRADER_TEMPLATE as LEGACY_TEMPLATE,
    )
    from llm_behavior_eval.evaluation_utils.prompts import FREE_TEXT_SYSTEM_MESSAGE

    assert GRADER_TEMPLATE == LEGACY_TEMPLATE
    assert SYSTEM_MESSAGE == FREE_TEXT_SYSTEM_MESSAGE["content"]


@pytest.mark.parametrize(
    "raw",
    ["A", "B", "C", "garbage", "", "AA", "a", "B then A", "(A)", "<think>B</think>A"],
)
def test_parse_legacy(raw):
    from llm_behavior_eval.evaluation_utils.free_text_hallu_evaluator import (
        FreeTextHaluEvaluator,
    )

    assert (
        parse_grade(raw)
        == FreeTextHaluEvaluator._map_judge_outputs([[{"generated_text": raw}]])[0]
    )


def test_dataset(tmp_path):
    path = snapshot(
        tmp_path, [{"question": f"q{i}", "answer": "a"} for i in range(502)]
    )
    full = dataset(dataset_path=path)
    order = list(
        Dataset.from_dict({"index": list(range(502))}).shuffle(seed=42)["index"]
    )
    assert [s.id for s in full] == [f"{DATASET_ID}:{i}" for i in order]
    assert full.name == DATASET_ID
    assert full[0].input == f"q{order[0]}"
    assert full[0].metadata is not None
    assert full[0].metadata["parity_gaps"] == PARITY_GAPS
    assert full[0].metadata["source_index"] == order[0]
    invalid = snapshot(
        tmp_path,
        [{"question": "q", "answer": "a"}, {"question": "", "answer": "a"}],
        "invalid",
    )
    with pytest.raises(ValueError, match="row 1"):
        dataset(dataset_path=invalid)


def test_loader_identity_and_token(monkeypatch):
    module = importlib.import_module("llm_behavior_eval.medhallu.dataset")
    calls = []

    def load(identity, **kwargs):
        calls.append((identity, kwargs))
        return Dataset.from_list([{"question": "medical question", "answer": "answer"}])

    monkeypatch.setattr(module, "load_hf_dataset", load)
    sample = dataset(dataset_path="offline/source", dataset_token="fixture")[0]
    assert calls == [
        (DATASET_ID, {"dataset_path": "offline/source", "dataset_token": "fixture"})
    ]
    assert sample.id == f"{DATASET_ID}:0"
    assert sample.target == "answer"


def test_parity_cap_selects_legacy_samples(monkeypatch):
    """Legacy shuffled with seed 42 and kept max_samples=500; the cap must match."""
    from llm_behavior_eval.evaluation_utils import base_evaluator

    source = Dataset.from_list(
        [
            {"question": f"q{i}", "answer": f"a{i}", "source_index": i}
            for i in range(502)
        ]
    )
    module = importlib.import_module("llm_behavior_eval.medhallu.dataset")
    monkeypatch.setattr(module, "load_hf_dataset", lambda *args, **kwargs: source)
    monkeypatch.setattr(
        base_evaluator,
        "CustomDataset",
        lambda *args, **kwargs: SimpleNamespace(
            preprocess=lambda *args, **kwargs: source
        ),
    )
    monkeypatch.setattr(base_evaluator, "DataLoader", lambda *args, **kwargs: None)
    limit = yaml.safe_load(parity_config_path(PACKAGE).read_text())["eval_config"][
        "limit"
    ]
    legacy = SimpleNamespace(
        dataset_config=SimpleNamespace(
            file_path="offline",
            dataset_type=None,
            dataset_id=DATASET_ID,
            preprocess_config=None,
            seed=42,
        ),
        eval_config=SimpleNamespace(
            model_token=None,
            dataset_token=None,
            max_answer_tokens=128,
            enable_thinking=False,
            enable_thinking_arg_name="enable_thinking",
            thinking_start_token="<think>",
            thinking_end_token="</think>",
            pass_max_answer_tokens=False,
            max_samples=500,
        ),
        tokenizer=None,
        trust_remote_code=False,
        data_collator=None,
        eval_engine=SimpleNamespace(
            set_dataset=lambda ds: None, get_batch_size=lambda: 1
        ),
    )
    base_evaluator.BaseEvaluator.prepare_dataloader(cast("Any", legacy))
    capped = list(dataset())[:limit]
    assert len(capped) == len(legacy.eval_dataset) == 500
    assert [s.id for s in capped] == [
        f"{DATASET_ID}:{i}" for i in legacy.eval_dataset["source_index"]
    ]
    assert [s.input for s in capped] == list(legacy.eval_dataset["question"])
    assert [s.target for s in capped] == list(legacy.eval_dataset["answer"])


def test_scorer_matches_legacy_aggregate(monkeypatch):
    from llm_behavior_eval.evaluation_utils.free_text_hallu_evaluator import (
        GRADER_TEMPLATE as LEGACY_TEMPLATE,
    )
    from llm_behavior_eval.evaluation_utils.free_text_hallu_evaluator import (
        FreeTextHaluEvaluator,
        _HalluGenerationRecord,
    )

    module = importlib.import_module("llm_behavior_eval.medhallu.scorer")
    raw = iter(["A", "B", "garbage", "C"])
    calls = []
    grader = get_model("mockllm/model")

    async def generate(prompt, **kwargs):
        # The scorer passes no GenerateConfig: the grader role's config applies.
        assert kwargs == {}
        calls.append(prompt)
        return ModelOutput.from_content("mockllm/model", next(raw))

    monkeypatch.setattr(grader, "generate", generate)
    monkeypatch.setattr(module, "get_model", lambda **kwargs: grader)
    stops = ["stop", "stop", "stop", "stop", "max_tokens", "unknown", None]
    scores = []
    for stop in stops:
        output = ModelOutput.from_content(
            "mockllm/model", "<think>trace</think> answer"
        )
        if stop is None:
            output.choices = []
        else:
            output.choices[0].stop_reason = cast("Any", stop)
        state = TaskState(
            model=ModelName("mockllm/model"),
            sample_id=1,
            epoch=1,
            input="question",
            messages=[ChatMessageUser(content="question")],
            output=output,
        )
        scores.append(
            SampleScore(score=asyncio.run(medhallu_scorer()(state, Target("gold"))))
        )
    assert len(calls) == 4
    assert calls[0] == LEGACY_TEMPLATE.format(
        question="question", target="gold", predicted_answer="answer"
    )
    metadata = [s.score.metadata or {} for s in scores]
    assert metadata[0]["raw_judge_text"] == "A"
    assert metadata[2]["parse_error"] and metadata[2]["grade"] == "NOT_ATTEMPTED"
    assert all(m["grade"] == "NOT_ATTEMPTED" and m["skipped"] for m in metadata[4:])
    assert [s.score.value for s in scores] == [1, 0, 1, 1, 1, 1, 1]

    legacy = FreeTextHaluEvaluator.__new__(FreeTextHaluEvaluator)
    legacy.num_samples = len(stops)
    monkeypatch.setattr(legacy, "_format_answers", lambda answers: answers)
    monkeypatch.setattr(
        legacy,
        "_grade_batch",
        lambda *args: ["CORRECT", "INCORRECT", "NOT_ATTEMPTED", "NOT_ATTEMPTED"],
    )
    saved = {}
    monkeypatch.setattr(legacy, "save_results", lambda **kwargs: saved.update(kwargs))
    legacy._grade_impl(
        [
            _HalluGenerationRecord(
                answers=["answer"] * len(stops),
                input_texts=["question"] * len(stops),
                gt_answers=["gold"] * len(stops),
                finish_reasons=["length" if s == "max_tokens" else s for s in stops],
            )
        ],
        cast("Any", object()),
    )
    assert mapping(medhallu_accuracy())(scores)["accuracy"] == pytest.approx(
        100 * saved["accuracy"]
    )
    assert (
        mapping(medhallu_empty_responses())(scores)["empty_responses"]
        == saved["empty_responses"]
        == 5
    )
    assert rate(incomplete_response_rate())(scores) == saved["incomplete_response_rate"]
    assert (
        mapping(medhallu_accuracy())(
            [SampleScore(score=Score(value=1, metadata={"grade": "NOT_ATTEMPTED"}))]
        )["accuracy"]
        == 100
    )


def test_code_sets_no_inference_settings():
    settings = {
        "config",
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "reasoning_effort",
        "reasoning_tokens",
        "enable_thinking",
        "extra_body",
    }
    for path in PACKAGE_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            assert not (isinstance(node, ast.Name) and node.id == "GenerateConfig")
            assert not (isinstance(node, ast.keyword) and node.arg in settings), (
                path.name,
                ast.dump(node),
            )


def test_native_task(tmp_path):
    path = snapshot(tmp_path, [{"question": "question", "answer": "gold"}])
    calls = []

    def answer(messages, tools, choice, config):
        calls.append(messages)
        # Nothing in the task, solver or scorer sets inference options.
        assert config == GenerateConfig()
        return ModelOutput.from_content("mockllm/model", "answer")

    task = medhallu(dataset_path=path)
    assert task.config == GenerateConfig()
    logs = inspect_eval(
        task,
        model=get_model("mockllm/model", custom_outputs=answer),
        model_roles={"grader": mock_grader("B")},
        log_dir=str(tmp_path / "logs"),
        display="none",
        log_format="json",
    )
    assert logs[0].status == "success", logs[0].error
    assert calls[0][0].text == SYSTEM_MESSAGE
    assert calls[0][-1].text == "question\n"
    assert logs[0].results is not None
    metrics = logs[0].results.scores[0].metrics
    assert set(metrics) == set(METRICS)
    assert metrics["accuracy"].value == 0
    assert metrics["empty_responses"].value == 0
    saved_log = json.loads(Path(logs[0].location).read_text())
    saved_score = saved_log["samples"][0]["scores"]["medhallu_scorer"]
    assert saved_score["value"] == 0
    assert saved_score["explanation"] == "B"
    metadata = saved_score["metadata"]
    assert metadata["raw_judge_text"] == "B"
    assert metadata["question"] == "question"
    assert metadata["dataset_id"] == DATASET_ID
    assert metadata["source_index"] == 0
    assert metadata["parity_gaps"] == PARITY_GAPS
    assert metadata["stop_reason"] == "stop"
    assert metadata["grade"] == "INCORRECT"
    assert not metadata["skipped"] and not metadata["parse_error"]


def test_grader_role_is_required(tmp_path):
    path = snapshot(tmp_path, [{"question": "q", "answer": "a"}])
    logs = inspect_eval(
        medhallu(dataset_path=path),
        model="mockllm/model",
        log_dir=str(tmp_path / "logs"),
        display="none",
        log_format="json",
    )
    assert logs[0].status == "error"


def test_documented_tokenizer_parity_gaps(monkeypatch, tmp_path):
    from llm_behavior_eval.evaluation_utils import custom_dataset

    class Tokenizer:
        def __call__(self, texts, *, max_length, **kwargs):
            ids = [[ord(c) for c in text[:max_length]] for text in texts]
            return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}

    monkeypatch.setattr(
        custom_dataset,
        "safe_apply_chat_template",
        lambda tokenizer, messages, **kwargs: "TEMPLATE:" + str(messages),
    )
    question, target = "q" * 1200, "a" * 300
    legacy = custom_dataset.free_text_preprocess_function(
        {"question": [question], "answer": [target]},
        cast("Any", Tokenizer()),
        1024,
        256,
        False,
    )
    assert legacy["test_input_ids"].shape[1] == 1024
    assert legacy["gt_answers"].shape[1] == 256
    decoded = "".join(chr(c) for c in legacy["test_input_ids"][0])
    assert decoded.startswith("TEMPLATE:") and decoded != question
    path = snapshot(tmp_path, [{"question": question, "answer": target}], "long")
    sample = dataset(dataset_path=path)[0]
    assert sample.input == question and sample.target == target
    assert sample.metadata is not None
    assert len(sample.metadata["parity_gaps"]) == 3


def test_independent_api_grader(tmp_path, monkeypatch):
    """Check independent API role routing with deterministic generation, not HTTP."""
    path = snapshot(tmp_path, [{"question": "q", "answer": "a"}])
    grader = get_model(
        "openai-api/airgap/judge",
        base_url="http://judge.invalid/v1",
        api_key="fixture-not-secret",
    )
    requests = []

    async def generate(prompt, **kwargs):
        requests.append((prompt, kwargs))
        return ModelOutput.from_content("judge", "B")

    monkeypatch.setattr(grader, "generate", generate)
    logs = inspect_eval(
        medhallu(dataset_path=path),
        model="mockllm/model",
        model_roles={"grader": grader},
        display="none",
        log_format="json",
        log_dir=str(tmp_path / "logs"),
    )
    assert logs[0].status == "success", logs[0].error
    assert str(cast("Any", grader.api).client.base_url) == "http://judge.invalid/v1/"
    assert requests == [(requests[0][0], {})]
    assert logs[0].results is not None
    assert logs[0].results.scores[0].metrics["accuracy"].value == 0


def run_cli(*args: str, cwd: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "inspect_ai", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    assert result.returncode == 0, result.stderr


def test_registry_discovery(tmp_path):
    from llm_behavior_eval import _registry

    assert "medhallu" in _registry.__all__
    assert vars(_registry)["medhallu"] is medhallu
    listed = subprocess.run(
        [sys.executable, "-m", "inspect_ai", "list", "tasks", "llm_behavior_eval"],
        capture_output=True,
        text=True,
    )
    assert listed.returncode == 0, listed.stderr
    assert "medhallu" in listed.stdout
    path = snapshot(tmp_path, [{"question": "q", "answer": "a"}])
    # cwd=tmp_path forces package entry-point resolution.
    run_cli(
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
        "--display",
        "none",
        cwd=tmp_path,
    )
    log = read_eval_log(list_eval_logs(str(tmp_path / "logs"))[0])
    assert log.status == "success"
    assert log.results is not None
    assert set(log.results.scores[0].metrics) == set(METRICS)


def test_target_metrics_entry(tmp_path):
    path = parity_config_path(PACKAGE).with_name(TARGET_METRICS_FILE)
    assert path == PACKAGE_DIR / "target_metrics.json"
    config = read_target_metrics_file(path)
    assert list(config) == ["llm_behavior_eval/medhallu"]
    targets = config["llm_behavior_eval/medhallu"]
    assert load_target_metrics()["llm_behavior_eval/medhallu"] == targets
    entry = {m["name"]: m for m in targets}
    assert list(entry) == METRICS
    assert entry["accuracy"] == {
        "name": "accuracy",
        "is_percentage": True,
        "is_higher_better": True,
        "is_normalized": False,
    }
    assert not entry["empty_responses"]["is_higher_better"]
    assert entry["incomplete_response_rate"]["is_normalized"]
    # The names match those in a real .eval log.
    data = snapshot(tmp_path, [{"question": "q", "answer": "a"}])
    log = inspect_eval(
        medhallu(dataset_path=data),
        model="mockllm/model",
        model_roles={"grader": mock_grader("A")},
        log_dir=str(tmp_path / "logs"),
        display="none",
    )[0]
    assert log.status == "success", log.error
    assert Path(log.location).suffix == ".eval"
    # inspect-brief finds every target in the written log.
    assert missing_target_metrics(log.location) == []


def test_metric_registry_names_are_benchmark_specific(tmp_path):
    """Logs name Med-Hallu's own metrics, so rebuilding them cannot pick another
    benchmark's same-named metric; the reported names stay unchanged."""
    path = snapshot(
        tmp_path, [{"question": "q", "answer": "a"}, {"question": "r", "answer": "b"}]
    )
    log = inspect_eval(
        medhallu(dataset_path=path),
        model="mockllm/model",
        model_roles={"grader": mock_grader("B")},
        log_dir=str(tmp_path / "logs"),
        display="none",
    )[0]
    assert log.status == "success", log.error
    assert log.eval.scorers is not None
    assert log.eval.scorers[0].name == "medhallu_scorer"
    assert [m.name for m in cast("Any", log.eval.scorers[0].metrics)] == [
        "llm_behavior_eval/medhallu_accuracy",
        "llm_behavior_eval/medhallu_empty_responses",
        "llm_behavior_eval/incomplete_response_rate",
    ]
    assert log.results is not None
    reported = {k: m.value for k, m in log.results.scores[0].metrics.items()}
    assert reported == {
        "accuracy": 0.0,
        "empty_responses": 0,
        "incomplete_response_rate": 0.0,
    }
    recompute_metrics(log)
    assert {k: m.value for k, m in log.results.scores[0].metrics.items()} == reported


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


def test_shipped_parity_config_holds_legacy_settings():
    from llm_behavior_eval.evaluation_utils.dataset_config import DatasetConfig
    from llm_behavior_eval.evaluation_utils.eval_config import (
        FAMILY_TOKEN_DEFAULTS,
        EvaluationConfig,
    )

    path = parity_config_path(PACKAGE)
    assert path == PACKAGE_DIR / "parity.yaml"
    config = yaml.safe_load(path.read_text())
    assert set(config) == {"generate_config", "eval_config", "model_roles"}
    legacy = FAMILY_TOKEN_DEFAULTS["hallucination"]
    seed = DatasetConfig.model_fields["seed"].default
    answer = GenerateConfig.model_validate(config["generate_config"])
    assert not EvaluationConfig.model_fields["sample"].default
    assert answer.temperature == 0.0
    assert answer.max_tokens == legacy["max_answer_tokens"] == 128
    assert answer.seed == seed == 42
    assert legacy["enable_thinking"] is False
    assert answer.extra_body == ANSWER_BODY
    assert (
        config["eval_config"]["limit"]
        == EvaluationConfig.model_fields["max_samples"].default
        == 500
    )
    grader = config["model_roles"]["grader"]
    assert grader["model"] == "hf/google/gemma-3-12b-it"
    judge = GenerateConfig.model_validate(grader["config"])
    assert legacy["sample_judge"] is False
    assert judge.temperature == 0.0
    assert judge.max_tokens == legacy["max_judge_tokens"] == 32
    assert judge.seed == 42
    assert grader["args"] == {"do_sample": False, "enable_thinking": False}


@pytest.fixture
def mock_parity_config(tmp_path) -> Path:
    """The shipped parity config with only the grader model swapped for a mock."""
    config = yaml.safe_load(parity_config_path(PACKAGE).read_text())
    config["model_roles"]["grader"]["model"] = "mockllm/model"
    config["model_roles"]["grader"]["args"] = {}
    path = tmp_path / "parity.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def assert_parity_recorded(spec: EvalSpec) -> None:
    generate = spec.model_generate_config
    assert (generate.temperature, generate.max_tokens, generate.seed) == (0.0, 128, 42)
    assert spec.config.limit == 500
    assert spec.model_roles
    grader = spec.model_roles["grader"]
    assert not isinstance(grader, list)
    assert (grader.config.temperature, grader.config.max_tokens) == (0.0, 32)


def test_parity_config_cli(tmp_path, mock_parity_config):
    path = snapshot(
        tmp_path, [{"question": f"q{i}", "answer": "a"} for i in range(502)]
    )
    run_cli(
        "eval",
        "llm_behavior_eval/medhallu",
        "--model",
        "mockllm/model",
        "--run-config",
        str(mock_parity_config),
        "-T",
        f"dataset_path={path}",
        "--log-dir",
        str(tmp_path / "logs"),
        "--display",
        "none",
        cwd=tmp_path,
    )
    log = read_eval_log(list_eval_logs(str(tmp_path / "logs"))[0])
    assert log.status == "success"
    assert_parity_recorded(log.eval)
    order = Dataset.from_dict({"index": list(range(502))}).shuffle(seed=42)["index"]
    assert log.samples is not None
    # Logs store samples in completion order; the selected set is what matters.
    ids = [s.id for s in log.samples]
    assert len(ids) == 500
    assert set(ids) == {f"{DATASET_ID}:{i}" for i in order[:500]}


def test_parity_config_python(tmp_path, mock_parity_config):
    path = snapshot(tmp_path, [{"question": "q", "answer": "a"}])
    logs = inspect_eval(
        medhallu(dataset_path=path),
        model="mockllm/model",
        log_dir=str(tmp_path / "logs"),
        display="none",
        **read_parity_config(mock_parity_config),
    )
    assert logs[0].status == "success", logs[0].error
    assert_parity_recorded(logs[0].eval)
