import importlib
import subprocess
import sys
from math import isnan
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml
from datasets import Dataset
from inspect_ai import eval as _inspect_eval
from inspect_ai.log import list_eval_logs, read_eval_log
from inspect_ai.model import GenerateConfig, Model, ModelOutput, get_model

from llm_behavior_eval.common.parity import parity_config_path, read_parity_config
from llm_behavior_eval.common.target_metrics import (
    TARGET_METRICS_FILE,
    load_target_metrics,
    missing_target_metrics,
    read_target_metrics_file,
)
from llm_behavior_eval.prompt_injection.dataset import DATASET_ID, dataset
from llm_behavior_eval.prompt_injection.scorer import parse_yes_no
from llm_behavior_eval.prompt_injection.task import prompt_injection

PACKAGE = "llm_behavior_eval.prompt_injection"
TASK = "llm_behavior_eval/prompt_injection"
REPO = Path(__file__).parents[1]


def inspect_eval(*args, **kwargs) -> Any:
    # JSON avoids lazy binary-log loading hangs in some sandboxes.
    return _inspect_eval(*args, log_format="json", display="none", **kwargs)


@pytest.fixture
def source(monkeypatch):
    module = importlib.import_module("llm_behavior_eval.prompt_injection.dataset")

    def install(rows):
        monkeypatch.setattr(
            module, "load_hf_dataset", lambda *a, **k: Dataset.from_list(rows)
        )

    return install


@pytest.fixture
def mock_parity_config(tmp_path) -> Path:
    """The shipped parity config with only the grader model swapped for a mock."""
    config = yaml.safe_load(parity_config_path(PACKAGE).read_text())
    config["model_roles"]["grader"]["model"] = "mockllm/judge"
    config["model_roles"]["grader"]["args"] = {}
    path = tmp_path / "parity.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _metrics(log) -> dict[str, float]:
    return {
        name: metric.value for name, metric in log.results.scores[0].metrics.items()
    }


# --- Registry and settings ---------------------------------------------------


def test_registry_discovery():
    from llm_behavior_eval import _registry

    assert "prompt_injection" in _registry.__all__
    assert vars(_registry)["prompt_injection"] is prompt_injection
    listed = subprocess.run(
        [sys.executable, "-m", "inspect_ai", "list", "tasks", "llm_behavior_eval"],
        capture_output=True,
        text=True,
    )
    assert listed.returncode == 0, listed.stderr
    assert "prompt_injection" in listed.stdout


def test_task_sets_no_inference_settings(source):
    source([dict(question="q", answer="a")])
    task = prompt_injection()
    assert task.config == GenerateConfig()
    assert not task.model_roles


def test_shipped_parity_config_holds_legacy_settings():
    from llm_behavior_eval.evaluation_utils.eval_config import (
        FAMILY_TOKEN_DEFAULTS,
        EvaluationConfig,
    )

    path = parity_config_path(PACKAGE)
    assert path == REPO / "llm_behavior_eval" / "prompt_injection" / "parity.yaml"
    config = yaml.safe_load(path.read_text())
    assert set(config) == {"generate_config", "eval_config", "model_roles"}
    legacy = FAMILY_TOKEN_DEFAULTS["prompt-injection"]
    legacy_defaults = EvaluationConfig.model_fields
    assert config["generate_config"] == {
        "temperature": 0.0,
        "max_tokens": legacy["max_answer_tokens"],
    }
    assert legacy_defaults["sample"].default is False
    assert config["eval_config"] == {"limit": legacy_defaults["max_samples"].default}
    grader = config["model_roles"]["grader"]
    assert grader["model"] == "hf/google/gemma-3-12b-it"
    assert grader["config"] == {
        "temperature": 0.0,
        "max_tokens": legacy["max_judge_tokens"],
    }
    assert legacy["sample_judge"] is False and legacy["enable_thinking"] is False
    assert grader["args"] == {"do_sample": False, "enable_thinking": False}
    assert config["generate_config"]["max_tokens"] == 128  # not HPLT-338's 512


def _assert_parity_recorded(log) -> None:
    spec = log.eval
    # Eval-level GenerateConfig (the model under test) is recorded in the plan.
    generate = log.plan.config
    assert (generate.temperature, generate.max_tokens) == (0.0, 128)
    assert spec.config.limit == 500
    grader = spec.model_roles["grader"]
    assert (grader.config.temperature, grader.config.max_tokens) == (0.0, 32)


def test_parity_config_python(tmp_path, source, mock_parity_config):
    source([dict(question=f"q{i}", answer="a") for i in range(3)])
    seen = []

    def answer(messages, tools, choice, config):
        assert (config.temperature, config.max_tokens) == (0.0, 128)
        seen.append(config)
        return ModelOutput.from_content("mockllm/answer", "No way.")

    log = inspect_eval(
        prompt_injection(),
        model=get_model("mockllm/answer", custom_outputs=answer),
        log_dir=str(tmp_path),
        **read_parity_config(mock_parity_config),
    )[0]
    assert log.status == "success", log.error
    _assert_parity_recorded(log)
    assert len(seen) == 3


def test_cli_parity_run_and_target_metrics(tmp_path, mock_parity_config):
    """Registered CLI run on a local snapshot; metric names match target_metrics."""
    data = tmp_path / "data"
    Dataset.from_list([dict(question="q", answer="a")]).save_to_disk(str(data))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "inspect_ai",
            "eval",
            TASK,
            "--model",
            "mockllm/answer",
            "-T",
            f"dataset_path={data}",
            "--run-config",
            str(mock_parity_config),
            "--log-dir",
            str(tmp_path / "logs"),
            "--display",
            "none",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    logs = list_eval_logs(str(tmp_path / "logs"))
    assert len(logs) == 1
    log = read_eval_log(logs[0])
    assert log.status == "success", log.error
    _assert_parity_recorded(log)
    path = parity_config_path(PACKAGE).with_name(TARGET_METRICS_FILE)
    assert (
        path == REPO / "llm_behavior_eval" / "prompt_injection" / "target_metrics.json"
    )
    targets = read_target_metrics_file(path)
    assert list(targets) == [TASK]
    assert load_target_metrics()[TASK] == targets[TASK]
    declared = [entry["name"] for entry in targets[TASK]]
    assert declared == list(_metrics(log))
    # inspect-brief finds every target in the written log.
    assert missing_target_metrics(log) == []


def test_registry_names_are_package_specific():
    # Inspect's registry keeps the last object registered under a name, so a
    # generic name (`attack_success_rate`) reused by another benchmark would
    # replace this package's when a log is rescored; see
    # _registry.check_unique_registry_names.
    from llm_behavior_eval import _registry

    names = {
        name
        for (_, name), modules in _registry.registry_definitions().items()
        if any(module.startswith(f"{PACKAGE}.") for module in modules)
    }
    assert names == {
        "llm_behavior_eval/prompt_injection",
        "llm_behavior_eval/prompt_injection_metrics",
        "llm_behavior_eval/prompt_injection_scorer",
    }


# --- Dataset conversion ------------------------------------------------------


def test_dataset_legacy_order_and_overrides(source):
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
    assert len(data) == 503  # the cap is Inspect's limit, not conversion's
    expected = (
        Dataset.from_list(rows).add_column("idx", list(range(503))).shuffle(seed=42)
    )
    assert [s.id for s in data] == [f"{DATASET_ID}:{i}" for i in expected["idx"]]
    first: Any = next(s for s in data if s.metadata["source_index"] == 0)
    assert first.input[0].text == "override"
    assert first.input[1].text == "q0\n"
    assert first.metadata["judge_question"] == "j"
    assert first.metadata["judge_question_source"] == "override"
    other: Any = next(s for s in data if s.metadata["source_index"] == 1)
    assert other.metadata["judge_question"] == "q1"
    assert other.metadata["judge_question_source"] == "question"
    source(rows + [dict(question="", answer="a")])
    with pytest.raises(ValueError, match="required nonempty text question"):
        dataset()
    source([dict(question="q", answer="a", judge_question=3)])
    with pytest.raises(ValueError, match="optional text judge_question"):
        dataset()


def test_limit_selects_legacy_rows(tmp_path, source):
    """Inspect's limit after conversion equals legacy shuffle(seed=42).select(500)."""
    rows = [dict(question=f"q{i}", answer="a") for i in range(503)]
    source(rows)
    log = inspect_eval(
        prompt_injection(),
        model="mockllm/answer",
        model_roles={"grader": get_model("mockllm/judge")},
        log_dir=str(tmp_path),
        limit=500,
    )[0]
    assert log.status == "success", log.error
    legacy = Dataset.from_list(rows).shuffle(seed=42)[:500]
    assert sorted(s.metadata["question"] for s in log.samples) == sorted(
        legacy["question"]
    )


def test_default_system_prompt_matches_legacy(source):
    from llm_behavior_eval.evaluation_utils.prompts import FREE_TEXT_SYSTEM_MESSAGE

    source([dict(question="q", answer="a", system_prompt=None)])
    sample: Any = dataset()[0]
    assert sample.input[0].text == FREE_TEXT_SYSTEM_MESSAGE["content"]


def test_offline_snapshot(tmp_path):
    Dataset.from_list([dict(question="offline", answer="target")]).save_to_disk(
        str(tmp_path / "data")
    )
    loaded: Any = dataset(dataset_path=str(tmp_path / "data"))
    assert loaded[0].id == DATASET_ID + ":0"
    assert loaded[0].metadata["question"] == "offline"


def test_current_preprocessor_and_explicit_truncation_gap(source, monkeypatch):
    from llm_behavior_eval.evaluation_utils import custom_dataset

    captured = []
    calls = []

    def template(tokenizer, messages, **kwargs):
        captured.append(messages)
        return "CHAT:" + messages[-1]["content"]

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            calls.append((texts, kwargs))
            limit = kwargs["max_length"]
            return {
                "input_ids": [[ord(c) for c in t[:limit]] for t in texts],
                "attention_mask": [[1] * min(len(t), limit) for t in texts],
            }

    monkeypatch.setattr(custom_dataset, "safe_apply_chat_template", template)
    question = "q" * 1100
    source([dict(question=question, answer="a")])
    native: Any = dataset()[0]
    legacy = custom_dataset.free_text_preprocess_function(
        {"question": [question], "answer": ["a"]},
        cast("Any", Tokenizer()),
        max_length=1024,
        gt_max_length=128,
        has_stereotype=False,
    )
    assert captured[0] == [
        {"role": "system", "content": native.input[0].text},
        {"role": "user", "content": native.input[1].text},
    ]
    assert calls[2][0] == [native.metadata["judge_question"]] == [question]
    assert legacy["test_input_ids"].shape[1] == 1024
    assert legacy["judge_questions"].shape[1] == 128
    assert len(native.input[1].text) == 1101  # Unresolved native truncation difference.


# --- Scoring -----------------------------------------------------------------


@pytest.mark.parametrize(
    "raw", ["Yes", "NO!", "yesterday", "", "No then yes", "YES/no", "  yes  "]
)
def test_parser_parity(raw):
    from llm_behavior_eval.evaluation_utils.free_text_injection_evaluator import (
        FreeTextPromptInjectionEvaluator,
    )

    assert (
        parse_yes_no(raw)
        == FreeTextPromptInjectionEvaluator._map_judge_outputs_yes_no(
            [[{"generated_text": raw}]]
        )[0]
    )


def test_prompt_matches_legacy():
    from llm_behavior_eval.evaluation_utils.free_text_injection_evaluator import (
        FreeTextPromptInjectionEvaluator,
    )
    from llm_behavior_eval.prompt_injection.prompts import (
        JUDGE_PROMPT,
        PROMPT_JUDGE_RESPONSE,
    )

    assert (
        PROMPT_JUDGE_RESPONSE == FreeTextPromptInjectionEvaluator.PROMPT_JUDGE_RESPONSE
    )
    assert JUDGE_PROMPT.format(
        answer="A", metadata={"judge_question": "Q"}
    ) == PROMPT_JUDGE_RESPONSE.format(llm_response="A", question="Q")


def test_native_legacy_grade_parity(source, tmp_path, monkeypatch):
    from llm_behavior_eval.evaluation_utils.free_text_injection_evaluator import (
        FreeTextPromptInjectionEvaluator,
        _InjectionGenerationRecord,
    )

    raw = ["Yes", "malformed", "No", "unused"]
    answers = ["<think>secret</think>a0", "a1", "a2", "a3"]
    source([dict(question=f"q{i}", answer="a") for i in range(4)])

    def answer(messages, tools, choice, config):
        i = int(messages[-1].text.strip()[1:])
        output = ModelOutput.from_content("mockllm/answer", answers[i])
        if i == 3:
            output.choices[0].stop_reason = "max_tokens"
        return output

    def judge(messages, tools, choice, config):
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
    )[0]
    assert log.status == "success", log.error
    legacy: Any = object.__new__(FreeTextPromptInjectionEvaluator)
    legacy.eval_config = SimpleNamespace(
        exclude_thinking_trace_for_judge=True,
        thinking_start_token="<think>",
        thinking_end_token="</think>",
    )
    monkeypatch.setattr(legacy, "_grade_batch", lambda *args: ["Yes", "No", "No"])
    saved = {}
    monkeypatch.setattr(legacy, "save_results", lambda **kw: saved.update(kw))
    legacy._grade_impl(
        [
            _InjectionGenerationRecord(
                input_texts=["q0", "q1", "q2", "q3"],
                gt_answers=["a"] * 4,
                answers=answers,
                finish_reasons=cast("list[str | None]", ["stop"] * 3 + ["length"]),
                judge_questions=["q0", "q1", "q2", "q3"],
                response_tokens=[2] * 4,
            )
        ],
        judge_engine=object(),
    )
    metrics = _metrics(log)
    assert metrics["attack_success_rate"] == pytest.approx(
        100 * (1 - saved["accuracy"])
    )
    assert (
        metrics["incomplete_response_rate"] == saved["incomplete_response_rate"] == 0.25
    )
    scores = [next(iter(s.scores.values())) for s in log.samples]
    assert sum(bool(s.metadata.get("skipped")) for s in scores) == 1
    assert sorted(s.metadata["raw_judge_text"] for s in scores) == [
        "",
        "No",
        "Yes",
        "malformed",
    ]
    assert sorted(s.answer for s in scores if not s.metadata.get("skipped")) == [
        "No",
        "No",
        "Yes",
    ]
    assert all(s.metadata["response_tokens"] is None for s in scores)
    assert all("1024-token" in s.metadata["prompt_parity"] for s in scores)


def test_requires_grader(source, tmp_path):
    source([dict(question="q", answer="a")])
    log = inspect_eval(
        prompt_injection(),
        model="mockllm/answer",
        log_dir=str(tmp_path),
    )[0]
    assert log.status == "error"
    assert "grader" in log.error.message


@pytest.mark.parametrize(
    "stop", ["max_tokens", "model_length", "unknown", "content_filter"]
)
def test_token_count_and_all_skipped(source, tmp_path, monkeypatch, stop):
    from transformers import AutoTokenizer

    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert text == "formatted" and add_special_tokens is False
            return [1, 2, 3]

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer())
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
    )[0]
    assert log.status == "success", log.error
    score = next(iter(log.samples[0].scores.values()))
    assert score.metadata["response_tokens"] == 3
    assert score.metadata["formatted_answer"] == "formatted"
    metrics = _metrics(log)
    assert isnan(metrics["attack_success_rate"])
    assert metrics["incomplete_response_rate"] == int(stop == "max_tokens")


def test_independent_api_role_without_code_settings(tmp_path, monkeypatch, source):
    """Role routing fixture, not HTTP/provider inference parity."""
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
            assert "config" not in kwargs or kwargs["config"] == GenerateConfig()
            return ModelOutput.from_content("judge", "Yes")
        return await original(self, input, *args, **kwargs)

    monkeypatch.setattr(Model, "generate", generate)
    log = inspect_eval(
        prompt_injection(),
        model="mockllm/answer",
        model_roles={"grader": grader},
        log_dir=str(tmp_path),
    )[0]
    assert log.status == "success", log.error
    assert len(calls) == 1 and "Question: override?" in calls[0]
    assert _metrics(log)["attack_success_rate"] == 100
