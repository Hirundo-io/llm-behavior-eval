import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from datasets import Dataset
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log
from inspect_ai.model import GenerateConfig, Model, ModelOutput, get_model
from inspect_ai.scorer import SampleScore, Score

from llm_behavior_eval.bbq import prompts
from llm_behavior_eval.bbq.dataset import BIAS_TYPES, KINDS, dataset
from llm_behavior_eval.bbq.metrics import bbq_metrics
from llm_behavior_eval.bbq.scorer import classify, parse_agreement, parse_uncertainty
from llm_behavior_eval.bbq.task import bbq
from llm_behavior_eval.common.parity import parity_config_path, read_parity_config
from llm_behavior_eval.common.target_metrics import (
    TARGET_METRICS_FILE,
    load_target_metrics,
    missing_target_metrics,
    read_target_metrics_file,
)
from tests.golden import golden

REPO = Path(__file__).parents[1]
PACKAGE = "llm_behavior_eval.bbq"
SCORER = "bbq_scorer"
# Frozen outputs of the removed legacy engine (tests/golden/README.md).
LEGACY = golden("bbq")
GENERATION_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "seed",
    "reasoning_effort",
    "reasoning_tokens",
    "extra_body",
)


def generation_settings(config: GenerateConfig) -> dict[str, Any]:
    return {
        name: getattr(config, name)
        for name in GENERATION_FIELDS
        if getattr(config, name) is not None
    }


def scores_of(log: EvalLog) -> list[Score]:
    assert log.status == "success", log.error
    assert log.samples
    result = []
    for sample in log.samples:
        assert sample.scores
        result.append(sample.scores[SCORER])
    return result


def metrics_of(log: EvalLog) -> dict[str, float]:
    assert log.results
    return {k: v.value for k, v in log.results.scores[0].metrics.items()}


@pytest.fixture
def source(monkeypatch):
    module = importlib.import_module("llm_behavior_eval.bbq.dataset")

    def install(rows):
        monkeypatch.setattr(
            module, "load_hf_dataset", lambda *a, **k: Dataset.from_list(rows)
        )

    return install


# --- prompts and parsing -----------------------------------------------------


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
    # Legacy extract_agreements(...)[0].value on the same raw judge text.
    assert parse_agreement(raw) == LEGACY["agreement"][raw]


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
    # Legacy map_uncertainty_outputs(...)[0] on the same raw judge text.
    assert parse_uncertainty(raw) == LEGACY["uncertainty"][raw]


def test_exact_legacy_constants():
    constants = LEGACY["constants"]
    assert prompts.SYSTEM_MESSAGE == constants["system_message"]
    assert list(prompts.UNKNOWN_ANSWERS) == constants["unknown_answers"]
    for key in ("AGREEMENT_CHECK_SYSTEM_PROMPT", "UNCERTAINTY_CHECK_SYSTEM_PROMPT"):
        assert getattr(prompts, key) == constants[key.lower()]
    # Legacy _JUDGE_OUTPUT_RE and _UNCERT_RE.
    patterns = LEGACY["patterns"]
    assert prompts.AGREEMENT_PATTERN.pattern == patterns["agreement"]["pattern"]
    assert prompts.AGREEMENT_PATTERN.flags == patterns["agreement"]["flags"]
    assert prompts.UNCERTAINTY_PATTERN.pattern == patterns["uncertainty"]["pattern"]
    assert prompts.UNCERTAINTY_PATTERN.flags == patterns["uncertainty"]["flags"]


def test_judge_messages_match_legacy(tmp_path, source):
    """The grader receives exactly the legacy judge messages (raw question)."""
    source([dict(question="q0", answer="person", stereotyped_answer="stereo")])
    seen: list[list[dict[str, str]]] = []

    def judge(messages, tools, choice, config):
        seen.append([{"role": m.role, "content": m.text} for m in messages])
        return ModelOutput.from_content("mockllm/grader", 'gt_agree: "yes"')

    logs = inspect_eval(
        bbq(),
        model=get_model(
            "mockllm/answer",
            custom_outputs=[ModelOutput.from_content("mockllm/answer", "a0")],
        ),
        model_roles={"grader": get_model("mockllm/grader", custom_outputs=judge)},
        log_dir=str(tmp_path),
        display="none",
    )
    scores_of(logs[0])

    # The messages legacy FreeTextBiasEvaluator._match_llm_answers rendered for
    # answers ["a0"], targets ["person"], stereotypes ["stereo"], questions ["q0"].
    # Legacy batches all GT prompts, then all stereotype prompts, then all
    # uncertainty prompts; with one sample the order matches the native calls.
    assert seen == LEGACY["judge_messages"]


# --- dataset -----------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("bias_type", BIAS_TYPES)
def test_all_splits_reachable(source, bias_type, kind):
    source([dict(question="q", answer="a")])
    task = bbq(bias_type=bias_type, kind=kind)
    assert task.dataset.name == f"hirundo-io/bbq-{bias_type}-{kind}-free-text"


def test_rejects_unknown_split():
    with pytest.raises(ValueError, match="bias_type"):
        dataset("height", "bias")
    with pytest.raises(ValueError, match="kind"):
        dataset("gender", "ambiguous")


def test_full_split_in_legacy_shuffle_order(source):
    source([dict(question=f"q{i}", answer="a") for i in range(502)])
    data = dataset("gender", "bias")
    # Conversion keeps every row; Inspect's limit does the capping.
    assert len(data) == 502
    legacy_order = Dataset.from_list([{"i": i} for i in range(502)]).shuffle(seed=42)
    assert [s.id for s in data] == [
        f"hirundo-io/bbq-gender-bias-free-text:{i}" for i in legacy_order["i"]
    ]
    assert [s.input for s in data] == [f"q{i}" for i in legacy_order["i"]]


def test_validation(source):
    source([dict(question="q", answer="a")] * 3 + [dict(question="q", answer=None)])
    with pytest.raises(ValueError, match="required text"):
        dataset("gender", "bias")
    source([dict(question="q", answer="a", stereotyped_answer=None)])
    with pytest.raises(ValueError, match="required text"):
        dataset("gender", "bias")


def test_offline_snapshot(tmp_path):
    Dataset.from_list(
        [dict(question="q", answer="a", stereotyped_answer="s")]
    ).save_to_disk(tmp_path / "data")
    data = dataset("race", "unbias", dataset_path=str(tmp_path / "data"))
    assert (data[0].metadata or {})["has_stereotype"]
    assert (data[0].metadata or {})["stereotyped_answer"] == "s"
    assert data[0].id == "hirundo-io/bbq-race-unbias-free-text:0"
    assert data[0].target == "a"


# --- scorer and metrics ------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("has_stereotype", [True, False])
def test_legacy_grade_parity(tmp_path, source, kind, has_stereotype):
    targets = ["person", "Unknown", "person", "person", "person"]
    gt = ["yes", "no", "unmatched", "no"]
    uncertainty = ["no", "yes", "yes", "no"]
    stereo = ["yes", "yes", "yes", "yes"]
    finish_reasons = ["stop", "stop", "stop", "stop", "length"]
    # The fixture the frozen legacy _grade_impl results were captured on.
    assert LEGACY["grade_inputs"] == dict(
        targets=targets,
        gt=gt,
        uncertainty=uncertainty,
        stereo=stereo,
        finish_reasons=finish_reasons,
    )
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
        assert generation_settings(config) == {}
        question = messages[-1].text
        assert question.endswith("\n")
        i = int(question.strip()[1:])
        out = ModelOutput.from_content("mockllm/answer", f"a{i}")
        if i == 4:
            out.choices[0].stop_reason = "max_tokens"
        return out

    def judge(messages, tools, choice, config):
        assert generation_settings(config) == {}
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
    assert logs[0].samples
    by_id = {str(s.id): (s.scores or {})[SCORER] for s in logs[0].samples}
    scores = [by_id[f"hirundo-io/bbq-gender-{kind}-free-text:{i}"] for i in range(5)]
    assert [(s.metadata or {})["empty"] for s in scores] == [0, 0, 1, 0, 1]
    assert (scores[2].metadata or {})["correct"] == 0
    assert (scores[-1].metadata or {})["raw_judge_text"] == dict(
        gt="", stereotype="", uncertainty=""
    )
    assert (scores[0].metadata or {})["judge_question"] == "q0"
    assert "1024" in (scores[0].metadata or {})["prompt_parity"]

    metrics = metrics_of(logs[0])
    # Legacy FreeTextBiasEvaluator._grade_impl save_results on the same fixture
    # (the kind's dataset type, one record, finish_reasons above, stubbed judges).
    saved = LEGACY["grade"][kind][str(has_stereotype)]
    assert metrics["accuracy"] == saved["accuracy"]
    assert metrics["error"] == 1 - saved["accuracy"]
    assert metrics["empty_responses"] == saved["empty_responses"]
    assert metrics["incomplete_response_rate"] == saved["incomplete_response_rate"]
    if has_stereotype:
        assert metrics["stereotype_bias"] == saved["stereotype_bias"]
    else:
        assert "stereotype_bias" not in metrics


def test_classification_and_zero_stereotypes():
    flags = classify("Unknown", "unmatched", "yes", "yes", True)
    assert flags == dict(correct=1, empty=1, unknown=0, stereotyped=0)
    for kind in KINDS:
        metrics = cast("Any", bbq_metrics(kind, True))(
            [SampleScore(score=Score(value=0, metadata=flags))]
        )
        assert metrics["stereotype_bias"] == 0
    assert classify("person", "no", "yes", "yes", True)["unknown"] == 1
    assert classify("person", "no", "yes", "unparseable", True)["stereotyped"] == 1
    assert classify("person", "no", "yes", "no", False)["stereotyped"] == 0


@pytest.mark.parametrize("kind", KINDS)
def test_stereotype_bias_formula(kind):
    def sample(**flags):
        full = dict(correct=0, empty=0, unknown=0, stereotyped=0) | flags
        return SampleScore(score=Score(value=full["correct"], metadata=full))

    scores = [
        sample(stereotyped=1),
        sample(stereotyped=1),
        sample(correct=1),
        sample(unknown=1),
        sample(empty=1),
    ]
    metrics = cast("Any", bbq_metrics(kind, True))(scores)
    bias = 2 * 2 / (5 - 1 - 1) - 1
    accuracy = 1 / 5
    assert metrics["accuracy"] == accuracy
    expected = bias * (1 - accuracy) if kind == "unbias" else bias
    assert metrics["stereotype_bias"] == pytest.approx(expected)


def test_missing_grader(tmp_path, source):
    source([dict(question="q", answer="a")])
    logs = inspect_eval(
        bbq(), model="mockllm/model", display="none", log_dir=str(tmp_path)
    )
    assert logs[0].status == "error"
    assert logs[0].error
    assert "grader" in logs[0].error.message


@pytest.mark.parametrize(
    "stop", ["max_tokens", "model_length", "unknown", "content_filter"]
)
def test_nonstop_never_judged(tmp_path, source, stop):
    source([dict(question="q", answer="Unknown", stereotyped_answer="s")])
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
    (score,) = scores_of(logs[0])
    assert score.metadata
    assert score.value == 0 and score.metadata["empty"] == 1
    metrics = metrics_of(logs[0])
    assert metrics["accuracy"] == 0
    assert metrics["incomplete_response_rate"] == (stop == "max_tokens")


def test_thinking_trace_stripped_before_judging(tmp_path, source):
    source([dict(question="q", answer="a")])
    seen = []

    def judge(messages, tools, choice, config):
        seen.append(messages[-1].text)
        return ModelOutput.from_content("mockllm/grader", 'gt_agree: "yes"')

    logs = inspect_eval(
        bbq(),
        model=get_model(
            "mockllm/answer",
            custom_outputs=[
                ModelOutput.from_content("mockllm/answer", "<think>hmm</think> a")
            ],
        ),
        model_roles={"grader": get_model("mockllm/grader", custom_outputs=judge)},
        display="none",
        log_dir=str(tmp_path),
    )
    (score,) = scores_of(logs[0])
    assert score.answer == "a"
    assert all("hmm" not in text and "Candidate answer: a\n" in text for text in seen)


# --- inference settings and providers ----------------------------------------


def test_task_sets_no_inference_settings(source):
    source([dict(question="q", answer="a")])
    assert bbq().config == GenerateConfig()


def test_independent_api_grader_and_long_prompt(tmp_path, monkeypatch, source):
    """Native openai-api role routing, without network or provider inference."""
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
            # The scorer passes no GenerateConfig to the grader.
            assert (
                not args
                and generation_settings(kwargs.get("config", GenerateConfig())) == {}
            )
            text = "gt_agree: yes" if len(calls) == 1 else "candidate_uncertain: no"
            return ModelOutput.from_content("judge", text)
        return await original(self, input, *args, **kwargs)

    monkeypatch.setattr(Model, "generate", generate)
    logs = inspect_eval(
        bbq(),
        model="mockllm/answer",
        model_roles={"grader": grader},
        display="none",
        log_dir=str(tmp_path),
    )
    (score,) = scores_of(logs[0])
    assert len(calls) == 2
    # Lock down the documented legacy decoding/truncation divergence.
    assert question in calls[0][-1].text
    assert "Read the context carefully" not in calls[0][-1].text
    assert score.value == 1


# --- parity config -----------------------------------------------------------


# parity.yaml request bodies (vLLM/OpenAI-compatible): legacy thinking off, and
# the legacy prompt tokenization (special tokens; answers truncated to 1024).
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
    path = parity_config_path(PACKAGE)
    assert path.parent == REPO / "llm_behavior_eval" / "bbq"
    config = yaml.safe_load(path.read_text())
    assert set(config) == {"generate_config", "eval_config", "model_roles"}
    # FAMILY_TOKEN_DEFAULTS["bias"], EvaluationConfig.sample=False, seed 42,
    # and the legacy 1024-token, special-token prompt tokenization.
    answer = GenerateConfig.model_validate(config["generate_config"])
    assert generation_settings(answer) == {
        "temperature": 0.0,
        "max_tokens": 128,
        "seed": 42,
        "extra_body": ANSWER_BODY,
    }
    assert config["eval_config"] == {"limit": 500}
    grader = config["model_roles"]["grader"]
    assert grader["model"] == "hf/google/gemma-3-12b-it"
    assert generation_settings(GenerateConfig.model_validate(grader["config"])) == {
        "temperature": 0.0,
        "max_tokens": 32,
        "seed": 42,
        "extra_body": GRADER_BODY,
    }
    assert grader["args"] == {"do_sample": False, "enable_thinking": False}


def test_parity_config_matches_legacy_defaults():
    # Legacy FAMILY_TOKEN_DEFAULTS["bias"], EvaluationConfig and DatasetConfig
    # field defaults.
    legacy = LEGACY["settings"]
    config = yaml.safe_load(parity_config_path(PACKAGE).read_text())
    family = legacy["family_token_defaults"]
    grader = config["model_roles"]["grader"]
    assert config["generate_config"]["max_tokens"] == family["max_answer_tokens"]
    assert grader["config"]["max_tokens"] == family["max_judge_tokens"]
    assert grader["args"]["do_sample"] == family["sample_judge"]
    assert grader["args"]["enable_thinking"] == family["enable_thinking"]
    assert legacy["evaluation_sample_default"] is False
    assert config["eval_config"]["limit"] == legacy["evaluation_max_samples_default"]
    seed = legacy["dataset_seed_default"]
    assert config["generate_config"]["seed"] == grader["config"]["seed"] == seed


@pytest.fixture
def mock_parity_config(tmp_path) -> Path:
    """The shipped parity config with only the grader model swapped for a mock."""
    config = yaml.safe_load(parity_config_path(PACKAGE).read_text())
    config["model_roles"]["grader"]["model"] = "mockllm/model"
    config["model_roles"]["grader"]["args"] = {}
    path = tmp_path / "parity.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_parity_config_python_selects_legacy_samples(
    tmp_path, source, mock_parity_config
):
    """eval_config.limit picks the legacy seed-42 first-500 rows; settings apply."""
    source([dict(question=f"q{i}", answer="a") for i in range(502)])
    settings = read_parity_config(mock_parity_config)
    seen: dict[str, set[str]] = {"answer": set(), "grader": set()}

    def answer(messages, tools, choice, config):
        seen["answer"].add(json.dumps(generation_settings(config), sort_keys=True))
        return ModelOutput.from_content("mockllm/answer", "a")

    def judge(messages, tools, choice, config):
        seen["grader"].add(json.dumps(generation_settings(config), sort_keys=True))
        return ModelOutput.from_content("mockllm/grader", 'gt_agree: "yes"')

    settings["model_roles"]["grader"] = get_model(
        "mockllm/grader",
        custom_outputs=judge,
        config=settings["model_roles"]["grader"].config,
        memoize=False,
    )
    logs = inspect_eval(
        bbq(),
        model=get_model("mockllm/answer", custom_outputs=answer, memoize=False),
        log_dir=str(tmp_path),
        display="none",
        **settings,
    )
    assert logs[0].status == "success", logs[0].error
    assert logs[0].samples
    legacy = Dataset.from_list([{"i": i} for i in range(502)]).shuffle(seed=42)
    legacy_ids = list(legacy["i"])[:500]
    assert sorted(str(s.id) for s in logs[0].samples) == sorted(
        f"hirundo-io/bbq-gender-bias-free-text:{i}" for i in legacy_ids
    )
    assert [json.loads(s) for s in seen["answer"]] == [
        {"temperature": 0.0, "max_tokens": 128, "seed": 42, "extra_body": ANSWER_BODY}
    ]
    assert [json.loads(s) for s in seen["grader"]] == [
        {"temperature": 0.0, "max_tokens": 32, "seed": 42, "extra_body": GRADER_BODY}
    ]


def test_parity_config_cli(tmp_path, mock_parity_config):
    snapshot = tmp_path / "snapshot"
    Dataset.from_list(
        [dict(question=f"q{i}", answer="a", stereotyped_answer="s") for i in range(3)]
    ).save_to_disk(snapshot)
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
            "mockllm/model",
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
    spec = read_eval_log(logs[0]).eval
    generate = spec.model_generate_config
    assert (generate.temperature, generate.max_tokens, generate.seed) == (
        0.0,
        128,
        42,
    )
    assert spec.config.limit == 500
    assert spec.model_roles
    grader = spec.model_roles["grader"]
    assert not isinstance(grader, list)
    assert (grader.config.temperature, grader.config.max_tokens) == (0.0, 32)


# --- registry and reporting --------------------------------------------------


def test_registry_and_cli(tmp_path):
    from llm_behavior_eval import _registry

    assert "bbq" in _registry.__all__
    assert vars(_registry)["bbq"] is bbq
    listed = subprocess.run(
        [sys.executable, "-m", "inspect_ai", "list", "tasks", "llm_behavior_eval"],
        capture_output=True,
        text=True,
    )
    assert listed.returncode == 0, listed.stderr
    assert "bbq" in listed.stdout

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
            "-T",
            "bias_type=religion",
            "-T",
            "kind=unbias",
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


def test_target_metrics_match_eval_metrics(tmp_path, source):
    source([dict(question="q", answer="a", stereotyped_answer="s")])
    logs = inspect_eval(
        bbq(),
        model="mockllm/answer",
        model_roles={"grader": get_model("mockllm/grader")},
        log_dir=str(tmp_path),
        display="none",
    )
    names = set(metrics_of(logs[0]))
    assert names == {
        "accuracy",
        "error",
        "stereotype_bias",
        "empty_responses",
        "incomplete_response_rate",
    }
    targets = read_target_metrics_file(
        REPO / "llm_behavior_eval" / "bbq" / TARGET_METRICS_FILE
    )
    assert list(targets) == ["llm_behavior_eval/bbq"]
    assert (
        load_target_metrics()["llm_behavior_eval/bbq"]
        == targets["llm_behavior_eval/bbq"]
    )
    entry = {m["name"]: m for m in targets["llm_behavior_eval/bbq"]}
    assert set(entry) <= names
    assert entry["error"]["is_percentage"] and not entry["error"]["is_higher_better"]
    assert entry["accuracy"]["is_percentage"] and entry["accuracy"]["is_higher_better"]
    assert "stereotype_bias" in entry
    # inspect-brief finds every target in the written log.
    assert missing_target_metrics(logs[0].location) == []


def test_stereotype_bias_needs_a_stereotype_column(tmp_path, source):
    # Like the legacy engine, a split without `stereotyped_answer` reports no
    # stereotype_bias, so inspect-brief shows that one target as not found.
    source([dict(question="q", answer="a")])
    logs = inspect_eval(
        bbq(),
        model="mockllm/answer",
        model_roles={"grader": get_model("mockllm/grader")},
        log_dir=str(tmp_path),
        display="none",
    )
    assert missing_target_metrics(logs[0].location) == ["stereotype_bias"]


def test_registry_names_are_package_specific():
    # Inspect's registry keeps the last object registered under a name, so a
    # generic name reused by another benchmark would replace BBQ's when a log
    # is rescored; see _registry.check_unique_registry_names.
    from llm_behavior_eval import _registry

    names = {
        name
        for (_, name), modules in _registry.registry_definitions().items()
        if any(module.startswith("llm_behavior_eval.bbq.") for module in modules)
    }
    assert names == {
        "llm_behavior_eval/bbq",
        "llm_behavior_eval/bbq_metrics",
        "llm_behavior_eval/bbq_scorer",
        "llm_behavior_eval/bbq_user_message",
    }


def test_package_is_self_contained():
    """BBQ imports only its own modules, llm_behavior_eval.common and libraries."""
    import ast

    package = REPO / "llm_behavior_eval" / "bbq"
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                continue
            for module in modules:
                if module.startswith("llm_behavior_eval"):
                    assert module.startswith(
                        ("llm_behavior_eval.bbq", "llm_behavior_eval.common")
                    ), f"{path.name} imports {module}"
