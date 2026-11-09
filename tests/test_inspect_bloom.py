import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml
from datasets import Dataset
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalSpec, list_eval_logs, read_eval_log
from inspect_ai.model import GenerateConfig, ModelOutput, get_model

from llm_behavior_eval.bloom.scorer import parse_agreement, parse_uncertainty
from llm_behavior_eval.bloom.task import bloom
from llm_behavior_eval.common.parity import parity_config_path, read_parity_config
from llm_behavior_eval.common.target_metrics import (
    TARGET_METRICS_FILE,
    load_target_metrics,
    missing_target_metrics,
    read_target_metrics_file,
)

BLOOM = "llm_behavior_eval.bloom"
REPO = Path(__file__).parents[1]


def no_settings(config: GenerateConfig) -> None:
    """Neither role receives code-set inference settings."""
    assert config.temperature is None
    assert config.max_tokens is None
    assert config.top_p is None and config.top_k is None
    assert config.reasoning_effort is None and config.reasoning_tokens is None
    assert config.extra_body is None


@pytest.fixture
def source(monkeypatch):
    module = importlib.import_module("llm_behavior_eval.bloom.dataset")

    def install(rows):
        monkeypatch.setattr(
            module, "load_hf_dataset", lambda *a, **k: Dataset.from_list(rows)
        )

    return install


def run_cli(*args: str, cwd: Path | None = None) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "inspect_ai", *args, "--display", "none"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


# --- Registry and package ---------------------------------------------------


def test_registry_discovery():
    from llm_behavior_eval import _registry

    assert "bloom" in _registry.__all__
    assert vars(_registry)["bloom"] is bloom
    listed = subprocess.run(
        [sys.executable, "-m", "inspect_ai", "list", "tasks", "llm_behavior_eval"],
        capture_output=True,
        text=True,
    )
    assert listed.returncode == 0, listed.stderr
    assert "bloom" in listed.stdout


def test_package_is_self_contained():
    """BLOOM imports only common mechanics, never another benchmark or family."""
    package = REPO / "llm_behavior_eval" / "bloom"
    for path in package.glob("*.py"):
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and "llm_behavior_eval" in (
                stripped
            ):
                assert stripped.startswith(
                    ("from llm_behavior_eval.common", "from llm_behavior_eval.bloom")
                ), f"{path.name}: {stripped}"


def test_task_sets_no_inference_settings(source):
    source([dict(question="q", answer="a")])
    task = bloom()
    assert task.config == GenerateConfig()
    assert not task.model_roles


# --- Dataset ----------------------------------------------------------------


def test_full_source_in_legacy_shuffle_order(source):
    from llm_behavior_eval.bloom.dataset import BIAS_TYPES, dataset

    source([dict(question=f"q{i}", answer="a") for i in range(502)])
    legacy = Dataset.from_list([{"i": i} for i in range(502)]).shuffle(seed=42)
    for kind in ("bias", "unbias"):
        for bias_type in BIAS_TYPES:
            data = dataset(bias_type, kind)
            # Conversion keeps every row; Inspect's limit applies the legacy cap.
            assert len(data) == 502
            identity = f"hirundo-io/bloom-{bias_type}-{kind}-free-text"
            assert [s.id for s in data] == [f"{identity}:{i}" for i in legacy["i"]]
            assert [s.input for s in data[:3]] == [f"q{i}" for i in legacy["i"][:3]]
    full = dataset("age", "ambiguous")
    assert [s.id for s in full] != [s.id for s in dataset("age", "ambiguous", seed=43)]


def test_limit_selects_legacy_samples(tmp_path, source):
    """eval_config.limit=500 picks the rows legacy kept after its seeded shuffle."""
    source([dict(question=f"q{i}", answer="a") for i in range(502)])
    logs = inspect_eval(
        bloom(),
        model="mockllm/answer",
        model_roles={"grader": "mockllm/grader"},
        limit=500,
        log_dir=str(tmp_path),
        display="none",
    )
    assert logs[0].status == "success", logs[0].error
    assert logs[0].samples
    # Legacy prepare_dataloader: shuffle(seed=42), then select(range(500)).
    legacy = Dataset.from_list([{"i": i} for i in range(502)]).shuffle(seed=42)
    assert {str(s.id) for s in logs[0].samples} == {
        f"hirundo-io/bloom-gender-bias-free-text:{i}" for i in legacy["i"][:500]
    }


@pytest.mark.parametrize("column", ["question", "answer", "stereotyped_answer"])
def test_validation_covers_complete_source(source, column):
    rows: list[dict[str, str | None]] = [
        dict(question="q", answer="a", stereotyped_answer="s") for _ in range(501)
    ]
    rows[-1][column] = None
    source(rows)
    with pytest.raises(ValueError, match="required text"):
        bloom()


def test_offline_snapshot(tmp_path):
    from llm_behavior_eval.bloom.dataset import dataset

    Dataset.from_list(
        [dict(question="q", answer="a", stereotyped_answer="s")]
    ).save_to_disk(tmp_path / "data")
    data = dataset("race", "unbias", dataset_path=str(tmp_path / "data"))
    assert (data[0].metadata or {})["has_stereotype"]
    assert data[0].id == "hirundo-io/bloom-race-unbias-free-text:0"


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
    from llm_behavior_eval.evaluate import _behavior_presets
    from llm_behavior_eval.evaluation_utils.enums import BLOOM_BIAS_TYPES

    source([dict(question="q", answer="a")])
    assert set(BIAS_TYPES) == BLOOM_BIAS_TYPES
    assert [dataset(t, kind).name for t in BIAS_TYPES] == _behavior_presets(
        f" BLOOM : {kind.upper()} : ALL "
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


# --- Prompts, parsing and classification -------------------------------------


def test_exact_legacy_prompts():
    from llm_behavior_eval.bloom import prompts
    from llm_behavior_eval.evaluation_utils.free_text_bias_evaluator import (
        FreeTextBiasEvaluator,
    )
    from llm_behavior_eval.evaluation_utils.prompts import (
        FREE_TEXT_SYSTEM_MESSAGE,
        UNKNOWN_ANSWERS,
    )

    for key in ("AGREEMENT_CHECK_SYSTEM_PROMPT", "UNCERTAINTY_CHECK_SYSTEM_PROMPT"):
        assert getattr(prompts, key) == getattr(FreeTextBiasEvaluator, key)
    assert prompts.SYSTEM_MESSAGE == FREE_TEXT_SYSTEM_MESSAGE["content"]
    assert list(prompts.UNKNOWN_ANSWERS) == UNKNOWN_ANSWERS


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
def test_agreement_parser_parity(raw):
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
def test_uncertainty_parser_parity(raw):
    from llm_behavior_eval.evaluation_utils.free_text_bias_evaluator import (
        map_uncertainty_outputs,
    )

    assert (
        parse_uncertainty(raw)
        == map_uncertainty_outputs([[{"generated_text": raw}]])[0]
    )


def test_classification_and_zero_stereotypes():
    from inspect_ai.scorer import SampleScore, Score

    from llm_behavior_eval.bloom.metrics import bloom_metrics
    from llm_behavior_eval.bloom.scorer import classify

    flags = classify("Unknown", "unmatched", "yes", "yes", True)
    assert flags == dict(correct=1, empty=1, unknown=0, stereotyped=0)
    for kind in ("bias", "unbias"):
        metrics = cast("Any", bloom_metrics(kind, True))(
            [SampleScore(score=Score(value=0, metadata=flags))]
        )
        assert metrics["stereotype_bias"] == 0
    assert classify("person", "no", "yes", "yes", True)["unknown"] == 1
    assert classify("person", "no", "yes", "unparseable", True)["stereotyped"] == 1


# --- Native scoring ----------------------------------------------------------


@pytest.mark.parametrize("kind", ["bias", "unbias", "ambiguous"])
@pytest.mark.parametrize("has_stereotype", [True, False])
def test_legacy_grade_parity(tmp_path, monkeypatch, source, kind, has_stereotype):
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
        no_settings(config)
        question = messages[-1].text
        assert question.endswith("\n")
        i = int(question.strip()[1:])
        out = ModelOutput.from_content("mockllm/answer", f"a{i}")
        if i == 4:
            out.choices[0].stop_reason = "max_tokens"
        return out

    def judge(messages, tools, choice, config):
        no_settings(config)
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
        item = sample.scores["bloom_scorer"]
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


def test_missing_grader(tmp_path, source):
    source([dict(question="q", answer="a")])
    logs = inspect_eval(
        bloom(), model="mockllm/model", display="none", log_dir=str(tmp_path)
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
            # The scorer passes no GenerateConfig to the grader.
            assert not args and kwargs.get("config") is None
            text = "gt_agree: yes" if len(calls) == 1 else "candidate_uncertain: no"
            return ModelOutput.from_content("judge", text)
        return await original(self, input, *args, **kwargs)

    monkeypatch.setattr(Model, "generate", generate)
    logs = inspect_eval(
        bloom(),
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
    assert logs[0].samples[0].scores["bloom_scorer"].value == 1


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
    score = logs[0].samples[0].scores["bloom_scorer"]
    assert score.metadata
    assert score.value == 0 and score.metadata["empty"] == 1


def test_cli_registration(tmp_path):
    snapshot = tmp_path / "snapshot"
    Dataset.from_list([dict(question="q", answer="a")]).save_to_disk(snapshot)
    run_cli(
        "eval",
        "llm_behavior_eval/bloom",
        "-T",
        f"dataset_path={snapshot}",
        "--model",
        "mockllm/answer",
        "--model-role",
        "grader=mockllm/grader",
        "--log-dir",
        str(tmp_path / "logs"),
        cwd=tmp_path,
    )


# --- Target metrics ------------------------------------------------------------


def test_target_metrics_match_eval_metric_names(tmp_path):
    """Every declared metric is emitted under that name in the .eval log."""
    snapshot = tmp_path / "snapshot"
    Dataset.from_list(
        [dict(question="q", answer="a", stereotyped_answer="s")]
    ).save_to_disk(snapshot)
    run_cli(
        "eval",
        "llm_behavior_eval/bloom",
        "-T",
        f"dataset_path={snapshot}",
        "--model",
        "mockllm/answer",
        "--model-role",
        "grader=mockllm/grader",
        "--log-dir",
        str(tmp_path / "logs"),
        cwd=tmp_path,
    )
    logs = list_eval_logs(str(tmp_path / "logs"))
    assert len(logs) == 1
    log = read_eval_log(logs[0])
    # inspect-brief finds every target in the written log.
    assert missing_target_metrics(log) == []
    results = log.results
    assert results
    emitted = {name for score in results.scores for name in score.metrics}
    targets = read_target_metrics_file(
        REPO / "llm_behavior_eval" / "bloom" / TARGET_METRICS_FILE
    )
    assert list(targets) == ["llm_behavior_eval/bloom"]
    declared = targets["llm_behavior_eval/bloom"]
    assert load_target_metrics()["llm_behavior_eval/bloom"] == declared
    assert {m["name"] for m in declared} <= emitted
    for entry in declared:
        assert set(entry) == {
            "name",
            "is_percentage",
            "is_higher_better",
            "is_normalized",
        }


def test_stereotype_bias_needs_a_stereotype_column(tmp_path, source):
    # Like the legacy engine, a split without `stereotyped_answer` reports no
    # stereotype_bias, so inspect-brief shows that one target as not found.
    source([dict(question="q", answer="a")])
    logs = inspect_eval(
        bloom(),
        model="mockllm/answer",
        model_roles={"grader": "mockllm/grader"},
        log_dir=str(tmp_path),
        display="none",
    )
    assert missing_target_metrics(logs[0].location) == ["stereotype_bias"]


def test_registry_names_are_package_specific():
    # Inspect's registry keeps the last object registered under a name, so a
    # generic name reused by another benchmark would replace BLOOM's when a log
    # is rescored; see _registry.check_unique_registry_names.
    from llm_behavior_eval import _registry

    names = {
        name
        for (_, name), modules in _registry.registry_definitions().items()
        if any(module.startswith(f"{BLOOM}.") for module in modules)
    }
    assert names == {
        "llm_behavior_eval/bloom",
        "llm_behavior_eval/bloom_generate",
        "llm_behavior_eval/bloom_metrics",
        "llm_behavior_eval/bloom_scorer",
    }


# --- Parity config -------------------------------------------------------------


def test_shipped_parity_config_holds_legacy_settings():
    from llm_behavior_eval.evaluation_utils.eval_config import (
        FAMILY_TOKEN_DEFAULTS,
        EvaluationConfig,
    )

    path = parity_config_path(BLOOM)
    assert path.parent == REPO / "llm_behavior_eval" / "bloom"
    config = yaml.safe_load(path.read_text())
    assert set(config) == {"generate_config", "eval_config", "model_roles"}
    legacy = FAMILY_TOKEN_DEFAULTS["bias"]
    no_thinking = {"chat_template_kwargs": {"enable_thinking": False}}
    assert legacy["enable_thinking"] is False and legacy["sample_judge"] is False
    answer = GenerateConfig.model_validate(config["generate_config"])
    assert (answer.temperature, answer.max_tokens) == (
        0.0,
        legacy["max_answer_tokens"],
    )
    assert answer.extra_body == no_thinking
    assert config["eval_config"] == {
        "limit": EvaluationConfig.model_fields["max_samples"].default
    }
    grader = config["model_roles"]["grader"]
    assert grader["model"] == "hf/google/gemma-3-12b-it"
    judge = GenerateConfig.model_validate(grader["config"])
    assert (judge.temperature, judge.max_tokens) == (0.0, legacy["max_judge_tokens"])
    assert judge.extra_body == no_thinking
    assert grader["args"] == {"do_sample": False, "enable_thinking": False}


@pytest.fixture
def mock_parity_config(tmp_path) -> Path:
    """The shipped parity config with only the grader model swapped for a mock."""
    config = yaml.safe_load(parity_config_path(BLOOM).read_text())
    config["model_roles"]["grader"]["model"] = "mockllm/model"
    config["model_roles"]["grader"]["args"] = {}
    path = tmp_path / "parity.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def assert_parity_recorded(spec: EvalSpec) -> None:
    generate = spec.model_generate_config
    assert (generate.temperature, generate.max_tokens) == (0.0, 128)
    assert spec.config.limit == 500
    assert spec.model_roles
    grader = spec.model_roles["grader"]
    assert not isinstance(grader, list)
    assert (grader.config.temperature, grader.config.max_tokens) == (0.0, 32)


def test_parity_config_cli(tmp_path, mock_parity_config):
    snapshot = tmp_path / "snapshot"
    Dataset.from_list([dict(question="q", answer="a")]).save_to_disk(snapshot)
    run_cli(
        "eval",
        "llm_behavior_eval/bloom",
        "-T",
        f"dataset_path={snapshot}",
        "--model",
        "mockllm/model",
        "--run-config",
        str(mock_parity_config),
        "--log-dir",
        str(tmp_path / "logs"),
    )
    logs = list_eval_logs(str(tmp_path / "logs"))
    assert len(logs) == 1
    assert_parity_recorded(read_eval_log(logs[0]).eval)


def test_parity_config_python(tmp_path, source, mock_parity_config):
    source([dict(question="q", answer="a")])
    grader_configs: list[GenerateConfig] = []

    def judge(messages, tools, choice, config):
        grader_configs.append(config)
        return ModelOutput.from_content("mockllm/model", "gt_agree: yes")

    kwargs = read_parity_config(mock_parity_config)
    kwargs["model_roles"] = {
        "grader": get_model(
            "mockllm/model",
            config=kwargs["model_roles"]["grader"].config,
            custom_outputs=judge,
            memoize=False,
        )
    }
    logs = inspect_eval(
        bloom(),
        model="mockllm/model",
        log_dir=str(tmp_path),
        display="none",
        **kwargs,
    )
    assert logs[0].status == "success", logs[0].error
    assert_parity_recorded(logs[0].eval)
    # The grader receives only its role's settings, not the answer budget.
    assert {(c.temperature, c.max_tokens) for c in grader_configs} == {(0.0, 32)}
