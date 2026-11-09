import json
import math
import subprocess
import sys
from collections.abc import Iterator
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, cast

import pytest
from datasets import Dataset, DatasetDict
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessage,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    ModelName,
    ModelOutput,
    StopReason,
    get_model,
)
from inspect_ai.scorer import SampleScore, Score, accuracy, mean, stderr
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_ai.util import registry_info

from llm_behavior_eval.common import judge as judge_module
from llm_behavior_eval.common.dataset import (
    LEGACY_SHUFFLE_SEED,
    legacy_order,
    load_hf_dataset,
    to_inspect_dataset,
)
from llm_behavior_eval.common.judge import (
    StopReasonUnavailableError,
    judge_scorer,
    require_stop_reason,
    strip_thinking,
)
from llm_behavior_eval.common.metrics import (
    field_rate,
    incomplete_response_rate,
    reduced_metrics,
    scores_by_sample,
)
from llm_behavior_eval.inspect_template.dataset import record_to_sample
from llm_behavior_eval.inspect_template.task import inspect_template

from inspect_helpers import (
    LocalOpenAIJudge,
    assert_grader_required,
    eval_metrics,
    generation_settings,
    message_text,
    only_log,
    rate,
    run_inspect_cli,
    sample_scores,
)


class _Loaded:
    def __init__(self, kind: str, source: str, kwargs: dict[str, Any]) -> None:
        self.kind = kind
        self.source = source
        self.kwargs = kwargs
        self.eos_token = "</s>"


@pytest.fixture
def hf_stub(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Any, list[_Loaded]]]:
    # Configure the real HF provider with stub loaders: no weights, no GPU.
    hf = pytest.importorskip("inspect_ai.model._providers.hf")
    calls: list[_Loaded] = []

    def loader(kind: str) -> type:
        def from_pretrained(source: str, **kwargs: Any) -> _Loaded:
            loaded = _Loaded(kind, source, kwargs)
            calls.append(loaded)
            return loaded

        return type(kind, (), {"from_pretrained": staticmethod(from_pretrained)})

    monkeypatch.setattr(hf, "AutoModelForCausalLM", loader("model"))
    monkeypatch.setattr(hf, "AutoTokenizer", loader("tokenizer"))
    yield hf, calls


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("plain", "plain"),
        ("<think>x</think> 2 ", "2"),
        ("x</think>a</think>b", "b"),
        ("<think>unfinished", "<think>unfinished"),
    ],
)
def test_thinking(answer: str, expected: str) -> None:
    assert strip_thinking(answer) == expected
    assert strip_thinking(answer, exclude_thinking_trace_for_judge=False) == answer
    assert strip_thinking(answer, thinking_start_token=None) == answer


def test_rates() -> None:
    scores = [
        SampleScore(
            score=Score(value=1, metadata={"group": "safe", "stop_reason": "stop"})
        ),
        SampleScore(
            score=Score(
                value=0,
                metadata={
                    "group": "unsafe",
                    "stop_reason": "max_tokens",
                    "skipped": True,
                },
            )
        ),
        SampleScore(score=Score(value=0, metadata={"parse_error": "invalid"})),
    ]
    assert rate(field_rate())(scores) == 1
    assert rate(field_rate(metadata_key="group", metadata_value="safe"))(scores) == 1
    assert math.isnan(
        rate(field_rate(metadata_key="group", metadata_value="missing"))(scores)
    )
    # The third score records no stop reason, so it cannot be classified.
    assert rate(incomplete_response_rate())(scores) == pytest.approx(1 / 2)
    assert math.isnan(rate(incomplete_response_rate())([]))
    assert rate(field_rate("a"))([SampleScore(score=Score(value={"a": 0.5}))]) == 0.5


def stopped(reason: str | None) -> SampleScore:
    return SampleScore(score=Score(value=1, metadata={"stop_reason": reason}))


def test_incomplete_rate_counts_context_limit_and_skips_unknown() -> None:
    known = [stopped("stop"), stopped("max_tokens"), stopped("model_length")]
    assert rate(incomplete_response_rate())(known) == pytest.approx(2 / 3)
    # Inspect's hf/ provider reports "unknown" for every answer. Those answers
    # leave the denominator instead of counting as complete, and a run with no
    # known stop reason reports NaN, not 0.
    unknown = [stopped("unknown"), stopped(None)]
    assert rate(incomplete_response_rate())([*known, *unknown]) == pytest.approx(2 / 3)
    assert math.isnan(rate(incomplete_response_rate())(unknown))


def test_scores_by_sample() -> None:
    def observation(sample_id: str | None, value: int) -> SampleScore:
        return SampleScore(score=Score(value=value), sample_id=sample_id)

    scores = [
        observation("b", 1),
        observation("a", 0),
        observation(None, 1),
        observation("b", 0),
        observation(None, 0),
        observation("a", 1),
    ]
    groups = scores_by_sample(scores)
    assert [[s.score.value for s in group] for group in groups] == [
        [1, 0],
        [0, 1],
        [1],
        [0],
    ]
    assert [group[0].sample_id for group in groups] == ["b", "a", None, None]
    assert scores_by_sample([]) == []


def test_reduced_metrics() -> None:
    # The shared metrics, and so the template, see every epoch.
    assert reduced_metrics(inspect_template()) == []
    dataset = [Sample(input="question", target="answer")]
    scorer = judge_scorer("{answer}", r"SCORE:\s*(\S+)")
    assert reduced_metrics(Task(dataset=dataset, scorer=scorer)) == []
    # Task-level metrics replace the scorer's; Inspect built-ins are reduced.
    mixed = Task(
        dataset=dataset,
        scorer=scorer,
        metrics=[field_rate(), accuracy(), {"group": [mean(), stderr()]}],
    )
    assert reduced_metrics(mixed) == [
        "inspect_ai/accuracy",
        "inspect_ai/mean",
        "inspect_ai/stderr",
    ]


def test_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ds = Dataset.from_list(
        [{"id": str(i), "question": "q", "answer": "a"} for i in range(502)]
    )
    DatasetDict({"test": ds}).save_to_disk(tmp_path / "snapshot")
    loaded = load_hf_dataset("logical/id", dataset_path=str(tmp_path / "snapshot"))
    assert len(loaded) == 502
    # Conversion keeps every record in order; sample limits are Inspect's job.
    converted = to_inspect_dataset(loaded, record_to_sample, dataset_id="logical/id")
    assert len(converted) == 502
    assert converted.name == "logical/id"
    assert [s.id for s in converted][:2] == ["0", "1"]
    import datasets

    calls = []

    def load(source: str, **kwargs: Any) -> DatasetDict:
        calls.append((source, kwargs))
        # datasets types select() loosely; selecting rows of a Dataset is one.
        return DatasetDict({"train": ds, "test": cast("Dataset", ds.select([0]))})

    monkeypatch.setattr(datasets, "load_dataset", load)
    assert len(load_hf_dataset("logical", dataset_path="override")) == 502
    # No token is passed: datasets reads HF_TOKEN itself.
    assert calls == [("override", {})]
    load_hf_dataset("logical", revision="abc123")
    assert calls[-1] == ("logical", {"revision": "abc123"})
    monkeypatch.setattr(
        datasets, "load_dataset", lambda *a, **kw: DatasetDict({"a": ds, "b": ds})
    )
    with pytest.raises(ValueError, match="exactly one"):
        load_hf_dataset("logical")


@pytest.mark.parametrize("num_rows", [0, 1, 7, 50])
def test_legacy_order_matches_legacy_shuffle(num_rows: int) -> None:
    rows = Dataset.from_dict({"index": list(range(num_rows))})
    expected = list(rows.shuffle(seed=LEGACY_SHUFFLE_SEED)["index"]) if num_rows else []
    order = legacy_order(num_rows)
    assert order == expected
    assert sorted(order) == list(range(num_rows))
    if num_rows > 2:
        assert order != list(range(num_rows))
        assert legacy_order(num_rows, seed=7) != order
    if num_rows == 7:
        # Pinned: a datasets upgrade that changes shuffle(seed=42) breaks parity.
        assert order == [3, 2, 6, 4, 1, 5, 0]


@pytest.mark.parametrize(
    ("stop", "judge_text", "expected"),
    [
        ("stop", "SCORE: 1", 1),
        ("stop", "invalid", 0),
        ("max_tokens", "SCORE: 1", 0),
        ("stop", "SCORE: 0", 0),
    ],
)
def test_judge(
    tmp_path: Path, stop: StopReason, judge_text: str, expected: int
) -> None:
    answer = ModelOutput.from_content("mockllm/model", "<think>secret</think>2")
    answer.choices[0].stop_reason = stop
    requests = []

    def grade(
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        requests.append(messages)
        assert "secret" not in str(messages)
        return ModelOutput.from_content("mockllm/judge", judge_text)

    logs = inspect_eval(
        inspect_template(),
        model=get_model("mockllm/model", custom_outputs=[answer]),
        model_roles={"grader": get_model("mockllm/judge", custom_outputs=grade)},
        log_dir=str(tmp_path),
        display="none",
    )
    assert logs[0].status == "success"
    assert logs[0].samples
    assert logs[0].samples[0].scores
    score = logs[0].samples[0].scores["judge_scorer"]
    assert score.metadata
    assert score.value == expected
    assert bool(requests) == (stop == "stop")
    assert score.metadata["raw_judge_text"] == (judge_text if requests else "")
    assert bool(score.metadata.get("parse_error")) == (judge_text == "invalid")


@pytest.mark.parametrize("judge_text", ["SCORE: nan", "SCORE: inf", "SCORE: -inf"])
def test_judge_rejects_non_finite_scores(tmp_path: Path, judge_text: str) -> None:
    logs = inspect_eval(
        Task(
            dataset=[Sample(input="question", target="answer")],
            scorer=judge_scorer("{answer}", r"SCORE:\s*(\S+)"),
        ),
        model=get_model("mockllm/model"),
        model_roles={
            "grader": get_model(
                "mockllm/judge",
                custom_outputs=[ModelOutput.from_content("mockllm/judge", judge_text)],
            )
        },
        log_dir=str(tmp_path),
        display="none",
    )
    assert logs[0].status == "success"
    assert logs[0].samples
    assert logs[0].samples[0].scores
    score = logs[0].samples[0].scores["judge_scorer"]
    assert score.value == 0
    assert score.metadata
    assert "not finite" in score.metadata["parse_error"]
    assert logs[0].results
    metrics = logs[0].results.scores[0].metrics
    assert math.isnan(metrics["field_rate"].value)


def answer_output(stop: StopReason, text: str = "2") -> ModelOutput:
    output = ModelOutput.from_content("mockllm/model", text)
    output.choices[0].stop_reason = stop
    return output


def judge_output(text: str) -> ModelOutput:
    return ModelOutput.from_content("mockllm/judge", text)


@pytest.mark.parametrize(
    "grader_reply",
    [
        judge_output("<think>A wrong answer would get SCORE: 0.</think>\nSCORE: 1"),
        ModelOutput.from_content(
            "mockllm/judge",
            [
                ContentReasoning(reasoning="A wrong answer would get SCORE: 0."),
                ContentText(text="SCORE: 1"),
            ],
        ),
    ],
    ids=["inline", "structured"],
)
def test_verdict_ignores_grader_reasoning(
    tmp_path: Path, grader_reply: ModelOutput
) -> None:
    logs = inspect_eval(
        inspect_template(),
        model=get_model("mockllm/model", custom_outputs=[answer_output("stop")]),
        model_roles={
            "grader": get_model("mockllm/judge", custom_outputs=[grader_reply])
        },
        log_dir=str(tmp_path),
        display="none",
    )
    assert logs[0].status == "success", logs[0].error
    assert logs[0].samples
    assert logs[0].samples[0].scores
    score = logs[0].samples[0].scores["judge_scorer"]
    assert score.value == 1
    assert score.metadata
    # The grader's full reply stays available for diagnostics.
    assert score.metadata["raw_judge_text"] == grader_reply.completion


@pytest.mark.parametrize("epochs", [1, 2])
def test_metrics_aggregate_every_epoch(tmp_path: Path, epochs: int) -> None:
    # Inspect's epoch reducers keep only the first epoch's metadata. Here
    # epoch 1 is truncated (skipped, incomplete); epoch 2 finishes, graded 1.
    answers = [answer_output("max_tokens"), answer_output("stop")]
    logs = inspect_eval(
        inspect_template(),
        model=get_model("mockllm/model", custom_outputs=answers[-epochs:]),
        model_roles={
            "grader": get_model(
                "mockllm/judge", custom_outputs=[judge_output("SCORE: 1")]
            )
        },
        epochs=epochs,
        log_dir=str(tmp_path),
        display="none",
    )
    expected = {1: 0.0, 2: 0.5}[epochs]
    assert eval_metrics(logs[0]) == {
        "field_rate": 1.0,
        "incomplete_response_rate": expected,
    }
    assert logs[0].samples
    assert len(logs[0].samples) == epochs


@solver
def canned_answer(stop: StopReason) -> Solver:
    """Set the answer without calling the model under test."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.output = answer_output(stop)
        return state

    return solve


def only_stop_task(only_stop: bool, stop: StopReason) -> Task:
    return Task(
        dataset=[Sample(input="question", target="answer")],
        solver=canned_answer(stop),
        scorer=judge_scorer("{answer}", r"SCORE:\s*(\S+)", only_stop=only_stop),
    )


def grader_roles() -> dict[str, Any]:
    return {
        "grader": get_model("mockllm/judge", custom_outputs=[judge_output("SCORE: 1")])
    }


@pytest.mark.parametrize("only_stop", [True, False])
def test_only_stop_fails_on_hf_models_under_test(
    tmp_path: Path, hf_stub: tuple[Any, list[_Loaded]], only_stop: bool
) -> None:
    # The real HF provider (stub weights): it reports stop_reason "unknown".
    hf_model = get_model("hf/org/repo", memoize=False, device="cpu")
    logs = inspect_eval(
        only_stop_task(only_stop, "unknown"),
        model=hf_model,
        model_roles=grader_roles(),
        log_dir=str(tmp_path),
        display="none",
    )
    if only_stop:
        assert logs[0].status == "error"
        assert logs[0].error
        assert "StopReasonUnavailableError" in logs[0].error.traceback
        assert "hf/org/repo" in logs[0].error.message
        assert "vllm/" in logs[0].error.message
        return
    # Without only_stop every answer is judged; its stop reason is unknown.
    assert eval_metrics(logs[0])["field_rate"] == 1
    assert math.isnan(eval_metrics(logs[0])["incomplete_response_rate"])


def test_require_stop_reason_is_provider_based() -> None:
    def state(model: str) -> TaskState:
        return TaskState(
            model=ModelName(model), sample_id=1, epoch=1, input="q", messages=[]
        )

    for name in ("hf/org/repo", "hf/local"):
        with pytest.raises(StopReasonUnavailableError, match="hf/ provider"):
            require_stop_reason(state(name))
    for name in ("vllm/org/repo", "openai-api/airgap/judge", "mockllm/model"):
        require_stop_reason(state(name))
    assert judge_module.NO_STOP_REASON_PROVIDERS == {"hf"}


def test_only_stop_skips_unknown_stop_reasons_elsewhere(tmp_path: Path) -> None:
    # A provider that reports stop reasons but not for this answer: skipped,
    # and left out of the incomplete-response rate.
    logs = inspect_eval(
        only_stop_task(True, "unknown"),
        model="mockllm/model",
        model_roles=grader_roles(),
        log_dir=str(tmp_path),
        display="none",
    )
    metrics = eval_metrics(logs[0])
    assert math.isnan(metrics["field_rate"])
    assert math.isnan(metrics["incomplete_response_rate"])


@pytest.mark.parametrize("exclude", [True, False])
def test_template_thinking_task_args(tmp_path: Path, exclude: bool) -> None:
    graded: list[str] = []

    def grade(
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        graded.append(messages[-1].text)
        return judge_output("SCORE: 1")

    answer = answer_output("stop", "<reason>secret</reason>2")
    logs = inspect_eval(
        inspect_template(
            exclude_thinking_trace_for_judge=exclude,
            thinking_start_token="<reason>",
            thinking_end_token="</reason>",
        ),
        model=get_model("mockllm/model", custom_outputs=[answer]),
        model_roles={"grader": get_model("mockllm/judge", custom_outputs=grade)},
        log_dir=str(tmp_path),
        display="none",
    )
    assert eval_metrics(logs[0])["field_rate"] == 1
    [prompt] = graded
    assert ("secret" in prompt) is not exclude
    assert "Answer: 2\n" in prompt or not exclude


def test_cli_turns_thinking_stripping_off(tmp_path: Path) -> None:
    run_inspect_cli(
        "eval",
        "llm_behavior_eval/inspect_template",
        "--model",
        "mockllm/model",
        "-T",
        "exclude_thinking_trace_for_judge=false",
        "--log-dir",
        str(tmp_path / "logs"),
        cwd=tmp_path,
    )
    log = only_log(tmp_path / "logs")
    assert log.eval.task_args["exclude_thinking_trace_for_judge"] is False


def test_grader_role_is_required(tmp_path: Path) -> None:
    assert_grader_required(
        Task(
            dataset=[Sample(input="question", target="answer")],
            scorer=judge_scorer("{answer}", r"SCORE:\s*(\S+)"),
        ),
        tmp_path,
    )


@pytest.mark.parametrize("user_settings", [False, True])
def test_settings_are_user_supplied_per_role(
    tmp_path: Path, user_settings: bool
) -> None:
    seen: dict[str, GenerateConfig] = {}

    def answer(
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        seen["answer"] = config
        return ModelOutput.from_content("mockllm/model", "2")

    def grade(
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        seen["grader"] = config
        return ModelOutput.from_content("mockllm/judge", "SCORE: 1")

    grader_config = (
        GenerateConfig(temperature=0.0, max_tokens=7)
        if user_settings
        else GenerateConfig()
    )
    eval_settings: dict[str, Any] = (
        {"temperature": 0.3, "max_tokens": 99} if user_settings else {}
    )
    logs = inspect_eval(
        inspect_template(),
        model=get_model("mockllm/model", custom_outputs=answer, memoize=False),
        model_roles={
            "grader": get_model(
                "mockllm/judge",
                custom_outputs=grade,
                config=grader_config,
                memoize=False,
            )
        },
        log_dir=str(tmp_path),
        display="none",
        **eval_settings,
    )
    assert logs[0].status == "success"
    if user_settings:
        # Eval-level settings reach only the model under test; the grader
        # gets only its role config.
        assert generation_settings(seen["answer"]) == eval_settings
        assert generation_settings(seen["grader"]) == {
            "temperature": 0.0,
            "max_tokens": 7,
        }
    else:
        # Benchmark code sets nothing, so both fall through to provider defaults.
        assert generation_settings(seen["answer"]) == {}
        assert generation_settings(seen["grader"]) == {}
        assert generation_settings(logs[0].eval.model_generate_config) == {}


@pytest.mark.parametrize("source", ["flags", "generate_config_file"])
def test_cli_per_role_settings(tmp_path: Path, source: str) -> None:
    if source == "flags":
        answer_args = ["--temperature", "0.3", "--max-tokens", "99"]
    else:
        config_file = tmp_path / "answer.yaml"
        config_file.write_text("temperature: 0.3\nmax_tokens: 99\n")
        answer_args = ["--generate-config", str(config_file)]
    run_inspect_cli(
        "eval",
        "llm_behavior_eval/inspect_template",
        "--model",
        "mockllm/model",
        *answer_args,
        "--model-role",
        "grader={model: mockllm/model, temperature: 0.0, max_tokens: 7}",
        "--log-dir",
        str(tmp_path / "logs"),
    )
    spec = only_log(tmp_path / "logs").eval
    assert generation_settings(spec.model_generate_config) == {
        "temperature": 0.3,
        "max_tokens": 99,
    }
    assert spec.model_roles
    grader = spec.model_roles["grader"]
    assert not isinstance(grader, list)
    assert generation_settings(grader.config) == {"temperature": 0.0, "max_tokens": 7}


def test_openai_compatible_grader_defaults(
    tmp_path: Path, openai_judge: LocalOpenAIJudge
) -> None:
    base_url, requests = openai_judge.base_url, openai_judge.requests
    grader = get_model(
        "openai-api/airgap/judge",
        base_url=base_url,
        api_key="local-test-placeholder",
    )
    logs = inspect_eval(
        inspect_template(),
        model="mockllm/model",
        model_roles={"grader": grader},
        log_dir=str(tmp_path),
        display="none",
    )
    assert logs[0].status == "success"
    assert logs[0].samples
    assert logs[0].samples[0].scores
    # The endpoint's result overrides the template's mock grader.
    assert logs[0].samples[0].scores["judge_scorer"].value == 0
    assert len(requests) == 1
    assert openai_judge.paths == ["/v1/chat/completions"]
    body = requests[0]
    assert body["model"] == "judge"
    # Nothing set, so the endpoint's own generation defaults apply.
    for key in ("temperature", "top_p", "max_tokens", "max_completion_tokens"):
        assert key not in body


def test_openai_judge_reply_can_depend_on_the_request(
    tmp_path: Path, openai_judge: LocalOpenAIJudge
) -> None:
    openai_judge.reply = lambda body: f"SCORE: {len(body['messages'])}"
    grader = get_model(
        "openai-api/airgap/judge",
        base_url=openai_judge.base_url,
        api_key="local-test-placeholder",
        memoize=False,
    )
    [log] = inspect_eval(
        inspect_template(),
        model="mockllm/model",
        model_roles={"grader": grader},
        log_dir=str(tmp_path),
        display="none",
    )
    [request] = openai_judge.requests
    assert [message_text(m) for m in request["messages"]]
    [score] = sample_scores(log, "judge_scorer")
    assert score.value == len(request["messages"])


def test_openai_compatible_grader_cli_role_config(
    tmp_path: Path, openai_judge: LocalOpenAIJudge
) -> None:
    base_url, requests = openai_judge.base_url, openai_judge.requests
    role = json.dumps(
        {
            "model": "openai-api/airgap/judge",
            "model_args": {"base_url": base_url, "api_key": "local-test"},
            "temperature": 0.0,
            "max_tokens": 32,
        }
    )
    run_inspect_cli(
        "eval",
        "llm_behavior_eval/inspect_template",
        "--model",
        "mockllm/model",
        "--temperature",
        "0.9",
        "--model-role",
        f"grader={role}",
        "--log-dir",
        str(tmp_path / "logs"),
    )
    assert len(requests) == 1
    body = requests[0]
    assert body["temperature"] == 0
    assert body["max_tokens"] == 32
    log = only_log(tmp_path / "logs")
    assert log.samples
    assert log.samples[0].scores
    assert log.samples[0].scores["judge_scorer"].value == 0


def test_registration_and_lazy_import(tmp_path: Path) -> None:
    assert any(e.name == "llm_behavior_eval" for e in entry_points(group="inspect_ai"))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'import llm_behavior_eval._registry; import sys; assert "torch" not in sys.modules; assert "transformers" not in sys.modules',
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    run_inspect_cli(
        "eval",
        "llm_behavior_eval/inspect_template",
        "--model",
        "mockllm/model",
        "--log-dir",
        str(tmp_path / "local"),
    )
    # Force package entry-point resolution rather than the same-named local directory.
    run_inspect_cli(
        "eval",
        "llm_behavior_eval/inspect_template",
        "--model",
        "mockllm/model",
        "--log-dir",
        str(tmp_path / "registered"),
        cwd=tmp_path,
    )
    listed = subprocess.run(
        [sys.executable, "-m", "inspect_ai", "list", "tasks", "llm_behavior_eval"],
        capture_output=True,
        text=True,
    )
    assert listed.returncode == 0, listed.stderr
    assert "inspect_template" in listed.stdout


def test_in_process_registry_names_are_namespaced() -> None:
    # Inspect adds the package namespace only when the package's metadata shows
    # an install. If the source-tree egg-info comes before site-packages on
    # sys.path, as in pytest's prepend import mode, it hides the editable
    # install's PEP 610 metadata, and in-process task and metric names lose
    # the prefix that target_metrics.json and inspect-brief key on.
    assert registry_info(inspect_template).name == "llm_behavior_eval/inspect_template"
    assert (
        registry_info(incomplete_response_rate()).name
        == "llm_behavior_eval/incomplete_response_rate"
    )


@pytest.mark.parametrize(
    ("name", "model_args", "model_source", "tokenizer_source"),
    [
        ("hf/org/repo", {}, "org/repo", "org/repo"),
        (
            "hf/local",
            {"model_path": "/models/answer", "tokenizer_path": "/models/tok"},
            "/models/answer",
            "/models/tok",
        ),
        ("hf/local", {"model_path": "/models/answer"}, "/models/answer", None),
    ],
)
def test_hf_provider_configuration(
    hf_stub: tuple[Any, list[_Loaded]],
    name: str,
    model_args: dict[str, Any],
    model_source: str,
    tokenizer_source: str | None,
) -> None:
    hf, calls = hf_stub
    model = get_model(
        name,
        memoize=False,
        device="cpu",
        batch_size=4,
        trust_remote_code=True,
        **model_args,
    )
    api = model.api
    assert isinstance(api, hf.HuggingFaceAPI)
    assert [(c.kind, c.source) for c in calls] == [
        ("model", model_source),
        ("tokenizer", tokenizer_source or model_source),
    ]
    assert calls[0].kwargs["device_map"] == "cpu"
    assert calls[0].kwargs["trust_remote_code"] is True
    assert "model_path" not in calls[0].kwargs
    assert api.batch_size == 4


def test_hf_provider_documented_defaults(hf_stub: tuple[Any, list[_Loaded]]) -> None:
    # Pins the HF caveats in INSPECT_MIGRATION.md; recheck them on Inspect upgrades.
    hf, _ = hf_stub
    default = get_model("hf/org/repo", memoize=False, device="cpu").api
    assert isinstance(default, hf.HuggingFaceAPI)
    assert default.do_sample is True
    assert default.max_tokens() == 2048
    greedy = get_model("hf/org/repo", memoize=False, device="cpu", do_sample=False).api
    assert isinstance(greedy, hf.HuggingFaceAPI)
    assert greedy.do_sample is False


def test_vllm_provider_configuration() -> None:
    for name in ("vllm/org/repo", "vllm/local"):
        model = get_model(name)
        assert model.api.model_name == name.removeprefix("vllm/")
    # User-supplied -M options are forwarded as server args; no server starts here.
    model = get_model(
        "vllm/org/repo",
        memoize=False,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
    )
    from inspect_ai.model._providers.vllm import VLLMAPI

    assert isinstance(model.api, VLLMAPI)
    server_args = model.api.server_args
    assert {
        key: server_args[key]
        for key in (
            "max_model_len",
            "gpu_memory_utilization",
            "tokenizer_mode",
            "config_format",
            "load_format",
        )
    } == {
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.8,
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
    }


def test_structured_reasoning() -> None:
    from inspect_ai.model import (
        ChatCompletionChoice,
        ChatMessageAssistant,
        ContentReasoning,
        ContentText,
    )

    output = ModelOutput(
        model="mockllm/model",
        choices=[
            ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content=[
                        ContentReasoning(reasoning="secret"),
                        ContentText(text="2"),
                    ]
                ),
                stop_reason="stop",
            )
        ],
    )
    assert output.completion == "2"
