"""Contracts of the shared code in ``llm_behavior_eval.common`` and its discovery."""

import json
import math
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import EvalLog
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    ModelName,
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState

from llm_behavior_eval.common import dataset as common_dataset
from llm_behavior_eval.common.dataset import (
    DATASET_TOKEN_VARIABLE,
    dataset_token,
    is_saved_dataset,
    load_dataset,
)
from llm_behavior_eval.common.scoring import (
    ask_grader,
    completion_scorer,
    judged_answer,
    stop_reason,
)
from llm_behavior_eval.common.target_metrics import (
    load_target_metrics,
    missing_target_metrics,
)

from inspect_helpers import (
    assert_grader_required,
    eval_metrics,
    mock_grader,
    reduced_metrics,
    sample_scores,
)

ROWS = [{"id": f"r{i}", "text": f"t{i}"} for i in range(12)]


def to_sample(record: dict[str, Any]) -> Sample:
    return Sample(id=record["id"], input=record["text"])


def permutation(seed: int) -> list[int]:
    return cast(
        "list[int]", HFDataset.from_dict({"i": list(range(12))}).shuffle(seed=seed)["i"]
    )


# Dataset loading


def test_dataset_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DATASET_TOKEN_VARIABLE, raising=False)
    assert dataset_token() is None
    monkeypatch.setenv(DATASET_TOKEN_VARIABLE, "")
    assert dataset_token() is None
    monkeypatch.setenv(DATASET_TOKEN_VARIABLE, "hf_x")
    assert dataset_token() == "hf_x"


@pytest.mark.parametrize("seed", [None, 42])
def test_pinned_hub_load_uses_the_retrying_loader(
    monkeypatch: pytest.MonkeyPatch,
    hub: Callable[..., list[Any]],
    seed: int | None,
) -> None:
    calls = hub(ROWS)
    loaders: list[str] = []
    pinned = common_dataset.pinned_hf_dataset

    def spy(*args: Any, **kwargs: Any) -> Any:
        loaders.append("pinned")
        return pinned(*args, **kwargs)

    monkeypatch.setattr(common_dataset, "pinned_hf_dataset", spy)
    monkeypatch.setenv(DATASET_TOKEN_VARIABLE, "hf_secret")
    data = load_dataset(
        "org/data",
        split="test",
        sample_fields=to_sample,
        revision="abc123",
        name="default",
        seed=seed,
    )
    assert loaders == ["pinned"]
    kwargs = calls[0][1]
    assert (kwargs["path"], kwargs["split"], kwargs["name"]) == (
        "org/data",
        "test",
        "default",
    )
    assert (kwargs["revision"], kwargs["token"]) == ("abc123", "hf_secret")
    order = list(range(12)) if seed is None else permutation(seed)
    assert [s.id for s in data] == [f"r{i}" for i in order]


def test_local_source_without_revision_uses_inspect_loader(
    monkeypatch: pytest.MonkeyPatch, hub: Callable[..., list[Any]]
) -> None:
    hub(ROWS)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("an unpinned source must not need a revision")

    monkeypatch.setattr(common_dataset, "pinned_hf_dataset", fail)
    data = load_dataset("some/folder", split="train", sample_fields=to_sample, seed=7)
    assert [s.id for s in data] == [f"r{i}" for i in permutation(7)]


@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize("seed", [None, 42])
def test_save_to_disk_folder_matches_the_hub(
    tmp_path: Path, hf_dataset_cache: Path, as_dict: bool, seed: int | None
) -> None:
    raw = HFDataset.from_list(ROWS)
    path = tmp_path / "saved"
    (DatasetDict({"test": raw}) if as_dict else raw).save_to_disk(path)
    assert is_saved_dataset(str(path))
    files = sorted(p.relative_to(path) for p in path.rglob("*"))
    data = load_dataset(str(path), split="test", sample_fields=to_sample, seed=seed)
    order = list(range(12)) if seed is None else permutation(seed)
    assert [s.id for s in data] == [f"r{i}" for i in order]
    assert isinstance(data, MemoryDataset) and data.shuffled == (seed is not None)
    # Only read: no indices cache files are written into the folder.
    assert sorted(p.relative_to(path) for p in path.rglob("*")) == files


def test_saved_dataset_dict_needs_the_split(tmp_path: Path) -> None:
    DatasetDict({"train": HFDataset.from_list(ROWS)}).save_to_disk(tmp_path / "s")
    with pytest.raises(ValueError, match="no split 'test'"):
        load_dataset(str(tmp_path / "s"), split="test", sample_fields=to_sample)


@pytest.mark.parametrize("index", ["{}", '["train"]', '{"splits": "train"}'])
def test_saved_dataset_dict_index_needs_splits(tmp_path: Path, index: str) -> None:
    DatasetDict({"train": HFDataset.from_list(ROWS)}).save_to_disk(tmp_path / "s")
    (tmp_path / "s" / "dataset_dict.json").write_text(index)
    with pytest.raises(ValueError, match="lists no splits"):
        load_dataset(str(tmp_path / "s"), split="train", sample_fields=to_sample)


def test_stray_dataset_dict_index_is_reported(tmp_path: Path) -> None:
    write_rows(tmp_path / "d", "What is")
    (tmp_path / "d" / "dataset_dict.json").write_text('{"note": "unrelated"}')
    with pytest.raises(ValueError, match="remove the stray dataset_dict.json"):
        load_dataset(str(tmp_path / "d"), split="train", sample_fields=to_sample)


def write_rows(folder: Path, prefix: str) -> None:
    folder.mkdir(exist_ok=True)
    (folder / "train.jsonl").write_text(
        "".join(
            json.dumps({"id": f"r{i}", "text": f"{prefix}{i}"}) + "\n" for i in range(5)
        )
    )


def test_replaced_local_file_is_read_again(
    tmp_path: Path, hf_dataset_cache: Path
) -> None:
    folder = tmp_path / "data"
    write_rows(folder, "What is 2 + 2? ")
    first = load_dataset(str(folder), split="train", sample_fields=to_sample)
    assert [s.input for s in first][0] == "What is 2 + 2? 0"
    # Same path, new contents: the new rows are scored, not a cached copy.
    write_rows(folder, "What is 3 + 3? ")
    second = load_dataset(str(folder), split="train", sample_fields=to_sample)
    assert [s.input for s in second] == [f"What is 3 + 3? {i}" for i in range(5)]


def test_plain_folder_is_not_a_saved_dataset(
    tmp_path: Path, hf_dataset_cache: Path
) -> None:
    write_rows(tmp_path / "data", "t")
    assert not is_saved_dataset(str(tmp_path / "data"))
    # A lone dataset_info.json neither makes it a saved folder nor stops it
    # loading as data files.
    (tmp_path / "data" / "dataset_info.json").write_text("{}")
    assert not is_saved_dataset(str(tmp_path / "data"))
    assert (
        len(
            load_dataset(str(tmp_path / "data"), split="train", sample_fields=to_sample)
        )
        == 5
    )


def test_lone_state_json_is_reported(tmp_path: Path, hf_dataset_cache: Path) -> None:
    # datasets refuses any folder with a state.json and points to
    # load_from_disk, which cannot read an incomplete snapshot either.
    write_rows(tmp_path / "data", "t")
    (tmp_path / "data" / "state.json").write_text("{}")
    assert not is_saved_dataset(str(tmp_path / "data"))
    with pytest.raises(
        ValueError, match="holds a state.json but is not a save_to_disk"
    ):
        load_dataset(str(tmp_path / "data"), split="train", sample_fields=to_sample)


# Scoring


def state_with(output: ModelOutput) -> TaskState:
    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id=1,
        epoch=1,
        input="q",
        messages=[],
        output=output,
    )


def test_judged_answer_and_stop_reason() -> None:
    message = ChatMessageAssistant(
        content=[ContentReasoning(reasoning="secret"), ContentText(text="answer")]
    )
    output = ModelOutput(
        model="m",
        choices=[ChatCompletionChoice(message=message, stop_reason="max_tokens")],
    )
    state = state_with(output)
    assert judged_answer(state) == "answer"
    assert judged_answer(state, include_reasoning=True) == (
        "<reasoning>\nsecret\n</reasoning>\nanswer"
    )
    assert stop_reason(state) == "max_tokens"
    empty = state_with(ModelOutput(model="m", choices=[]))
    assert stop_reason(empty) is None
    assert judged_answer(empty, include_reasoning=True) == ""


@scorer(metrics=[])
def echo_grader() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        output = await ask_grader("system text", "user text")
        return Score(value=0, answer=output.completion)

    return score


def run(task: Task, tmp_path: Path, **options: Any) -> EvalLog:
    defaults: dict[str, Any] = {
        "model": mock_grader("answer", "mockllm/answer"),
        "log_dir": str(tmp_path),
        "display": "none",
    }
    [log] = inspect_eval(task, **(defaults | options))
    return log


def test_ask_grader_sends_system_and_user_with_role_settings(tmp_path: Path) -> None:
    seen: list[Any] = []

    def grade(
        messages: Any, tools: Any, choice: Any, config: GenerateConfig
    ) -> ModelOutput:
        seen.append(([(m.role, m.text) for m in messages], config.temperature))
        return ModelOutput.from_content("mockllm/grader", "reply")

    grader = get_model(
        "mockllm/grader", custom_outputs=grade, config=GenerateConfig(temperature=0.7)
    )
    task = Task(dataset=[Sample(input="q")], scorer=echo_grader())
    log = run(task, tmp_path, model_roles={"grader": grader}, temperature=0.2)
    assert log.status == "success", log.error
    assert seen == [([("system", "system text"), ("user", "user text")], 0.7)]


def test_ask_grader_requires_the_grader_role(tmp_path: Path) -> None:
    assert_grader_required(
        Task(dataset=[Sample(input="q")], scorer=echo_grader()), tmp_path
    )


@pytest.mark.parametrize(
    ("stop", "value"),
    [("stop", 0), ("max_tokens", 1), ("model_length", 1), ("content_filter", 0)],
)
def test_completion_scorer(tmp_path: Path, stop: str, value: int) -> None:
    def answer(*args: Any) -> ModelOutput:
        out = ModelOutput.from_content("mockllm/answer", "a")
        out.choices[0].stop_reason = stop  # type: ignore[assignment]
        return out

    task = Task(dataset=[Sample(input="q")], scorer=completion_scorer())
    log = run(task, tmp_path, model=get_model("mockllm/answer", custom_outputs=answer))
    assert eval_metrics(log) == {"incomplete_response_rate": value}
    [score] = sample_scores(log, "completion_scorer")
    assert score.metadata == {"stop_reason": stop}


def test_completion_scorer_unscored_without_stop_reason(tmp_path: Path) -> None:
    def answer(*args: Any) -> ModelOutput:
        out = ModelOutput.from_content("mockllm/answer", "a")
        out.choices[0].stop_reason = "unknown"
        return out

    task = Task(dataset=[Sample(input="q")], scorer=completion_scorer())
    log = run(task, tmp_path, model=get_model("mockllm/answer", custom_outputs=answer))
    assert math.isnan(eval_metrics(log)["incomplete_response_rate"])
    assert log.results and log.results.scores[0].unscored_samples == 1


def test_completion_scorer_pools_epochs(tmp_path: Path) -> None:
    calls = {"n": 0}

    def answer(*args: Any) -> ModelOutput:
        calls["n"] += 1
        out = ModelOutput.from_content("mockllm/answer", "a")
        out.choices[0].stop_reason = "max_tokens" if calls["n"] == 1 else "stop"
        return out

    task = Task(dataset=[Sample(input="q")], scorer=completion_scorer())
    assert reduced_metrics(task) == []
    log = run(
        task,
        tmp_path,
        model=get_model("mockllm/answer", custom_outputs=answer),
        epochs=2,
    )
    assert eval_metrics(log) == {"incomplete_response_rate": 0.5}


# Target metrics and discovery


def make_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    # A package another test imported under this name would shadow this one.
    monkeypatch.delitem(sys.modules, name, raising=False)
    return root


def metric(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "is_percentage": True,
        "is_higher_better": False,
        "is_normalized": True,
    }


def test_load_target_metrics_merges_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_root(tmp_path, monkeypatch, "fake_root_a")
    for package, task in [("b", "beta"), ("a", "alpha")]:
        (root / package).mkdir()
        (root / package / "__init__.py").write_text("")
        (root / package / "target_metrics.json").write_text(
            json.dumps({f"fake_root_a/{task}": [metric("m")]})
        )
    assert list(load_target_metrics("fake_root_a")) == [
        "fake_root_a/alpha",
        "fake_root_a/beta",
    ]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ([metric("m")], "expected a JSON object, got list"),
        ({"other/task": [metric("m")]}, "must start with fake_root_b/"),
        ({"fake_root_b/task": []}, "needs a non-empty list"),
        ({"fake_root_b/task": metric("m")}, "needs a non-empty list"),
        ({"fake_root_b/task": ["m"]}, r"task\[0\] needs exactly the keys"),
        ({"fake_root_b/task": [{"name": "m"}]}, r"task\[0\] needs exactly the keys"),
        ({"fake_root_b/task": [metric("m") | {"name": ["m"]}]}, r"\.name must be"),
        ({"fake_root_b/task": [metric("m") | {"name": ""}]}, r"\.name must be"),
        (
            {"fake_root_b/task": [metric("m") | {"is_percentage": "false"}]},
            r"\.is_percentage must be true or false, got 'false'",
        ),
        (
            {"fake_root_b/task": [metric("m"), metric("n") | {"is_normalized": 1}]},
            r"task\[1\]\.is_normalized must be true or false",
        ),
    ],
)
def test_load_target_metrics_rejects_bad_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: object, message: str
) -> None:
    root = make_root(tmp_path, monkeypatch, "fake_root_b")
    (root / "p").mkdir()
    (root / "p" / "target_metrics.json").write_text(json.dumps(content))
    with pytest.raises(ValueError, match=message) as error:
        load_target_metrics("fake_root_b")
    assert "target_metrics.json" in str(error.value)


def test_load_target_metrics_rejects_a_repeated_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_root(tmp_path, monkeypatch, "fake_root_c")
    for package in ("a", "b"):
        (root / package).mkdir()
        (root / package / "target_metrics.json").write_text(
            json.dumps({"fake_root_c/task": [metric("m")]})
        )
    with pytest.raises(ValueError, match="is repeated"):
        load_target_metrics("fake_root_c")


def test_load_target_metrics_skips_build_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A benchmark's directory is also its distribution's project directory, so
    # an in-place wheel build leaves build/ (a copy of the package) and an
    # *.egg-info folder inside the package; groups have no __init__.py.
    root = make_root(tmp_path, monkeypatch, "fake_root_e")
    for package, task in [("solo", "solo"), ("group/member", "member")]:
        directory = root / package
        copy = directory / "build" / "lib" / "fake_root_e" / package
        for where in (directory, copy, directory / "fake_root_e.egg-info"):
            where.mkdir(parents=True, exist_ok=True)
            (where / "__init__.py").write_text("")
            (where / "target_metrics.json").write_text(
                json.dumps({f"fake_root_e/{task}": [metric("m")]})
            )
    assert list(load_target_metrics("fake_root_e")) == [
        "fake_root_e/member",
        "fake_root_e/solo",
    ]


def test_load_target_metrics_skips_pycache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A group level has no __init__.py, but its __pycache__ is not a package.
    root = make_root(tmp_path, monkeypatch, "fake_root_f")
    for where in (root / "group" / "member", root / "group" / "__pycache__"):
        where.mkdir(parents=True)
        (where / "target_metrics.json").write_text(
            json.dumps({f"fake_root_f/{where.name}": [metric("m")]})
        )
    (root / "group" / "member" / "__init__.py").write_text("")
    assert list(load_target_metrics("fake_root_f")) == ["fake_root_f/member"]


def test_missing_target_metrics(tmp_path: Path) -> None:
    task = Task(
        dataset=[Sample(input="q")],
        scorer=completion_scorer(),
        name="llm_behavior_eval/demo",
    )
    log = run(task, tmp_path)
    targets = {
        "llm_behavior_eval/demo": [metric("incomplete_response_rate"), metric("x")]
    }
    assert missing_target_metrics(log.location, targets) == ["x"]
    with pytest.raises(ValueError, match="no target metrics"):
        missing_target_metrics(log, {})


def test_load_target_metrics_spans_namespace_portions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two portions of a PEP 420 namespace (no __init__.py) on two sys.path
    # entries, as wheels installed without the legacy distribution in two
    # site-packages directories, one with a grouped benchmark.
    for portion, package, task in [
        ("one", "solo", "solo"),
        ("two", "group/member", "member"),
    ]:
        directory = tmp_path / portion / "fake_root_d" / package
        directory.mkdir(parents=True)
        (directory / "target_metrics.json").write_text(
            json.dumps({f"fake_root_d/{task}": [metric("m")]})
        )
        monkeypatch.syspath_prepend(str(tmp_path / portion))
    monkeypatch.delitem(sys.modules, "fake_root_d", raising=False)
    assert list(load_target_metrics("fake_root_d")) == [
        "fake_root_d/member",
        "fake_root_d/solo",
    ]


def test_entry_points_do_not_load_the_legacy_engine() -> None:
    # Inspect loads every inspect_ai entry point named after a task's package
    # prefix; none of them may import the legacy engine (torch, transformers).
    code = (
        "import sys; from importlib.metadata import entry_points; "
        "[ep.load() for ep in entry_points(group='inspect_ai') "
        "if ep.name == 'llm_behavior_eval']; "
        "print(any(m.startswith(('llm_behavior_eval.evaluation_utils', 'torch')) "
        "for m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
