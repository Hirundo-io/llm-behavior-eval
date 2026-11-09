"""CPU-only smoke run of every registered task on local data and mock models.

Each task runs by its registered name, offline, with a mock answer model and a
mock grader. The written ``.eval`` log must carry the registered task name and
every target metric inspect-brief will look for.
"""

import socket

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput, ModelUsage, get_model

from llm_behavior_eval import _registry, list_dataset_presets
from llm_behavior_eval.common.target_metrics import (
    load_target_metrics,
    missing_target_metrics,
)
from tests.native_fixtures import GRADER_VERDICTS, local_task_dataset, require_fixture

TEMPLATE = "inspect_template"


def mock_output(model: str, content: str) -> ModelOutput:
    output = ModelOutput.from_content(model, content)
    # Otherwise MockLLM estimates usage with tiktoken, which downloads its
    # vocabulary on a cold machine. This smoke tests task wiring, not usage.
    output.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
    return output


@pytest.mark.parametrize("name", _registry.__all__)
def test_registered_task(name, tmp_path, monkeypatch):
    require_fixture(name)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("INSPECT_BRIEF_ENABLED", "false")
    monkeypatch.delenv("LLM_BEHAVIOR_EVAL_MLFLOW_ENABLED", raising=False)

    def no_network(*args, **kwargs):
        raise AssertionError("Smoke attempted network access")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    args = (
        {}
        if name == TEMPLATE
        else {"dataset_path": str(local_task_dataset(name, tmp_path, monkeypatch))}
    )
    calls = []

    def grade(messages, tools, choice, config):
        calls.append(messages)
        assert "secret-reasoning" not in str(messages)
        return mock_output("mockllm/grader", GRADER_VERDICTS[name])

    [log] = inspect_eval(
        f"llm_behavior_eval/{name}",
        task_args=args,
        limit=2,
        model=get_model(
            "mockllm/model",
            custom_outputs=[
                mock_output("mockllm/model", "<think>secret-reasoning</think>answer")
                for _ in range(2)
            ],
        ),
        # Every task gets this grader, the template included: its own mock
        # grader reports no usage, so MockLLM would fetch a tiktoken vocabulary.
        model_roles={"grader": get_model("mockllm/grader", custom_outputs=grade)},
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert log.status == "success", log.error.traceback if log.error else None
    assert calls
    saved = read_eval_log(log.location)
    assert saved.eval.task == f"llm_behavior_eval/{name}"
    assert saved.samples and saved.results
    assert missing_target_metrics(saved) == []


def test_every_preset_maps_to_a_registered_task():
    registered = {f"llm_behavior_eval/{name}" for name in _registry.__all__}
    assert set(load_target_metrics()) == registered
    mapped = {
        mapping.task_name
        for preset in list_dataset_presets()
        for mapping in preset.inspect_tasks
    }
    # Each legacy benchmark is reachable; new benchmarks need no legacy preset.
    assert mapped <= registered
    assert mapped == {
        f"llm_behavior_eval/{name}"
        for name in (
            "bbq",
            "bloom",
            "ccpc",
            "halueval",
            "medhallu",
            "orbench",
            "prompt_injection",
            "unqover",
            "xstest",
        )
    }


def test_ccpc_fixture_requires_explicit_fingerprint_override(tmp_path, monkeypatch):
    from llm_behavior_eval.ccpc.dataset import load_ccpc_dataset

    with monkeypatch.context() as fixture_patch:
        path = local_task_dataset("ccpc", tmp_path, fixture_patch)
    with pytest.raises(ValueError, match="fingerprint"):
        load_ccpc_dataset(dataset_path=str(path))
