"""CPU-only registered-task discovery, source conversion, scoring and log smoke."""

import socket

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput, ModelUsage, get_model

from llm_behavior_eval import _registry, list_dataset_presets
from llm_behavior_eval.target_metrics import target_metrics
from tests.native_fixtures import local_task_dataset


@pytest.mark.parametrize("name", _registry.__all__)
def test_registered_task(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("INSPECT_BRIEF_ENABLED", "false")
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "token-cache"))

    def no_network(*args, **kwargs):
        raise AssertionError("Smoke attempted network access")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    args = (
        {}
        if name == "inspect_template"
        else {"dataset_path": str(local_task_dataset(name, tmp_path, monkeypatch))}
    )
    verdict = {
        "bbq": "gt_agree: yes candidate_uncertain: no",
        "bloom": "gt_agree: yes candidate_uncertain: no",
        "unqover": "gt_agree: yes candidate_uncertain: no",
        "halueval": "A",
        "medhallu": "A",
        "prompt_injection": "Yes",
        "ccpc": "True",
        "xstest": "2_full_refusal",
        "orbench": "[[direct_refusal]]",
        "inspect_template": "SCORE: 1",
    }[name]
    calls = []

    def mock_output(model: str, content: str) -> ModelOutput:
        output = ModelOutput.from_content(model, content)
        # Otherwise MockLLM estimates usage with tiktoken, which downloads its
        # vocabulary on a cold machine. This smoke tests task wiring, not usage.
        output.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
        return output

    def grade(messages, tools, choice, config):
        calls.append(messages)
        assert "secret-reasoning" not in str(messages)
        return mock_output("mockllm/grader", verdict)

    log = inspect_eval(
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
        model_roles={"grader": get_model("mockllm/grader", custom_outputs=grade)},
        log_dir=str(tmp_path / "logs"),
        display="none",
    )[0]
    assert log.status == "success", log.error.traceback if log.error else None
    assert calls
    saved = read_eval_log(str(next((tmp_path / "logs").glob("*.eval"))))
    assert saved.eval.task == f"llm_behavior_eval/{name}"
    assert saved.samples and saved.results
    emitted = {key for score in saved.results.scores for key in score.metrics}
    assert {m["name"] for m in target_metrics()[saved.eval.task]} <= emitted


def test_registry_presets_and_target_schema():
    registered = {f"llm_behavior_eval/{n}" for n in _registry.__all__}
    assert set(target_metrics()) == registered
    assert {
        mapping.task_name
        for preset in list_dataset_presets()
        for mapping in preset.inspect_tasks
    } == registered - {"llm_behavior_eval/inspect_template"}
    for metrics in target_metrics().values():
        assert len({m["name"] for m in metrics}) == len(metrics)
        for metric in metrics:
            assert set(metric) == {
                "name",
                "is_percentage",
                "is_normalized",
                "is_higher_better",
            }
            assert isinstance(metric["name"], str)
            assert all(type(metric[k]) is bool for k in metric if k != "name")


def test_ccpc_fixture_requires_explicit_fingerprint_override(tmp_path, monkeypatch):
    from llm_behavior_eval.ccpc.dataset import load_ccpc_dataset

    with monkeypatch.context() as fixture_patch:
        path = local_task_dataset("ccpc", tmp_path, fixture_patch)
    with pytest.raises(ValueError, match="fingerprint"):
        load_ccpc_dataset(dataset_path=str(path), max_samples=1)
