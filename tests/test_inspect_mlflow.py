"""Native reporting tests use only temporary local MLflow file stores."""

import asyncio
import builtins
import sys

import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai._eval.eval import EvalLogs
from inspect_ai.dataset import Sample
from inspect_ai.hooks import RunEnd, RunStart
from inspect_ai.scorer import exact
from inspect_ai.solver import generate

from llm_behavior_eval.hooks import mlflow as hook_module
from llm_behavior_eval.hooks._mlflow_worker import report
from llm_behavior_eval.hooks.mlflow import (
    PREFIX,
    MlflowHook,
    adapter_run_id,
    metric_step,
    report_payload,
)

mlflow = pytest.importorskip("mlflow")


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv(PREFIX + "ENABLED", "1")
    monkeypatch.setenv(PREFIX + "TIMEOUT", "60")
    for name in ("RUN_ID", "ADAPTER_REF", "STEP", "ARTIFACT_SUBFOLDER"):
        monkeypatch.delenv(PREFIX + name, raising=False)
    return mlflow.MlflowClient(tracking_uri=uri)


@pytest.mark.parametrize(
    "ref,uri,expected",
    [
        ("mlflow://abc/hf/checkpoint-000020", "", "abc"),
        ("runs:/abc/checkpoint-20", "", "abc"),
        ("https://host/runs/abc/artifacts", "https://host", "abc"),
        ("https://other/runs/abc", "https://host", None),
        ("local/checkpoint-20", "", None),
    ],
)
def test_adapter(ref, uri, expected):
    assert adapter_run_id(ref, uri) == expected
    assert metric_step(ref) == (20 if "checkpoint-" in ref else 0)


def test_disabled_does_not_import_mlflow(monkeypatch):
    monkeypatch.delenv(PREFIX + "ENABLED", raising=False)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "mlflow":
            raise AssertionError("disabled hook imported mlflow")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    hook = MlflowHook()
    asyncio.run(hook.on_run_start(RunStart(None, "run", [])))
    assert not hook._runs


def test_run_selection(tracking, monkeypatch):
    experiment = tracking.create_experiment("training")
    explicit = tracking.create_run(experiment).info.run_id
    inferred = tracking.create_run(experiment).info.run_id
    monkeypatch.setenv(PREFIX + "ADAPTER_REF", f"mlflow://{inferred}/checkpoint-000020")
    hook = MlflowHook()
    with mlflow.start_run(experiment_id=experiment) as active:
        asyncio.run(hook.on_run_start(RunStart(None, "active", [])))
        assert hook._runs["active"]["run_id"] == active.info.run_id
        monkeypatch.setenv(PREFIX + "RUN_ID", explicit)
        asyncio.run(hook.on_run_start(RunStart(None, "explicit", [])))
        assert hook._runs["explicit"]["run_id"] == explicit
    monkeypatch.delenv(PREFIX + "RUN_ID")
    asyncio.run(hook.on_run_start(RunStart(None, "inferred", [])))
    assert hook._runs["inferred"]["run_id"] == inferred
    assert hook._runs["inferred"]["step"] == 20


def test_native_lifecycle(tracking, tmp_path, monkeypatch):
    monkeypatch.setenv(PREFIX + "STEP", "20")
    logs = inspect_eval(
        [
            Task(
                dataset=[Sample(input="one", target="Default output")],
                solver=generate(),
                scorer=exact(),
                name="first",
            ),
            Task(
                dataset=[Sample(input="two", target="Default output")],
                solver=generate(),
                scorer=exact(),
                name="second",
            ),
        ],
        model="mockllm/model",
        log_dir=str(tmp_path / "logs"),
        display="none",
    )
    assert all(log.status == "success" for log in logs)
    experiment = tracking.get_experiment_by_name("llm-behavior-eval")
    assert experiment is not None
    runs = tracking.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.info.status == "FINISHED"
    assert run.data.metrics["datasets_attached"] == 2
    keys = [key for key in run.data.metrics if key.endswith("/mean")]
    assert len(keys) == 2
    assert all(
        tracking.get_metric_history(run.info.run_id, key)[0].step == 20 for key in keys
    )
    for log in logs:
        artifacts = tracking.list_artifacts(
            run.info.run_id, f"llm-behavior-eval/{log.eval.run_id}/{log.eval.eval_id}"
        )
        assert len(artifacts) == 1
        assert artifacts[0].path.endswith(".eval")
    payload = report_payload(
        RunEnd(None, "again", None, EvalLogs(logs)),
        {
            "tracking_uri": tracking.tracking_uri,
            "run_id": run.info.run_id,
            "step": 30,
            "subfolder": "checkpoint-30",
        },
    )
    # Percentage values are not scaled; repeated task configuration remains stable.
    assert logs[0].results is not None
    logs[0].results.scores[0].metrics["mean"].value = 75.0
    payload = report_payload(
        RunEnd(None, "again", None, EvalLogs(logs)), payload["config"]
    )
    assert 75.0 in payload["tasks"][0]["metrics"].values()
    report(payload)
    # Identical task names with different dataset/config identities stay isolated.
    variant = logs[0].model_copy()
    variant.eval = logs[0].eval.model_copy(deep=True)
    variant.results = logs[0].results.model_copy(deep=True)
    variant.eval.task_args = {"variant": "second"}
    assert variant.results is not None
    variant.results.scores[0].metrics["mean"].value = 0.25
    combined = report_payload(
        RunEnd(None, "variants", None, EvalLogs([logs[0], variant])), payload["config"]
    )
    first, second = (task["metrics"] for task in combined["tasks"])
    assert not set(first).intersection(second)
    assert 75.0 in first.values() and 0.25 in second.values()
    variant.results.scores[0].metrics["mean"].value = float("nan")
    omitted = report_payload(
        RunEnd(None, "empty", None, EvalLogs([variant])), payload["config"]
    )
    assert not any(key.endswith("/mean") for key in omitted["tasks"][0]["metrics"])
    assert tracking.get_run(run.info.run_id).info.status == "FINISHED"
    assert tracking.get_run(run.info.run_id).data.metrics["datasets_attached"] == 4


def test_owned_failed_and_attached_unchanged(tracking):
    config = {
        "tracking_uri": tracking.tracking_uri,
        "run_id": None,
        "experiment": "failure",
        "run_name": "failed",
        "step": 0,
        "subfolder": "",
    }
    payload = {
        "config": config,
        "tasks": [],
        "failed": True,
        "inspect_run_id": "failure",
    }
    report(payload)
    experiment = tracking.get_experiment_by_name("failure")
    run = tracking.search_runs([experiment.experiment_id])[0]
    assert run.info.status == "FAILED"
    attached = tracking.create_run(experiment.experiment_id)
    config["run_id"] = attached.info.run_id
    report(payload)
    assert tracking.get_run(attached.info.run_id).info.status == "RUNNING"


def test_reporting_failure_is_redacted(tracking, monkeypatch, caplog):
    async def fail(*args):
        raise RuntimeError("secret-token")

    monkeypatch.setattr(hook_module, "_send", fail)
    hook = MlflowHook()
    asyncio.run(hook.on_run_start(RunStart(None, "run", [])))
    asyncio.run(
        hook.on_run_end(RunEnd(None, "run", ValueError("eval error"), EvalLogs([])))
    )
    assert "secret-token" not in caplog.text
    assert "failed or timed out" in caplog.text
    assert not hook._runs


def test_worker_timeout_kills_and_reaps(monkeypatch):
    original = asyncio.create_subprocess_exec
    processes = []

    async def stalled(*args, **kwargs):
        process = await original(
            sys.executable, "-c", "import time; time.sleep(60)", **kwargs
        )
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", stalled)
    with pytest.raises(TimeoutError):
        asyncio.run(hook_module._send({}, 0.05))
    assert processes[0].returncode is not None


def test_unavailable_mlflow_is_optional(monkeypatch, caplog):
    monkeypatch.setenv(PREFIX + "ENABLED", "1")
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    hook = MlflowHook()
    asyncio.run(hook.on_run_start(RunStart(None, "missing", [])))
    assert not hook._runs
    assert "unavailable" in caplog.text
