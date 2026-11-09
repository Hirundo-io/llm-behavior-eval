"""Benchmark-agnostic helpers for the native Inspect tests.

Benchmark test files import these instead of copying them; ``pytest`` puts
``tests/`` on ``sys.path`` (``pythonpath`` in ``pyproject.toml``), so import
them as ``from inspect_helpers import ...``. Fixtures live in ``conftest.py``.
Only Inspect test modules import this file, so it may import ``inspect_ai``.
"""

import json
import subprocess
import sys
import threading
import zipfile
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest
import yaml
from datasets import Dataset
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log
from inspect_ai.model import GenerateConfig, Model, ModelOutput, get_model
from inspect_ai.scorer import Metric, SampleScore, Score

from llm_behavior_eval.common.parity import parity_config_path

GENERATION_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "seed",
    "frequency_penalty",
    "presence_penalty",
    "reasoning_effort",
    "reasoning_tokens",
    "reasoning_summary",
    "extra_body",
)
"""Inference settings that benchmark code must never set for either role."""


def generation_settings(config: GenerateConfig) -> dict[str, Any]:
    """The ``GENERATION_FIELDS`` a config sets; ``{}`` means none."""
    return {
        name: getattr(config, name)
        for name in GENERATION_FIELDS
        if getattr(config, name) is not None
    }


def rate(metric: Metric) -> Callable[[list[SampleScore]], float]:
    """``metric`` typed as returning a float (Inspect types it as ``Value``)."""
    return cast("Callable[[list[SampleScore]], float]", metric)


def mapping(metric: Metric) -> Callable[[list[SampleScore]], dict[str, float]]:
    """``metric`` typed as returning a mapping of named values."""
    return cast("Callable[[list[SampleScore]], dict[str, float]]", metric)


def eval_metrics(log: EvalLog) -> dict[str, Any]:
    """Metrics of a successful log's single scorer result.

    Every benchmark metric is unreduced (INSPECT_MIGRATION.md, "Epochs"), so
    the scorer writes exactly one ``EvalScore`` with no reducer.
    """
    assert log.status == "success", log.error
    assert log.results is not None
    [result] = log.results.scores
    assert result.reducer is None
    return {name: metric.value for name, metric in result.metrics.items()}


def sample_scores(log: EvalLog, scorer: str) -> list[Score]:
    """Every sample's score from ``scorer``, in log order."""
    assert log.status == "success", log.error
    assert log.samples
    scores = []
    for sample in log.samples:
        assert sample.scores
        scores.append(sample.scores[scorer])
    return scores


def only_log(log_dir: Path) -> EvalLog:
    """The single log in ``log_dir``."""
    logs = list_eval_logs(str(log_dir))
    assert len(logs) == 1
    return read_eval_log(logs[0])


def run_inspect_cli(*args: str, cwd: Path | None = None) -> None:
    """Run ``python -m inspect_ai *args --display none`` and require success.

    Pass a ``cwd`` other than the repository root: ``python -m`` puts the
    working directory first on ``sys.path``, where the source-tree egg-info
    hides the install metadata Inspect needs to prefix registry names.
    """
    result = subprocess.run(
        [sys.executable, "-m", "inspect_ai", *args, "--display", "none"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def mock_grader(reply: str, name: str = "mockllm/grader") -> Model:
    """A mock model that answers every request with ``reply``."""
    return get_model(
        name,
        custom_outputs=lambda *args: ModelOutput.from_content(name, reply),
    )


def patch_scorer_grader(
    monkeypatch: pytest.MonkeyPatch, scorer_module: ModuleType, reply: str
) -> list[str]:
    """Route ``scorer_module.get_model`` to a model that always replies ``reply``.

    For scorers called directly, outside an eval with a grader role.

    Returns:
        The prompts the grader received, in order.
    """
    calls: list[str] = []
    grader = get_model("mockllm/model")

    async def generate(prompt: Any, **kwargs: Any) -> ModelOutput:
        calls.append(prompt)
        return ModelOutput.from_content("mockllm/model", reply)

    monkeypatch.setattr(grader, "generate", generate)
    monkeypatch.setattr(scorer_module, "get_model", lambda **kwargs: grader)
    return calls


def snapshot(directory: Path, rows: list[dict[str, Any]], name: str = "data") -> str:
    """Save ``rows`` as a ``save_to_disk`` snapshot for ``-T dataset_path=...``."""
    path = directory / name
    Dataset.from_list(rows).save_to_disk(path)
    return str(path)


def mock_parity_config(
    package: str, directory: Path, grader: str = "mockllm/model"
) -> Path:
    """Write the shipped parity config with only the grader swapped for a mock.

    Args:
        package: Benchmark package, for example ``llm_behavior_eval.bbq``.
        directory: Where to write ``parity.yaml``.
        grader: Mock model that replaces the grader; its args are dropped.

    Returns:
        The written file.
    """
    config = yaml.safe_load(parity_config_path(package).read_text())
    config["model_roles"]["grader"]["model"] = grader
    config["model_roles"]["grader"]["args"] = {}
    path = directory / "parity.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def assert_grader_required(task: Task, tmp_path: Path, **eval_args: Any) -> EvalLog:
    """Run ``task`` without a grader role; it must fail and name the grader.

    ``eval_args`` go to ``inspect_ai.eval`` (for example a custom ``model``);
    the model under test defaults to ``mockllm/model``.
    """
    options: dict[str, Any] = {
        "model": "mockllm/model",
        "log_dir": str(tmp_path),
        "display": "none",
    } | eval_args
    [log] = inspect_eval(task, **options)
    assert log.status == "error"
    assert log.error is not None
    assert "grader" in log.error.message
    return log


def assert_no_credentials_logged(log: EvalLog, secret: str) -> None:
    """No file of ``log`` (a ``.eval`` archive or a ``.json`` log) holds ``secret``."""
    location = Path(log.location)
    if zipfile.is_zipfile(location):
        with zipfile.ZipFile(location) as archive:
            for name in archive.namelist():
                assert secret.encode() not in archive.read(name), name
    else:
        assert secret.encode() not in location.read_bytes()


@dataclass
class LocalOpenAIJudge:
    """A loopback OpenAI-compatible chat endpoint (see ``openai_judge``)."""

    base_url: str
    reply: str | Callable[[dict[str, Any]], str] = "SCORE: 0"
    """The reply text, or a function of the request body that returns it."""
    requests: list[dict[str, Any]] = field(default_factory=list)
    """Request bodies, in arrival order."""
    paths: list[str] = field(default_factory=list)
    """Request paths, in arrival order."""
    authorizations: list[str | None] = field(default_factory=list)
    """``Authorization`` headers (``None`` when absent), in arrival order."""


def message_text(message: dict[str, Any]) -> str:
    """The text of one chat message in an OpenAI request body."""
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content)


@contextmanager
def local_openai_judge(
    reply: str | Callable[[dict[str, Any]], str] = "SCORE: 0",
) -> Generator[LocalOpenAIJudge]:
    """Serve chat completions on loopback that always answer ``judge.reply``.

    Use it as an ``openai-api/<service>/<model>`` grader's ``base_url``. Set
    ``reply`` before the eval; read ``requests`` afterwards to see exactly
    what Inspect sent. The ``openai_judge`` fixture (``conftest.py``) wraps it.
    """
    judge = LocalOpenAIJudge(base_url="", reply=reply)

    # BaseHTTPRequestHandler logs each request to stderr, which pytest captures.
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            judge.paths.append(self.path)
            judge.authorizations.append(self.headers.get("Authorization"))
            judge.requests.append(body)
            reply = judge.reply if isinstance(judge.reply, str) else judge.reply(body)
            payload = json.dumps(
                {
                    "id": "local-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body.get("model", "judge"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": reply},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    judge.base_url = f"http://127.0.0.1:{server.server_port}/v1"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield judge
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
