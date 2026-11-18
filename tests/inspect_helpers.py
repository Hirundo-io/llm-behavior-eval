"""Test helpers both native benchmark suites use.

``pytest`` puts ``tests/`` on ``sys.path`` (``pythonpath`` in ``pyproject.toml``),
so import them as ``from inspect_helpers import ...``. Fixtures live in
``conftest.py``. Only Inspect test modules import this file.
"""

import json
import threading
import zipfile
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog
from inspect_ai.model import Model, ModelOutput, get_model
from inspect_ai.scorer import Metric, Score
from inspect_ai.util import registry_info


def reduced_metrics(target: Task | Callable[..., Any]) -> list[str]:
    """Metrics that would receive epoch-reduced scores.

    For a task, checks the metrics Inspect computes for each of its scorers
    (``Task(metrics=...)`` when given, otherwise each scorer's own). For a
    scorer, or a ``@scorer`` factory, checks its declared metrics. Inspect
    built-ins such as ``accuracy()`` and ``stderr()`` are reduced. Benchmark
    tests assert that this is empty for their tasks.

    Args:
        target: A constructed task, a scorer or a ``@scorer`` factory.

    Returns:
        Sorted registry names of the metrics not declared
        ``@metric(scores="unreduced")``.
    """
    scorers = (target.scorer or []) if isinstance(target, Task) else [target]
    names: set[str] = set()
    for scorer in scorers:
        # Inspect stores a scorer's metrics, or the task's metrics= override,
        # under this registry metadata key (scorer_metrics() reads it).
        declared = registry_info(scorer).metadata.get("metrics", [])
        for item in _flatten_metrics(declared):
            info = registry_info(item)
            if info.metadata.get("scores") != "unreduced":
                names.add(info.name)
    return sorted(names)


def _flatten_metrics(
    metrics: list[Metric | dict[str, list[Metric]]] | dict[str, list[Metric]],
) -> Iterator[Metric]:
    """Every metric in Inspect's ``metrics`` shapes (a list, a dict, or both)."""
    groups = [metrics] if isinstance(metrics, dict) else metrics
    for group in groups:
        if isinstance(group, dict):
            for members in group.values():
                yield from members
        else:
            yield group


def eval_metrics(log: EvalLog) -> dict[str, Any]:
    """Metrics of every scorer of a successful log, in one mapping.

    Every benchmark metric is unreduced (INSPECT_MIGRATION.md, "Epochs"), so
    each scorer writes exactly one ``EvalScore`` with no reducer. Metric keys
    must be unique across scorers (for example ``field_rate`` from the
    grading scorer and ``incomplete_response_rate`` from
    ``completion_scorer``).
    """
    assert log.status == "success", log.error
    assert log.results is not None
    names = [result.name for result in log.results.scores]
    assert len(names) == len(set(names)), names
    metrics: dict[str, Any] = {}
    for result in log.results.scores:
        assert result.reducer is None
        for name, metric in result.metrics.items():
            assert name not in metrics, name
            metrics[name] = metric.value
    return metrics


def sample_scores(log: EvalLog, scorer: str) -> list[Score]:
    """Every sample's score from ``scorer``, in log order."""
    assert log.status == "success", log.error
    assert log.samples
    scores = []
    for sample in log.samples:
        assert sample.scores
        scores.append(sample.scores[scorer])
    return scores


def mock_grader(reply: str, name: str = "mockllm/grader") -> Model:
    """A mock model that answers every request with ``reply``."""
    return get_model(
        name,
        custom_outputs=lambda *args: ModelOutput.from_content(name, reply),
    )


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
    reasoning: str | None = None
    """``reasoning_content`` to send, as vLLM's ``--reasoning-parser`` does."""
    finish_reason: str = "stop"
    """The choice's ``finish_reason``."""
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
    *,
    reasoning: str | None = None,
    finish_reason: str = "stop",
) -> Generator[LocalOpenAIJudge]:
    """Serve chat completions on loopback that always answer ``judge.reply``.

    Use it as an ``openai-api/<service>/<model>`` grader's ``base_url``. Set
    ``reply`` before the eval; read ``requests`` afterwards to see exactly
    what Inspect sent. The ``openai_judge`` fixture (``conftest.py``) wraps it.
    """
    judge = LocalOpenAIJudge(
        base_url="", reply=reply, reasoning=reasoning, finish_reason=finish_reason
    )

    # BaseHTTPRequestHandler logs each request to stderr, which pytest captures.
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            judge.paths.append(self.path)
            judge.authorizations.append(self.headers.get("Authorization"))
            judge.requests.append(body)
            reply = judge.reply if isinstance(judge.reply, str) else judge.reply(body)
            message: dict[str, Any] = {"role": "assistant", "content": reply}
            if judge.reasoning is not None:
                message["reasoning_content"] = judge.reasoning
            payload = json.dumps(
                {
                    "id": "local-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body.get("model", "judge"),
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": judge.finish_reason,
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
