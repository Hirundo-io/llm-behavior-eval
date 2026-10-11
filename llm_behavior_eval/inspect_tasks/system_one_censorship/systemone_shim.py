"""Thin ``/v1/systemone`` shim for decision models that ship without one.

Run it as a script so it needs nothing from this package (it is stdlib-only
apart from the chosen backend's own dependencies)::

    # Cloudflare Clef / Clef-Flash: wraps the model card's ``systemone()``
    python systemone_shim.py clef --model-path <snapshot dir> --port 30000

    # autotrust JEV-*: translates to the System 1 read-out of a vLLM server
    # started with ``--enable-lora --lora-modules jev-decision=<dir>/adapter_vllm
    # --logprobs-mode processed_logprobs``
    python systemone_shim.py jev-vllm --model-path <snapshot dir> \\
        --upstream http://127.0.0.1:8003 --port 30001

The JEV translation mirrors the model card and ``vl/serve_decide.py``: the
bare ``[kind]/[state]/[question]/[options]/[decision]:`` template, a one-token
read-out restricted to the verbalizer ids, plus the head bias, divided by the
per-kind temperature. JEV's native ``score`` kind is a fixed 0-5 scale with no
level descriptions, so a System One ``score`` question with described levels
is asked as a ``choice`` over those descriptions (in order) and answered with
the expected level index.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

JEV_LORA = "jev-decision"
JEV_CHOICE_LABELS = "ABCDEFGHIJKLMNOP"
# Read the full distribution: override generation defaults that would truncate it.
JEV_READ = {
    "max_tokens": 1,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "add_special_tokens": False,
    "return_tokens_as_token_ids": True,
}


def softmax(values: list[float]) -> list[float]:
    peak = max(values)
    weights = [math.exp(value - peak) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def jev_state_text(state: Any) -> str:
    """Render a System One state the way JEV's ``_parts`` renders text states."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def jev_plan(definition: dict[str, Any]) -> tuple[str, list[str]]:
    """Return the JEV kind and option lines for one System One question."""
    kind = definition["type"]
    criteria = definition.get("criteria")
    if kind == "noul":
        return "noul", ["false", "true"]
    if kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 16:
            raise ValueError("score questions need 2 to 16 described levels")
        options = [
            level if isinstance(level, str) else json.dumps(level) for level in criteria
        ]
        return "choice", options
    if kind == "choice":
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 16:
            raise ValueError("choice questions need 2 to 16 options")
        return "choice", [
            name if description is None else f"{name}: {description}"
            for name, description in criteria.items()
        ]
    raise ValueError(f"unsupported question type {kind!r}")


def jev_prompt(kind: str, state_text: str, question: str, options: list[str]) -> str:
    """Render JEV's bare System 1 template."""
    if kind == "choice":
        lines = [
            f"{JEV_CHOICE_LABELS[i]}) {option}" for i, option in enumerate(options)
        ]
    else:
        lines = options
    return (
        f"[kind] {kind}\n[state] {state_text}\n[question] {question}\n[options]\n"
        + "\n".join(lines)
        + "\n[decision]:"
    )


def jev_answer(
    definition: dict[str, Any], kind: str, probabilities: list[float]
) -> dict[str, Any]:
    """Format JEV probabilities as a System One answer."""
    if definition["type"] == "noul":
        return {"type": "noul", "noul": probabilities[1]}
    if definition["type"] == "score":
        expected = sum(index * p for index, p in enumerate(probabilities))
        return {
            "type": "score",
            "score": expected,
            "probabilities": {str(i): p for i, p in enumerate(probabilities)},
            "legend": {str(i): level for i, level in enumerate(definition["criteria"])},
        }
    names = list(definition["criteria"])
    best = max(range(len(names)), key=probabilities.__getitem__)
    return {
        "type": "choice",
        "choice": names[best],
        "confidence": probabilities[best],
        "probabilities": dict(zip(names, probabilities, strict=True)),
    }


class JevVllmBackend:
    """Ask a JEV System 1 LoRA served by vLLM, one read-out per question."""

    def __init__(self, model_path: str, upstream: str) -> None:
        if not upstream.startswith(("http://", "https://")):
            raise ValueError("upstream must be an http(s) URL")
        root = Path(model_path)
        head = json.loads((root / "adapter_vllm" / "decision_head.json").read_text())
        self.ranges = head["slots"]["ranges"]
        self.verbalizer_ids = head["verbalizer_ids"]
        self.bias = head["bias"]
        self.temperatures = json.loads((root / "calibration.json").read_text())[
            "per_kind"
        ]
        self.url = upstream.rstrip("/") + "/v1/completions"
        self.name = root.name

    def readout(self, kind: str, prompt: str, n_options: int) -> list[float]:
        start, end = self.ranges[kind]
        if n_options > end - start:
            raise ValueError(f"{kind} supports at most {end - start} options")
        ids = self.verbalizer_ids[start : start + n_options]
        bias = self.bias[start : start + n_options]
        body = {
            "model": JEV_LORA,
            "prompt": prompt,
            "logprobs": len(ids),
            "allowed_token_ids": ids,
            **JEV_READ,
        }
        request = urllib.request.Request(  # noqa: S310 (scheme checked in __init__)
            self.url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
            result = json.loads(response.read())
        top = result["choices"][0]["logprobs"]["top_logprobs"][0]
        logprobs = {int(key.split(":")[1]): value for key, value in top.items()}
        return softmax(
            [
                (max(logprobs.get(token, -1e9), -1e9) + b) / self.temperatures[kind]
                for token, b in zip(ids, bias, strict=True)
            ]
        )

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        state_text = jev_state_text(request["state"])
        answers = {}
        for question_id, definition in request["questions"].items():
            kind, options = jev_plan(definition)
            prompt = jev_prompt(
                kind, state_text, definition.get("instructions", ""), options
            )
            answers[question_id] = jev_answer(
                definition, kind, self.readout(kind, prompt, len(options))
            )
        return {"model": request.get("model", self.name), "answers": answers}


class ClefBackend:
    """Call the ``systemone`` function shipped in a Clef model snapshot."""

    def __init__(self, model_path: str, device: str) -> None:
        sys.path.insert(0, model_path)
        from joint_schema_model import (  # type: ignore[import-not-found]
            load_release_model,
            systemone,
        )

        self.model, self.processor = load_release_model(model_path, device=device)
        self.systemone = systemone
        # One forward pass at a time; extra connections only queue.
        self.lock = threading.Lock()

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            return self.systemone(self.model, self.processor, request)


def make_handler(backend: Callable[[dict[str, Any]], dict[str, Any]]) -> type:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            if self.path == "/health":
                self._send(200, {"status": "ready"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 (http.server API)
            if self.path.rstrip("/") != "/v1/systemone":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                request = json.loads(self.rfile.read(length))
                if (
                    not isinstance(request, dict)
                    or not {"state", "questions"} <= request.keys()
                ):
                    raise ValueError("request needs state and questions")
            except ValueError as error:
                self._send(422, {"error": str(error)})
                return
            try:
                self._send(200, backend(request))
            except ValueError as error:
                self._send(422, {"error": str(error)})
            except Exception as error:  # noqa: BLE001 (surface backend failures as 500)
                self._send(500, {"error": f"{type(error).__name__}: {error}"})

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("backend", choices=("clef", "jev-vllm"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--upstream", default="http://127.0.0.1:8003")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    args = parser.parse_args()
    if args.backend == "clef":
        backend: Callable[[dict[str, Any]], dict[str, Any]] = ClefBackend(
            args.model_path, args.device
        )
    else:
        backend = JevVllmBackend(args.model_path, args.upstream)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(backend))
    print(
        f"systemone shim ({args.backend}) on http://{args.host}:{args.port}", flush=True
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
