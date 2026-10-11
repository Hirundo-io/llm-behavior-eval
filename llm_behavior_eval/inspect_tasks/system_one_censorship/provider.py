"""Inspect model provider for System One (``/v1/systemone``) decision servers.

Use it as ``--model systemone/<name>`` with ``--model-base-url`` pointing at a
server that speaks the TypeSafe Jev System One wire format, for example
Open-Jev's ``python -m jev.server``. The decision server is not generative:
the solver sends the System One request as JSON in the last user message and
receives the JSON response as the completion.
"""

from __future__ import annotations

import json
import os
import time
from typing import TYPE_CHECKING, Any

import httpx
from inspect_ai.model import (
    ChatMessage,
    ChatMessageUser,
    GenerateConfig,
    ModelAPI,
    ModelCall,
    ModelOutput,
)

if TYPE_CHECKING:
    from inspect_ai.tool import ToolChoice, ToolInfo

SYSTEMONE_BASE_URL_VAR = "SYSTEMONE_BASE_URL"
SYSTEMONE_API_KEY_VAR = "SYSTEMONE_API_KEY"
DEFAULT_BASE_URL = "http://127.0.0.1:8791"
SYSTEMONE_PATH = "/v1/systemone"
DEFAULT_CONFIG = GenerateConfig()


def systemone_endpoint(base_url: str, path: str = SYSTEMONE_PATH) -> str:
    """Return the System One endpoint for a server base URL.

    ``path`` is appended unless the URL already ends with it; pass an empty
    ``path`` to use ``base_url`` verbatim (e.g. a hosted ``.../ai/run/<model>``).
    """
    base_url = base_url.rstrip("/")
    if not path or base_url.endswith(path):
        return base_url
    return base_url + path


def unwrap_response(body: Any) -> dict[str, Any]:
    """Return the System One response, unwrapping a ``{"result": ...}`` envelope."""
    if not isinstance(body, dict):
        raise ValueError("System One response is not a JSON object")
    result = body.get("result")
    if "answers" not in body and isinstance(result, dict) and "answers" in result:
        return result
    return body


class SystemOneAPI(ModelAPI):
    """Send System One requests to a decision server over HTTP.

    Model args (``-M``):
        request_model: Value of the request's ``model`` field (defaults to
            the model name after ``systemone/``).
        path: Path appended to the base URL (default ``/v1/systemone``); set
            it to an empty string to post to the base URL as given.
        timeout: Per-request timeout in seconds (default 300).
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = DEFAULT_CONFIG,
        request_model: str | None = None,
        path: str = SYSTEMONE_PATH,
        timeout: float = 300.0,
    ) -> None:
        super().__init__(
            model_name=model_name,
            base_url=base_url
            or os.environ.get(SYSTEMONE_BASE_URL_VAR, DEFAULT_BASE_URL),
            api_key=api_key,
            api_key_vars=[SYSTEMONE_API_KEY_VAR],
            config=config,
        )
        if self.api_key is None:
            self.api_key = os.environ.get(SYSTEMONE_API_KEY_VAR)
        self.request_model = request_model or model_name
        self.endpoint = systemone_endpoint(self.base_url or DEFAULT_BASE_URL, path)
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(headers=headers, timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def max_connections(self) -> int:
        # Open-Jev serializes inference behind a lock; extra connections only queue.
        return 4

    def should_retry(self, ex: Exception) -> bool:
        if isinstance(ex, httpx.HTTPStatusError):
            status = ex.response.status_code
            return status == 429 or status >= 500
        return isinstance(ex, httpx.TransportError)

    async def generate(
        self,
        input: list[ChatMessage],  # noqa: A002 (ModelAPI.generate signature)
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> tuple[ModelOutput, ModelCall]:
        if tools:
            raise ValueError("System One decision servers do not support tools")
        request = decode_request(input)
        payload = {"model": self.request_model, **request}
        start = time.monotonic()
        response = await self._http().post(self.endpoint, json=payload)
        if response.status_code == 422:
            # The server rejected the request itself; retrying cannot help.
            raise ValueError(f"System One request rejected: {response.text}")
        response.raise_for_status()
        result = unwrap_response(response.json())
        call = ModelCall.create(
            request=payload, response=result, time=time.monotonic() - start
        )
        output = ModelOutput.from_content(
            model=self.model_name, content=json.dumps(result, ensure_ascii=False)
        )
        return output, call


def encode_request(request: dict[str, Any]) -> ChatMessageUser:
    """Wrap a System One request body as the user message the provider reads."""
    return ChatMessageUser(content=json.dumps(request, ensure_ascii=False))


def decode_request(messages: list[ChatMessage]) -> dict[str, Any]:
    """Recover the System One request body from the last user message."""
    users = [message for message in messages if message.role == "user"]
    if not users:
        raise ValueError("System One generate() needs a user message")
    request = json.loads(users[-1].text)
    if not isinstance(request, dict) or not {"state", "questions"} <= request.keys():
        raise ValueError("the last user message is not a System One request")
    return request
