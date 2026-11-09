"""Opt-in, bounded MLflow reporting for finalized native Inspect runs.

The hook is registered by ``llm_behavior_eval._registry`` (Inspect's entry
point), so it is available to every ``inspect eval`` in an environment where
this package is installed. It does nothing, and never imports MLflow, unless
``LLM_BEHAVIOR_EVAL_MLFLOW_ENABLED`` is set. It reports aggregate metrics from
``EvalLog.results`` and uploads the finalized ``.eval`` files; it reads and
sets no inference settings. See ``MLFLOW_INTEGRATION.md``.
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import sys
from typing import Any
from urllib.parse import urlparse

from inspect_ai.hooks import Hooks, RunEnd, RunStart, hooks

logger = logging.getLogger(__name__)
PREFIX = "LLM_BEHAVIOR_EVAL_MLFLOW_"


def adapter_run_id(reference: str, tracking_uri: str) -> str | None:
    """Resolve reporting identity only; never download or load an adapter."""
    ref, tracking = urlparse(reference), urlparse(tracking_uri)
    parts = ref.path.strip("/").split("/")
    if ref.scheme == "mlflow":
        return ref.netloc or None
    if ref.scheme == "runs":
        return ref.netloc or parts[0] or None
    if ref.scheme in {"http", "https"} and (ref.scheme, ref.netloc) == (
        tracking.scheme,
        tracking.netloc,
    ):
        if len(parts) >= 2 and parts[0] == "runs":
            return parts[1]
        if len(parts) == 1:
            return parts[0] or None
    return None


def metric_step(reference: str) -> int:
    """Infer a checkpoint step from an exact path segment, defaulting to zero."""
    for segment in urlparse(reference).path.split("/"):
        match = re.fullmatch(r"checkpoint-(\d+)", segment)
        if match:
            return int(match[1])
    return 0


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _component(value: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_.-]", "_", value)[:32] or "unnamed"
    return clean if clean == value else f"{clean}-{_digest(value)}"


def report_payload(data: RunEnd, config: dict[str, Any]) -> dict[str, Any]:
    """Copy aggregate values without scaling; exclude raw samples and parameters."""
    tasks = []
    for log in data.logs:
        spec = log.eval
        identity = _digest([spec.dataset.model_dump(), spec.task_args, spec.model])
        prefix = f"{_component(spec.task)}/{identity}"
        metrics: dict[str, float | int] = {}
        if log.results:
            metrics[f"{prefix}/num_samples_evaluated"] = log.results.completed_samples
            for score in log.results.scores:
                scorer = _component(score.name)
                reducer = _component(score.reducer or "default")
                for name, metric in score.metrics.items():
                    if math.isfinite(metric.value):
                        metrics[f"{prefix}/{scorer}/{reducer}/{_component(name)}"] = (
                            metric.value
                        )
        tasks.append(
            {
                "metrics": metrics,
                "location": log.location,
                "artifact_id": _component(spec.eval_id),
                "failed": log.status != "success",
            }
        )
    return {
        "config": config,
        "tasks": tasks,
        "failed": data.exception is not None,
        "inspect_run_id": data.run_id,
    }


async def _send(payload: dict[str, Any], timeout: float) -> None:
    # A killable subprocess bounds even SDK shutdown/transport stalls. No shell,
    # credentials in argv, MLflow async queue, or executor shutdown joins.
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "llm_behavior_eval.hooks._mlflow_worker",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await asyncio.wait_for(
            process.communicate(json.dumps(payload).encode()), timeout
        )
        if process.returncode:
            logger.warning(
                "MLflow reporting failed; Inspect logs remain authoritative."
            )
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


@hooks(name="behavior_mlflow", description="Opt-in aggregate MLflow reporting")
class MlflowHook(Hooks):
    """One reporting operation per Inspect run, after log finalization."""

    def __init__(self) -> None:
        self._runs: dict[str, dict[str, Any]] = {}

    def enabled(self) -> bool:
        """Opt-in: report only when ``LLM_BEHAVIOR_EVAL_MLFLOW_ENABLED`` is set."""
        return os.getenv(PREFIX + "ENABLED", "").lower() in {"1", "true", "yes"}

    async def on_run_start(self, data: RunStart) -> None:
        if not self.enabled():
            return
        try:
            import mlflow

            uri = os.getenv("MLFLOW_TRACKING_URI") or mlflow.get_tracking_uri()
            active = mlflow.active_run()
            reference = os.getenv(PREFIX + "ADAPTER_REF", "")
            run_id = (
                os.getenv(PREFIX + "RUN_ID")
                or (active.info.run_id if active else None)
                or adapter_run_id(reference, uri)
            )
            step = int(os.getenv(PREFIX + "STEP", str(metric_step(reference))))
            timeout = float(os.getenv(PREFIX + "TIMEOUT", "30"))
            if step < 0 or not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("Invalid reporting configuration")
            self._runs[data.run_id] = {
                "tracking_uri": uri,
                "run_id": run_id,
                "step": step,
                "timeout": timeout,
                "experiment": os.getenv("MLFLOW_EXPERIMENT_NAME", "llm-behavior-eval"),
                "run_name": os.getenv(PREFIX + "RUN_NAME", "inspect-evaluation"),
                "subfolder": os.getenv(PREFIX + "ARTIFACT_SUBFOLDER", ""),
            }
        except Exception:
            logger.warning("MLflow reporting unavailable; evaluation continues.")

    async def on_run_end(self, data: RunEnd) -> None:
        config = self._runs.pop(data.run_id, None)
        if config is None:
            return
        try:
            await _send(report_payload(data, config), config["timeout"])
        except Exception:
            logger.warning(
                "MLflow reporting failed or timed out; Inspect logs remain authoritative."
            )
