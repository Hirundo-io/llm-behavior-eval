from __future__ import annotations

import csv
import gc
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Literal
from urllib import error as urllib_error
from urllib import request as urllib_request

import pytest

pytest.importorskip("torch")
import torch

from llm_behavior_eval.evaluate import main as run_evaluation

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

DEFAULT_BEHAVIORS = "bias:gender,unqover:bias:gender,prompt-injection"
DEFAULT_DATASET_COUNT = 3
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_JUDGE = "Qwen/Qwen2.5-1.5B-Instruct"
EngineName = Literal["transformers", "vllm", "api"]


def _reserve_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _cleanup_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _tail_lines(text: str, max_lines: int = 80) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def _wait_for_server_ready(
    api_base: str,
    process: subprocess.Popen[str],
    timeout_s: int,
) -> None:
    deadline = time.monotonic() + timeout_s
    models_endpoint = f"{api_base}/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM API server exited early with code {process.returncode}."
            )
        try:
            with urllib_request.urlopen(models_endpoint, timeout=2.0) as response:
                if response.status == 200:
                    payload = json.loads(response.read().decode("utf-8"))
                    if payload.get("data"):
                        return
        except (OSError, TimeoutError, ValueError, urllib_error.URLError):
            pass
        time.sleep(1.0)
    raise RuntimeError(
        f"Timed out waiting for vLLM API server readiness at {models_endpoint}."
    )


@contextmanager
def _local_vllm_api_server(
    model: str,
    gpu_memory_utilization: float,
    startup_timeout_s: int,
) -> Iterator[str]:
    vllm_executable = shutil.which("vllm")
    if not vllm_executable:
        raise RuntimeError("Could not find `vllm` executable on PATH.")

    port = _reserve_open_port()
    api_base = f"http://127.0.0.1:{port}/v1"
    command = [
        vllm_executable,
        "serve",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    process: subprocess.Popen[str] | None = None
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as server_log:
        try:
            process = subprocess.Popen(
                command,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                _wait_for_server_ready(api_base, process, startup_timeout_s)
            except RuntimeError as exc:
                server_log.seek(0)
                log_tail = _tail_lines(server_log.read())
                raise RuntimeError(
                    "vLLM API server failed to start. "
                    f"Command: {' '.join(command)}\n"
                    f"Log tail:\n{log_tail}"
                ) from exc
            yield api_base
        finally:
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=15)
            try:
                server_log.seek(0)
            except OSError:
                pass


def _read_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_metrics(output_dir: Path, model: str) -> dict[str, tuple[str, float]]:
    model_slug = model.split("/")[-1]
    metrics_files = sorted((output_dir / model_slug).rglob("metrics.csv"))
    assert metrics_files, (
        f"No metrics.csv files found under {(output_dir / model_slug)!s}"
    )
    metrics: dict[str, tuple[str, float]] = {}
    for metrics_file in metrics_files:
        with metrics_file.open(newline="") as file_handle:
            rows = list(csv.DictReader(file_handle))
        assert rows, f"Expected non-empty metrics file: {metrics_file}"
        first_row = rows[0]
        metric_key = "Error (%)" if "Error (%)" in first_row else "Accuracy (%)"
        metrics[metrics_file.parent.name] = (metric_key, float(first_row[metric_key]))
    return metrics


def _read_effective_tolerance(
    max_samples: int, env_name: str, default: str = "5.0"
) -> float:
    tolerance = float(os.getenv(env_name, default))
    granularity_tolerance = 100.0 / max_samples
    return max(tolerance, granularity_tolerance)


def _run_gpu_regression_eval(
    *,
    output_dir: Path,
    model: str,
    judge_model: str,
    model_engine: EngineName,
    judge_engine: EngineName,
    behavior: str,
    max_samples: int,
    max_answer_tokens: int,
    max_judge_tokens: int,
    trust_remote_code: bool,
    model_token: str | None,
    judge_token: str | None,
    vllm_gpu_memory_utilization: float,
) -> None:
    run_evaluation(
        model=model,
        behavior=behavior,
        output_dir=str(output_dir),
        model_token=model_token,
        judge_token=judge_token,
        judge_model=judge_model,
        model_engine=model_engine,
        judge_engine=judge_engine,
        replace_existing_output=True,
        max_samples=max_samples,
        batch_size=1,
        judge_batch_size=1,
        sample=False,
        sample_judge=False,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        seed=7,
        max_answer_tokens=max_answer_tokens,
        max_judge_tokens=max_judge_tokens,
        use_4bit=False,
        use_4bit_judge=False,
        trust_remote_code=trust_remote_code,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_enforce_eager=True,
    )
    _cleanup_gpu_memory()


def _assert_metric_parity(
    *,
    baseline_output: Path,
    baseline_model: str,
    comparison_output: Path,
    comparison_model: str,
    tolerance: float,
    label: str,
) -> None:
    baseline_metrics = _read_metrics(baseline_output, baseline_model)
    comparison_metrics = _read_metrics(comparison_output, comparison_model)
    assert len(baseline_metrics) == DEFAULT_DATASET_COUNT
    assert set(baseline_metrics) == set(comparison_metrics)
    for dataset_name, (baseline_metric_key, baseline_score) in baseline_metrics.items():
        comparison_metric_key, comparison_score = comparison_metrics[dataset_name]
        assert comparison_metric_key == baseline_metric_key
        assert abs(comparison_score - baseline_score) <= tolerance, (
            f"{label} exceeded tolerance for {dataset_name}: "
            f"{baseline_score:.3f} vs {comparison_score:.3f} ({baseline_metric_key}), "
            f"tolerance={tolerance:.3f}"
        )


@pytest.mark.gpu_regression
@pytest.mark.parametrize("engine", ["transformers", "vllm"])
def test_gpu_local_local_regression(
    engine: Literal["transformers", "vllm"],
    tmp_path: Path,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU regression tests.")

    if engine == "vllm" and importlib.util.find_spec("vllm") is None:
        pytest.skip("vLLM is not installed; install the 'vllm' extra to run this case.")

    model = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MODEL", DEFAULT_MODEL)
    judge_model = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_JUDGE", DEFAULT_JUDGE)

    behavior = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_BEHAVIOR", DEFAULT_BEHAVIORS)
    max_samples = int(os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MAX_SAMPLES", "4"))
    max_answer_tokens = int(
        os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MAX_ANSWER_TOKENS", "32")
    )
    max_judge_tokens = int(
        os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MAX_JUDGE_TOKENS", "16")
    )
    trust_remote_code = _read_bool_env(
        "LLM_BEHAVIOR_EVAL_GPU_REGRESSION_TRUST_REMOTE_CODE", False
    )
    vllm_gpu_memory_utilization = float(
        os.getenv(
            "LLM_BEHAVIOR_EVAL_GPU_REGRESSION_VLLM_GPU_MEMORY_UTILIZATION",
            "0.9",
        )
    )
    model_token = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MODEL_TOKEN")
    judge_token = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_JUDGE_TOKEN")
    output_dir = tmp_path / f"{engine}-local-local-regression"

    _run_gpu_regression_eval(
        output_dir=output_dir,
        model=model,
        judge_model=judge_model,
        model_engine=engine,
        judge_engine=engine,
        behavior=behavior,
        max_samples=max_samples,
        max_answer_tokens=max_answer_tokens,
        max_judge_tokens=max_judge_tokens,
        trust_remote_code=trust_remote_code,
        model_token=model_token,
        judge_token=judge_token,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
    )

    model_slug = model.split("/")[-1]
    metrics_files = sorted((output_dir / model_slug).rglob("metrics.csv"))
    assert len(metrics_files) == DEFAULT_DATASET_COUNT

    for metrics_file in metrics_files:
        with metrics_file.open(newline="") as file_handle:
            rows = list(csv.DictReader(file_handle))
        assert rows, f"Expected non-empty metrics file: {metrics_file}"
        first_row = rows[0]
        metric_key = "Error (%)" if "Error (%)" in first_row else "Accuracy (%)"
        score = float(first_row[metric_key])
        assert 0.0 <= score <= 100.0


@pytest.mark.gpu_regression
@pytest.mark.parametrize("api_target", ["model", "judge"])
def test_gpu_vllm_and_api_vllm_server_regression(
    api_target: Literal["model", "judge"],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU regression tests.")

    if importlib.util.find_spec("vllm") is None:
        pytest.skip("vLLM is not installed; install the 'vllm' extra to run this case.")
    if importlib.util.find_spec("litellm") is None:
        pytest.skip(
            "LiteLLM is not installed; install an API extra (e.g. api-openai or api-all)."
        )

    api_base = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_API_BASE") or os.getenv(
        "OPENAI_API_BASE"
    )
    model = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MODEL", DEFAULT_MODEL)
    judge_model = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_JUDGE", DEFAULT_JUDGE)
    api_model = os.getenv(
        "LLM_BEHAVIOR_EVAL_GPU_REGRESSION_API_MODEL",
        f"openai/{model}",
    )
    api_judge_model = os.getenv(
        "LLM_BEHAVIOR_EVAL_GPU_REGRESSION_API_JUDGE",
        f"openai/{judge_model}",
    )
    behavior = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_BEHAVIOR", DEFAULT_BEHAVIORS)
    max_samples = int(os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MAX_SAMPLES", "4"))
    max_answer_tokens = int(
        os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MAX_ANSWER_TOKENS", "32")
    )
    max_judge_tokens = int(
        os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MAX_JUDGE_TOKENS", "16")
    )
    trust_remote_code = _read_bool_env(
        "LLM_BEHAVIOR_EVAL_GPU_REGRESSION_TRUST_REMOTE_CODE", False
    )
    vllm_gpu_memory_utilization = float(
        os.getenv(
            "LLM_BEHAVIOR_EVAL_GPU_REGRESSION_VLLM_GPU_MEMORY_UTILIZATION",
            "0.9",
        )
    )
    api_server_gpu_memory_utilization = float(
        os.getenv(
            "LLM_BEHAVIOR_EVAL_GPU_REGRESSION_API_SERVER_GPU_MEMORY_UTILIZATION",
            str(vllm_gpu_memory_utilization),
        )
    )
    api_server_startup_timeout_s = int(
        os.getenv(
            "LLM_BEHAVIOR_EVAL_GPU_REGRESSION_API_SERVER_STARTUP_TIMEOUT_S",
            "240",
        )
    )
    model_token = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MODEL_TOKEN")
    judge_token = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_JUDGE_TOKEN")
    effective_tolerance = _read_effective_tolerance(
        max_samples,
        "LLM_BEHAVIOR_EVAL_GPU_REGRESSION_API_METRIC_TOLERANCE",
    )

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_OPENAI_API_KEY", "dummy"),
    )
    monkeypatch.setenv("LLM_EVAL_API_CONCURRENCY", "1")

    baseline_output = tmp_path / f"vllm-local-local-{api_target}"
    _run_gpu_regression_eval(
        output_dir=baseline_output,
        model=model,
        judge_model=judge_model,
        model_engine="vllm",
        judge_engine="vllm",
        behavior=behavior,
        max_samples=max_samples,
        max_answer_tokens=max_answer_tokens,
        max_judge_tokens=max_judge_tokens,
        trust_remote_code=trust_remote_code,
        model_token=model_token,
        judge_token=judge_token,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
    )

    comparison_output = tmp_path / f"vllm-api-{api_target}"
    comparison_model = api_model if api_target == "model" else model
    comparison_judge = api_judge_model if api_target == "judge" else judge_model
    comparison_model_engine: Literal["vllm", "api"] = (
        "api" if api_target == "model" else "vllm"
    )
    comparison_judge_engine: Literal["vllm", "api"] = (
        "api" if api_target == "judge" else "vllm"
    )

    if api_base:
        server_context = nullcontext(api_base)
    else:
        default_server_model = model if api_target == "model" else judge_model
        server_model = os.getenv(
            "LLM_BEHAVIOR_EVAL_GPU_REGRESSION_API_SERVER_MODEL",
            default_server_model,
        )
        server_context = _local_vllm_api_server(
            model=server_model,
            gpu_memory_utilization=api_server_gpu_memory_utilization,
            startup_timeout_s=api_server_startup_timeout_s,
        )

    with server_context as resolved_api_base:
        monkeypatch.setenv("OPENAI_API_BASE", resolved_api_base)
        _run_gpu_regression_eval(
            output_dir=comparison_output,
            model=comparison_model,
            judge_model=comparison_judge,
            model_engine=comparison_model_engine,
            judge_engine=comparison_judge_engine,
            behavior=behavior,
            max_samples=max_samples,
            max_answer_tokens=max_answer_tokens,
            max_judge_tokens=max_judge_tokens,
            trust_remote_code=trust_remote_code,
            model_token=model_token,
            judge_token=judge_token,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        )
    _assert_metric_parity(
        baseline_output=baseline_output,
        baseline_model=model,
        comparison_output=comparison_output,
        comparison_model=comparison_model,
        tolerance=effective_tolerance,
        label=f"{api_target} API regression",
    )


@pytest.mark.gpu_regression
def test_gpu_vllm_and_api_vllm_server_remote_remote_regression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU regression tests.")

    if importlib.util.find_spec("vllm") is None:
        pytest.skip("vLLM is not installed; install the 'vllm' extra to run this case.")
    if importlib.util.find_spec("litellm") is None:
        pytest.skip(
            "LiteLLM is not installed; install an API extra (e.g. api-openai or api-all)."
        )

    api_base = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_API_BASE") or os.getenv(
        "OPENAI_API_BASE"
    )
    model = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MODEL", DEFAULT_MODEL)
    judge_model = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_JUDGE", DEFAULT_JUDGE)
    api_model = os.getenv(
        "LLM_BEHAVIOR_EVAL_GPU_REGRESSION_API_MODEL",
        f"openai/{model}",
    )
    api_judge_model = os.getenv(
        "LLM_BEHAVIOR_EVAL_GPU_REGRESSION_API_JUDGE",
        f"openai/{judge_model}",
    )
    if not api_base and api_model != api_judge_model:
        pytest.skip(
            "Auto-started local vLLM API server only serves one model. "
            "Set LLM_BEHAVIOR_EVAL_GPU_REGRESSION_API_BASE for distinct API model/judge."
        )

    behavior = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_BEHAVIOR", DEFAULT_BEHAVIORS)
    max_samples = int(os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MAX_SAMPLES", "4"))
    max_answer_tokens = int(
        os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MAX_ANSWER_TOKENS", "32")
    )
    max_judge_tokens = int(
        os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MAX_JUDGE_TOKENS", "16")
    )
    trust_remote_code = _read_bool_env(
        "LLM_BEHAVIOR_EVAL_GPU_REGRESSION_TRUST_REMOTE_CODE", False
    )
    vllm_gpu_memory_utilization = float(
        os.getenv(
            "LLM_BEHAVIOR_EVAL_GPU_REGRESSION_VLLM_GPU_MEMORY_UTILIZATION",
            "0.9",
        )
    )
    api_server_gpu_memory_utilization = float(
        os.getenv(
            "LLM_BEHAVIOR_EVAL_GPU_REGRESSION_API_SERVER_GPU_MEMORY_UTILIZATION",
            str(vllm_gpu_memory_utilization),
        )
    )
    api_server_startup_timeout_s = int(
        os.getenv(
            "LLM_BEHAVIOR_EVAL_GPU_REGRESSION_API_SERVER_STARTUP_TIMEOUT_S",
            "240",
        )
    )
    model_token = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MODEL_TOKEN")
    judge_token = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_JUDGE_TOKEN")
    effective_tolerance = _read_effective_tolerance(
        max_samples,
        "LLM_BEHAVIOR_EVAL_GPU_REGRESSION_API_REMOTE_REMOTE_METRIC_TOLERANCE",
        os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_API_METRIC_TOLERANCE", "5.0"),
    )

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_OPENAI_API_KEY", "dummy"),
    )
    monkeypatch.setenv("LLM_EVAL_API_CONCURRENCY", "1")

    baseline_output = tmp_path / "vllm-local-local-remote-remote"
    _run_gpu_regression_eval(
        output_dir=baseline_output,
        model=model,
        judge_model=judge_model,
        model_engine="vllm",
        judge_engine="vllm",
        behavior=behavior,
        max_samples=max_samples,
        max_answer_tokens=max_answer_tokens,
        max_judge_tokens=max_judge_tokens,
        trust_remote_code=trust_remote_code,
        model_token=model_token,
        judge_token=judge_token,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
    )

    comparison_output = tmp_path / "api-remote-remote"
    if api_base:
        server_context = nullcontext(api_base)
    else:
        server_model = os.getenv(
            "LLM_BEHAVIOR_EVAL_GPU_REGRESSION_API_SERVER_MODEL",
            model,
        )
        server_context = _local_vllm_api_server(
            model=server_model,
            gpu_memory_utilization=api_server_gpu_memory_utilization,
            startup_timeout_s=api_server_startup_timeout_s,
        )

    with server_context as resolved_api_base:
        monkeypatch.setenv("OPENAI_API_BASE", resolved_api_base)
        _run_gpu_regression_eval(
            output_dir=comparison_output,
            model=api_model,
            judge_model=api_judge_model,
            model_engine="api",
            judge_engine="api",
            behavior=behavior,
            max_samples=max_samples,
            max_answer_tokens=max_answer_tokens,
            max_judge_tokens=max_judge_tokens,
            trust_remote_code=trust_remote_code,
            model_token=model_token,
            judge_token=judge_token,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        )

    _assert_metric_parity(
        baseline_output=baseline_output,
        baseline_model=model,
        comparison_output=comparison_output,
        comparison_model=api_model,
        tolerance=effective_tolerance,
        label="remote-remote API regression",
    )


@pytest.mark.gpu_regression
def test_gpu_local_mixed_engine_parity_regression(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU regression tests.")

    if importlib.util.find_spec("vllm") is None:
        pytest.skip("vLLM is not installed; install the 'vllm' extra to run this case.")

    model = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MODEL", DEFAULT_MODEL)
    judge_model = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_JUDGE", DEFAULT_JUDGE)
    behavior = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_BEHAVIOR", DEFAULT_BEHAVIORS)
    max_samples = int(os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MAX_SAMPLES", "4"))
    max_answer_tokens = int(
        os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MAX_ANSWER_TOKENS", "32")
    )
    max_judge_tokens = int(
        os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MAX_JUDGE_TOKENS", "16")
    )
    trust_remote_code = _read_bool_env(
        "LLM_BEHAVIOR_EVAL_GPU_REGRESSION_TRUST_REMOTE_CODE", False
    )
    vllm_gpu_memory_utilization = float(
        os.getenv(
            "LLM_BEHAVIOR_EVAL_GPU_REGRESSION_VLLM_GPU_MEMORY_UTILIZATION",
            "0.9",
        )
    )
    model_token = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_MODEL_TOKEN")
    judge_token = os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_JUDGE_TOKEN")
    effective_tolerance = _read_effective_tolerance(
        max_samples,
        "LLM_BEHAVIOR_EVAL_GPU_REGRESSION_LOCAL_MIXED_METRIC_TOLERANCE",
        os.getenv("LLM_BEHAVIOR_EVAL_GPU_REGRESSION_API_METRIC_TOLERANCE", "5.0"),
    )

    baseline_output = tmp_path / "vllm-local-local-mixed-baseline"
    _run_gpu_regression_eval(
        output_dir=baseline_output,
        model=model,
        judge_model=judge_model,
        model_engine="vllm",
        judge_engine="vllm",
        behavior=behavior,
        max_samples=max_samples,
        max_answer_tokens=max_answer_tokens,
        max_judge_tokens=max_judge_tokens,
        trust_remote_code=trust_remote_code,
        model_token=model_token,
        judge_token=judge_token,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
    )

    mixed_engine_combos: list[tuple[EngineName, EngineName]] = [
        ("transformers", "vllm"),
        ("vllm", "transformers"),
    ]
    for comparison_model_engine, comparison_judge_engine in mixed_engine_combos:
        comparison_output = (
            tmp_path
            / f"local-mixed-{comparison_model_engine}-{comparison_judge_engine}"
        )
        _run_gpu_regression_eval(
            output_dir=comparison_output,
            model=model,
            judge_model=judge_model,
            model_engine=comparison_model_engine,
            judge_engine=comparison_judge_engine,
            behavior=behavior,
            max_samples=max_samples,
            max_answer_tokens=max_answer_tokens,
            max_judge_tokens=max_judge_tokens,
            trust_remote_code=trust_remote_code,
            model_token=model_token,
            judge_token=judge_token,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        )
        _assert_metric_parity(
            baseline_output=baseline_output,
            baseline_model=model,
            comparison_output=comparison_output,
            comparison_model=model,
            tolerance=effective_tolerance,
            label=(
                "local mixed-engine regression "
                f"({comparison_model_engine}/{comparison_judge_engine})"
            ),
        )
