from typing import Any, Mapping


class ActiveRun:
    info: Any


def set_tracking_uri(uri: str | None) -> None: ...


def set_experiment(name: str | None) -> None: ...


def start_run(*, run_name: str | None = ...) -> ActiveRun: ...


def end_run(*, status: str | None = ...) -> None: ...


def log_params(params: Mapping[str, Any]) -> None: ...


def log_metrics(metrics: Mapping[str, float]) -> None: ...


def log_artifact(path: str) -> None: ...
