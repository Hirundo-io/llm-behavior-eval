"""Inspect AI entry point: registers this package's tasks and model providers."""

from inspect_ai.model import ModelAPI, modelapi

from .system_one_censorship.scenario_task import system_one_scenarios
from .system_one_censorship.task import ccpc_system_one


@modelapi(name="systemone")
def systemone() -> type[ModelAPI]:
    from .system_one_censorship.provider import SystemOneAPI

    return SystemOneAPI


__all__ = ["ccpc_system_one", "system_one_scenarios", "systemone"]
