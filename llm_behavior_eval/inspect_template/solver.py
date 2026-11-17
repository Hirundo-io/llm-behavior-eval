"""Benchmark-owned solver. It sets no GenerateConfig for the model under test."""

from inspect_ai.solver import Solver, generate, system_message

from .prompts import SYSTEM_MESSAGE


def solver() -> list[Solver]:
    return [system_message(SYSTEM_MESSAGE), generate()]
