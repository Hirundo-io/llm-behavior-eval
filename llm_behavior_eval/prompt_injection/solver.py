"""Benchmark-owned solver. It sets no GenerateConfig for the model under test.

The per-row system prompt and user message are already in each sample's input.
"""

from inspect_ai.solver import Solver, generate


def solver() -> list[Solver]:
    return [generate()]
