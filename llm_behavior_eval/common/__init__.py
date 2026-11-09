"""Benchmark-agnostic Inspect mechanics shared by self-contained benchmarks.

Only mechanics live here: dataset loading and conversion, environment
credential resolution, the grader call with answer parsing and thinking-trace
stripping, generic metric helpers and parity config loading. Prompts, datasets, solvers, scorers, metrics and inference
settings belong to each benchmark (or to the user), never to this package.
"""
