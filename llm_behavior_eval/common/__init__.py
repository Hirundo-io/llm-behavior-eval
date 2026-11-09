"""Benchmark-agnostic Inspect mechanics shared by self-contained benchmarks.

Only mechanics live here: dataset loading and conversion, the grader call with
answer parsing and thinking-trace stripping, generic metric helpers, parity
config loading, and the tokenizer mechanics of the opt-in legacy text
preprocessing that benchmarks may use for parity runs. Prompts, datasets,
solvers, scorers, metrics and inference settings belong to each benchmark (or
to the user), never to this package.
"""
