"""Capture the frozen legacy expectations in tests/golden/.

Each module here defines ``SOURCES`` (the legacy symbols it reads) and
``capture() -> dict`` (JSON data). ``python -m scripts.parity.removal_baseline
[name ...]`` runs them in the frozen pre-removal checkout and writes
``tests/golden/<name>.json`` with the checkout's commit as provenance. They need
the legacy engine and fail with a clear message without it; the tests read only
the JSON.
"""
