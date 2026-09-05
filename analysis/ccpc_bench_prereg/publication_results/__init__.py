"""Post-run aggregation layer for the CCPC-Bench seven-checkpoint publication.

Implements the frozen publication analysis contract (PP-1) mechanically: reads
already-produced evaluator output directories (``responses.json``,
``metrics.csv``, ``run_config.json``) pinned by an explicit manifest, and
emits the manuscript-ready CSV/JSON artifacts. This package never runs a
target model, never calls a judge, and never discovers a "latest" run
directory on its own -- every input path is supplied explicitly.

Human-readable contract and amendments: ``PUBLICATION_ANALYSIS_CONTRACT.md``.
Amendment 1 (2026-08-30): GLM4.7-Flash CCPC mechanical non-completion.
"""
