"""Comparator validity rules for publication result cells (PP-1R)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import tests._publication_results_fixtures as fx
from analysis.ccpc_bench_prereg.publication_results.aggregate import (
    build_model_summary,
    build_run_index,
)
from analysis.ccpc_bench_prereg.publication_results.cells import (
    summarize_ccpc_cell,
    summarize_refusal_cell,
)
from analysis.ccpc_bench_prereg.publication_results.load_all import load_publication_run
from analysis.ccpc_bench_prereg.publication_results.manifest_schema import (
    ManifestEntry,
    load_manifest,
)
from analysis.ccpc_bench_prereg.publication_results.raw_loader import (
    PublicationLoaderError,
    load_refusal_cells,
)


def _one_refusal_entry(
    tmp_path, rows: list[dict], dataset_id: str, subdir: str
) -> ManifestEntry:
    output_dir = tmp_path / subdir
    fx.write_refusal_run(output_dir, "org/model", dataset_id, rows)
    return ManifestEntry(
        model_key="m1",
        lineage="lineage1",
        checkpoint_label="ckpt1",
        model_repo_id="org/model",
        xstest_dir=str(output_dir) if subdir == "xstest" else None,
        orbench_dir=str(output_dir) if subdir == "orbench" else None,
    )


def test_ccpc_one_incomplete_of_500_is_invalid(tmp_path) -> None:
    dataset_path = tmp_path / "ccpc500.jsonl"
    rows = fx.default_ccpc500_dataset_rows()
    dataset_sha256, expected_rows = fx.write_ccpc_dataset(dataset_path, rows)
    all_ids = [row_id for row_id, _, _ in rows]
    ccpc_dir = tmp_path / "ccpc"
    row_statuses = [(row_id, "judged_false", False) for row_id in all_ids]
    row_statuses[0] = (all_ids[0], "model_incomplete", None)
    fx.write_ccpc_run(
        ccpc_dir,
        "org/model",
        str(dataset_path),
        dataset_sha256,
        expected_rows,
        row_statuses,
    )
    metrics = pd.read_csv(ccpc_dir / "metrics.csv")
    metrics.loc[0, "Judged False"] = 499
    metrics.loc[0, "Model Incomplete"] = 1
    metrics.to_csv(ccpc_dir / "metrics.csv", index=False)

    entry = ManifestEntry(
        model_key="m1",
        lineage="l",
        checkpoint_label="c",
        model_repo_id="org/model",
        ccpc_dir=str(ccpc_dir),
    )
    manifest_path = _write_single_entry_manifest(
        tmp_path,
        entry,
        ccpc_dir=ccpc_dir,
        include_refusal_dirs=True,
    )
    loaded = load_publication_run(load_manifest(manifest_path))
    summary = summarize_ccpc_cell(entry, loaded.ccpc_cells["m1"])
    assert not summary.valid
    assert summary.rate is None
    assert summary.denominator == 500
    assert summary.excluded_incomplete == 1


def test_xstest_full_cohort_with_exclusions_is_valid_comparator(tmp_path) -> None:
    rows = fx.refusal_rows_with_exclusions(
        250,
        200,
        safe_incomplete=5,
        safe_unparseable=2,
        unsafe_incomplete=3,
        unsafe_unknown_finish=1,
        refuse_fraction=0.2,
    )
    entry = _one_refusal_entry(tmp_path, rows, fx.XSTEST_DATASET, "xstest")
    cells = load_refusal_cells(entry, "xstest", fx.XSTEST_DATASET)
    safe_summary = summarize_refusal_cell(entry, "xstest_safe", cells["safe"])
    unsafe_summary = summarize_refusal_cell(entry, "xstest_unsafe", cells["unsafe"])

    assert safe_summary.valid
    assert safe_summary.observed_n == 250
    assert safe_summary.denominator == 250 - 5 - 2
    assert safe_summary.excluded_incomplete == 5
    assert safe_summary.excluded_judge_unparseable == 2
    assert safe_summary.rate == pytest.approx(
        safe_summary.numerator / safe_summary.denominator
    )

    assert unsafe_summary.valid
    assert unsafe_summary.observed_n == 200
    assert unsafe_summary.denominator == 200 - 3 - 1
    assert unsafe_summary.excluded_incomplete == 3
    assert unsafe_summary.excluded_unknown_finish == 1


def test_orbench_full_cohort_with_exclusions_is_valid_comparator(tmp_path) -> None:
    rows = fx.refusal_rows_with_exclusions(
        1319,
        655,
        safe_incomplete=4,
        safe_unparseable=1,
        unsafe_unparseable=2,
        refuse_fraction=0.1,
    )
    entry = _one_refusal_entry(tmp_path, rows, fx.OR_BENCH_DATASET, "orbench")
    cells = load_refusal_cells(entry, "orbench", fx.OR_BENCH_DATASET)
    hard_summary = summarize_refusal_cell(entry, "orbench_hard", cells["safe"])
    toxic_summary = summarize_refusal_cell(entry, "orbench_toxic", cells["unsafe"])

    assert hard_summary.valid
    assert hard_summary.observed_n == 1319
    assert hard_summary.denominator == 1319 - 4 - 1

    assert toxic_summary.valid
    assert toxic_summary.observed_n == 655
    assert toxic_summary.denominator == 655 - 2


def test_missing_refusal_rows_raise_at_loader(tmp_path) -> None:
    rows = fx.all_judged_true_refusal_rows(249, 200, refuse=False)
    entry = _one_refusal_entry(tmp_path, rows, fx.XSTEST_DATASET, "xstest")
    with pytest.raises(PublicationLoaderError, match="449 response rows"):
        load_refusal_cells(entry, "xstest", fx.XSTEST_DATASET)


def test_wrong_safe_unsafe_partition_invalidates_cell(tmp_path) -> None:
    rows = fx.all_judged_true_refusal_rows(260, 190, refuse=False)
    entry = _one_refusal_entry(tmp_path, rows, fx.XSTEST_DATASET, "xstest")
    cells = load_refusal_cells(entry, "xstest", fx.XSTEST_DATASET)
    safe_summary = summarize_refusal_cell(entry, "xstest_safe", cells["safe"])
    unsafe_summary = summarize_refusal_cell(entry, "xstest_unsafe", cells["unsafe"])

    assert not safe_summary.valid
    assert safe_summary.observed_n == 260
    assert safe_summary.rate is None
    assert not unsafe_summary.valid
    assert unsafe_summary.observed_n == 190


def test_zero_known_denominator_is_invalid(tmp_path) -> None:
    rows = fx.refusal_rows_with_exclusions(
        250,
        200,
        safe_incomplete=250,
        unsafe_incomplete=200,
    )
    entry = _one_refusal_entry(tmp_path, rows, fx.XSTEST_DATASET, "xstest")
    cells = load_refusal_cells(entry, "xstest", fx.XSTEST_DATASET)
    safe_summary = summarize_refusal_cell(entry, "xstest_safe", cells["safe"])
    unsafe_summary = summarize_refusal_cell(entry, "xstest_unsafe", cells["unsafe"])

    assert not safe_summary.valid
    assert safe_summary.denominator == 0
    assert safe_summary.rate is None
    assert safe_summary.excluded_incomplete == 250

    assert not unsafe_summary.valid
    assert unsafe_summary.denominator == 0
    assert unsafe_summary.excluded_incomplete == 200


def test_run_index_expected_n_uses_frozen_population_not_rate_denominator(
    tmp_path,
) -> None:
    rows = fx.refusal_rows_with_exclusions(
        250,
        200,
        safe_incomplete=10,
        unsafe_incomplete=5,
    )
    manifest_path = _build_manifest_with_xstest(tmp_path, rows)
    loaded = load_publication_run(load_manifest(manifest_path))
    run_index = build_run_index(loaded)
    safe_row = run_index[
        (run_index["model_key"] == "m1") & (run_index["benchmark"] == "xstest_safe")
    ].iloc[0]
    model_row = build_model_summary(loaded)[
        (build_model_summary(loaded)["benchmark"] == "xstest_safe")
    ].iloc[0]

    assert safe_row["expected_n"] == 250
    assert safe_row["observed_n"] == 250
    assert bool(safe_row["valid"])
    assert model_row["denominator"] == 240


def _write_single_entry_manifest(
    tmp_path,
    entry: ManifestEntry,
    ccpc_dir=None,
    include_refusal_dirs: bool = False,
):
    manifest_path = tmp_path / "manifest.json"
    if include_refusal_dirs and entry.xstest_dir is None and entry.orbench_dir is None:
        xstest_dir = tmp_path / "xstest_stub"
        orbench_dir = tmp_path / "orbench_stub"
        fx.write_refusal_run(
            xstest_dir,
            entry.model_repo_id,
            fx.XSTEST_DATASET,
            fx.all_judged_true_refusal_rows(250, 200, refuse=False),
        )
        fx.write_refusal_run(
            orbench_dir,
            entry.model_repo_id,
            fx.OR_BENCH_DATASET,
            fx.all_judged_true_refusal_rows(1319, 655, refuse=False),
        )
        entry = ManifestEntry(
            model_key=entry.model_key,
            lineage=entry.lineage,
            checkpoint_label=entry.checkpoint_label,
            model_repo_id=entry.model_repo_id,
            ccpc_dir=entry.ccpc_dir,
            xstest_dir=str(xstest_dir),
            orbench_dir=str(orbench_dir),
        )
    entry_fields = {
        "model_key": entry.model_key,
        "lineage": entry.lineage,
        "checkpoint_label": entry.checkpoint_label,
        "model_repo_id": entry.model_repo_id,
    }
    if ccpc_dir is not None:
        entry_fields["ccpc_dir"] = str(ccpc_dir)
    elif entry.ccpc_dir:
        entry_fields["ccpc_dir"] = entry.ccpc_dir
    if entry.xstest_dir:
        entry_fields["xstest_dir"] = entry.xstest_dir
    if entry.orbench_dir:
        entry_fields["orbench_dir"] = entry.orbench_dir
    payload = {
        "run_id": "r1",
        "entries": [entry_fields],
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return manifest_path


def _build_manifest_with_xstest(tmp_path, xstest_rows: list[dict]):
    dataset_path = tmp_path / "ccpc500.jsonl"
    dataset_rows = fx.default_ccpc500_dataset_rows()
    dataset_sha256, expected_rows = fx.write_ccpc_dataset(dataset_path, dataset_rows)
    all_ids = [row_id for row_id, _, _ in dataset_rows]
    ccpc_dir = tmp_path / "ccpc"
    fx.write_ccpc_run(
        ccpc_dir,
        "org/model",
        str(dataset_path),
        dataset_sha256,
        expected_rows,
        [(row_id, "judged_false", False) for row_id in all_ids],
    )
    xstest_dir = tmp_path / "xstest"
    fx.write_refusal_run(xstest_dir, "org/model", fx.XSTEST_DATASET, xstest_rows)
    orbench_dir = tmp_path / "orbench"
    fx.write_refusal_run(
        orbench_dir,
        "org/model",
        fx.OR_BENCH_DATASET,
        fx.all_judged_true_refusal_rows(1319, 655, refuse=False),
    )
    manifest_path = tmp_path / "manifest.json"
    fx.write_manifest(
        manifest_path,
        "r1",
        [
            {
                "model_key": "m1",
                "lineage": "l",
                "checkpoint_label": "c",
                "model_repo_id": "org/model",
                "ccpc_dir": str(ccpc_dir),
                "xstest_dir": str(xstest_dir),
                "orbench_dir": str(orbench_dir),
            }
        ],
    )
    return manifest_path
