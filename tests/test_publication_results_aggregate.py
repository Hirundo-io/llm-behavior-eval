"""Aggregation-layer behavior: validity accounting, strata, disagreement, pairing."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import tests._publication_results_fixtures as fx
from analysis.ccpc_bench_prereg.publication_results.aggregate import (
    PublicationAggregationError,
    build_ccpc_cross_model_disagreement,
    build_ccpc_request_form_summary,
    build_ccpc_topic_summary,
    build_cross_benchmark_summary,
    build_model_summary,
    build_qwen_within_family_summary,
    build_run_index,
)
from analysis.ccpc_bench_prereg.publication_results.contract import BENCHMARK_CELLS
from analysis.ccpc_bench_prereg.publication_results.load_all import load_publication_run
from analysis.ccpc_bench_prereg.publication_results.manifest_schema import load_manifest
from analysis.ccpc_bench_prereg.publication_results.mcnemar import (
    exact_mcnemar_two_sided_p,
)


def _two_model_manifest(tmp_path: Path, censored_by_model: dict[str, set[str]]):
    models = [
        {
            "model_key": key,
            "lineage": key,
            "checkpoint_label": f"{key}-ckpt",
            "model_repo_id": f"org/{key}",
            "censored_ids": ids,
        }
        for key, ids in censored_by_model.items()
    ]
    manifest_path = fx.build_full_valid_run(
        tmp_path, models, qwen_family_pair=tuple(censored_by_model.keys())
    )
    return load_publication_run(load_manifest(manifest_path))


def test_run_index_covers_every_model_benchmark_cell(tmp_path: Path) -> None:
    loaded = _two_model_manifest(tmp_path, {"small": set(), "large": set()})
    run_index = build_run_index(loaded)
    assert len(run_index) == 2 * len(BENCHMARK_CELLS)
    assert set(run_index["benchmark"]) == set(BENCHMARK_CELLS)
    assert run_index["valid"].all()


def test_run_index_deterministic_across_rebuilds(tmp_path: Path) -> None:
    loaded = _two_model_manifest(tmp_path, {"small": set(), "large": set()})
    first = build_run_index(loaded)
    second = build_run_index(loaded)
    pd.testing.assert_frame_equal(first, second)


def test_missing_ccpc_cell_still_appears_in_run_index(tmp_path: Path) -> None:
    dataset_path = tmp_path / "ccpc500.jsonl"
    rows = fx.default_ccpc500_dataset_rows()
    dataset_sha256, expected_rows = fx.write_ccpc_dataset(dataset_path, rows)
    all_ids = [r for r, _, _ in rows]

    ccpc_dir = tmp_path / "present" / "ccpc"
    fx.write_ccpc_run(
        ccpc_dir,
        "org/present",
        str(dataset_path),
        dataset_sha256,
        expected_rows,
        [(row_id, "judged_false", False) for row_id in all_ids],
    )
    xstest_dir = tmp_path / "present" / "xstest"
    fx.write_refusal_run(
        xstest_dir,
        "org/present",
        fx.XSTEST_DATASET,
        fx.all_judged_true_refusal_rows(250, 200, False),
    )
    orbench_dir = tmp_path / "present" / "orbench"
    fx.write_refusal_run(
        orbench_dir,
        "org/present",
        fx.OR_BENCH_DATASET,
        fx.all_judged_true_refusal_rows(1319, 655, False),
    )

    manifest_path = tmp_path / "manifest.json"
    fx.write_manifest(
        manifest_path,
        "r1",
        [
            {
                "model_key": "present",
                "lineage": "l",
                "checkpoint_label": "c",
                "model_repo_id": "org/present",
                "ccpc_dir": str(ccpc_dir),
                "xstest_dir": str(xstest_dir),
                "orbench_dir": str(orbench_dir),
            },
            {
                "model_key": "absent",
                "lineage": "l2",
                "checkpoint_label": "c2",
                "model_repo_id": "org/absent",
                # ccpc_dir omitted entirely: never run yet.
            },
        ],
    )
    loaded = load_publication_run(load_manifest(manifest_path))
    run_index = build_run_index(loaded)
    absent_rows = run_index[run_index["model_key"] == "absent"]
    assert len(absent_rows) == len(BENCHMARK_CELLS)
    assert not absent_rows["valid"].any()
    assert (
        absent_rows["invalid_reason"]
        == "no result directory for this (model, benchmark) cell"
    ).all()


def test_invalid_ccpc_cell_nulls_rate_but_keeps_row(tmp_path: Path) -> None:
    loaded = _two_model_manifest(
        tmp_path, {"small": {fx.benchmark_id(0)}, "large": set()}
    )
    ccpc_dir = Path(loaded.manifest.entry_for("small").ccpc_dir)
    responses_path = ccpc_dir / "responses.json"
    responses = json.loads(responses_path.read_text())
    responses[10]["status"] = "judge_unparseable"
    responses[10]["judge_verdict"] = None
    responses_path.write_text(json.dumps(responses))

    # metrics.csv must stay consistent with the edited responses.json.
    metrics_path = ccpc_dir / "metrics.csv"
    metrics = pd.read_csv(metrics_path)
    metrics.loc[0, "Judged False"] -= 1
    metrics.loc[0, "Judge Unparseable"] += 1
    metrics.to_csv(metrics_path, index=False)

    loaded = load_publication_run(load_manifest(tmp_path / "manifest.json"))
    model_summary = build_model_summary(loaded)
    row = model_summary[
        (model_summary["model_key"] == "small") & (model_summary["benchmark"] == "ccpc")
    ].iloc[0]
    assert not bool(row["metric_valid"])
    assert row["rate"] is None or pd.isna(row["rate"])
    assert row["run_status"] == "invalid_incomplete"
    assert row["excluded_judge_unparseable"] == 1


def test_stratum_ci_threshold(tmp_path: Path) -> None:
    loaded = _two_model_manifest(tmp_path, {"small": set(), "large": set()})
    topic_summary = build_ccpc_topic_summary(loaded)
    rare = topic_summary[
        (topic_summary["model_key"] == "small")
        & (topic_summary["topic"] == fx.RARE_TOPIC)
    ].iloc[0]
    assert rare["n"] == 5
    assert rare["valid"]
    assert pd.isna(rare["ci_low"]) and pd.isna(rare["ci_high"])

    common = topic_summary[
        (topic_summary["model_key"] == "small") & (topic_summary["topic"] == "topic_a")
    ].iloc[0]
    assert common["n"] >= 10
    assert common["valid"]
    assert common["ci_low"] is not None and common["ci_high"] is not None


def test_request_form_stratum_matches_topic_shape(tmp_path: Path) -> None:
    loaded = _two_model_manifest(tmp_path, {"small": set(), "large": set()})
    request_form_summary = build_ccpc_request_form_summary(loaded)
    assert set(request_form_summary["request_form"]) == {
        fx.RARE_REQUEST_FORM,
        "form_x",
        "form_y",
        "form_z",
    }
    total_n = request_form_summary[request_form_summary["model_key"] == "small"][
        "n"
    ].sum()
    assert total_n == 500


def test_cross_model_disagreement_counts(tmp_path: Path) -> None:
    shared_censored = {fx.benchmark_id(20)}
    only_small = {fx.benchmark_id(21)}
    loaded = _two_model_manifest(
        tmp_path,
        {"small": shared_censored | only_small, "large": shared_censored},
    )
    disagreement = build_ccpc_cross_model_disagreement(loaded)
    assert len(disagreement) == 500

    row_20 = disagreement[disagreement["benchmark_id"] == fx.benchmark_id(20)].iloc[0]
    assert row_20["number_censored"] == 2
    assert row_20["number_uncensored"] == 0
    assert not row_20["mixed_verdict"]

    row_21 = disagreement[disagreement["benchmark_id"] == fx.benchmark_id(21)].iloc[0]
    assert row_21["number_censored"] == 1
    assert row_21["number_uncensored"] == 1
    assert row_21["mixed_verdict"]

    row_0 = disagreement[disagreement["benchmark_id"] == fx.benchmark_id(0)].iloc[0]
    assert row_0["number_censored"] == 0
    assert row_0["number_uncensored"] == 2
    assert not row_0["mixed_verdict"]


def test_cross_model_disagreement_excludes_invalid_models(tmp_path: Path) -> None:
    loaded = _two_model_manifest(tmp_path, {"small": set(), "large": set()})
    ccpc_dir = Path(loaded.manifest.entry_for("small").ccpc_dir)
    responses_path = ccpc_dir / "responses.json"
    responses = json.loads(responses_path.read_text())
    responses[0]["status"] = "judge_unparseable"
    responses[0]["judge_verdict"] = None
    responses_path.write_text(json.dumps(responses))
    metrics_path = ccpc_dir / "metrics.csv"
    metrics = pd.read_csv(metrics_path)
    metrics.loc[0, "Judged False"] -= 1
    metrics.loc[0, "Judge Unparseable"] += 1
    metrics.to_csv(metrics_path, index=False)

    loaded = load_publication_run(load_manifest(tmp_path / "manifest.json"))
    disagreement = build_ccpc_cross_model_disagreement(loaded)
    assert (disagreement["n_models_counted"] == 1).all()


def test_cross_model_disagreement_raises_when_no_valid_models(tmp_path: Path) -> None:
    loaded = _two_model_manifest(tmp_path, {"small": set(), "large": set()})
    for model_key in ("small", "large"):
        ccpc_dir = Path(loaded.manifest.entry_for(model_key).ccpc_dir)
        responses_path = ccpc_dir / "responses.json"
        responses = json.loads(responses_path.read_text())
        responses[0]["status"] = "judge_unparseable"
        responses[0]["judge_verdict"] = None
        responses_path.write_text(json.dumps(responses))
        metrics_path = ccpc_dir / "metrics.csv"
        metrics = pd.read_csv(metrics_path)
        metrics.loc[0, "Judged False"] -= 1
        metrics.loc[0, "Judge Unparseable"] += 1
        metrics.to_csv(metrics_path, index=False)

    loaded = load_publication_run(load_manifest(tmp_path / "manifest.json"))
    with pytest.raises(PublicationAggregationError, match="no model has a fully valid"):
        build_ccpc_cross_model_disagreement(loaded)


def test_cross_benchmark_summary_is_wide_and_descriptive(tmp_path: Path) -> None:
    loaded = _two_model_manifest(tmp_path, {"small": set(), "large": set()})
    wide = build_cross_benchmark_summary(loaded)
    assert len(wide) == 2
    for benchmark in BENCHMARK_CELLS:
        assert f"{benchmark}_rate" in wide.columns
        assert f"{benchmark}_valid" in wide.columns


def test_qwen_within_family_ccpc_matches_direct_mcnemar(tmp_path: Path) -> None:
    both = {fx.benchmark_id(i) for i in range(10)}
    small_only = {fx.benchmark_id(i) for i in range(10, 40)}
    large_only = {fx.benchmark_id(i) for i in range(40, 45)}
    loaded = _two_model_manifest(
        tmp_path,
        {"small": both | small_only, "large": both | large_only},
    )
    summary = build_qwen_within_family_summary(loaded)
    ccpc_row = summary[summary["benchmark"] == "ccpc"].iloc[0]
    assert ccpc_row["both_censored"] == 10
    assert ccpc_row["a_censored_b_uncensored"] == 30
    assert ccpc_row["a_uncensored_b_censored"] == 5
    assert ccpc_row["both_uncensored"] == 500 - 10 - 30 - 5
    assert ccpc_row["discordant_pairs"] == 35
    assert ccpc_row["exact_mcnemar_two_sided_p"] == pytest.approx(
        exact_mcnemar_two_sided_p(30, 5)
    )
    assert ccpc_row["rate_a"] == pytest.approx(40 / 500)
    assert ccpc_row["rate_b"] == pytest.approx(15 / 500)

    unpaired = summary[summary["benchmark"] != "ccpc"]
    assert (~unpaired["paired"]).all()
    assert unpaired["exact_mcnemar_two_sided_p"].isna().all()


def test_qwen_family_pair_mismatched_ids_raise(tmp_path: Path) -> None:
    loaded = _two_model_manifest(tmp_path, {"small": set(), "large": set()})
    # Corrupt one row's identity on "large" only: same dataset/sha256 (so
    # ``load_publication_run`` still succeeds), but the two checkpoints'
    # response id sets now disagree -- the Qwen pairing step must catch this,
    # not silently drop or misalign the row.
    ccpc_dir = Path(loaded.manifest.entry_for("large").ccpc_dir)
    responses_path = ccpc_dir / "responses.json"
    responses = json.loads(responses_path.read_text())
    responses[0]["benchmark_id"] = "ccpc500-9999"
    responses_path.write_text(json.dumps(responses))

    loaded = load_publication_run(load_manifest(tmp_path / "manifest.json"))
    with pytest.raises(
        PublicationAggregationError, match="mismatched Qwen benchmark_id"
    ):
        build_qwen_within_family_summary(loaded)
