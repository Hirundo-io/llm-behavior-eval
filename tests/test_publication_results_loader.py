"""Fail-closed validation for the raw-results loader and input manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tests._publication_results_fixtures as fx
from analysis.ccpc_bench_prereg.publication_results.dataset_join import (
    DatasetJoinError,
    resolve_shared_ccpc_dataset,
)
from analysis.ccpc_bench_prereg.publication_results.manifest_schema import (
    ManifestEntry,
    PublicationManifestError,
    load_manifest,
)
from analysis.ccpc_bench_prereg.publication_results.raw_loader import (
    PublicationLoaderError,
    load_ccpc_cell,
)


def _one_model_setup(tmp_path: Path) -> tuple[ManifestEntry, Path, str, int]:
    dataset_path = tmp_path / "ccpc500.jsonl"
    rows = fx.default_ccpc500_dataset_rows()
    dataset_sha256, expected_rows = fx.write_ccpc_dataset(dataset_path, rows)
    all_ids = [row_id for row_id, _, _ in rows]
    ccpc_dir = tmp_path / "model" / "ccpc"
    row_statuses = [(row_id, "judged_false", False) for row_id in all_ids]
    fx.write_ccpc_run(
        ccpc_dir,
        "org/model",
        str(dataset_path),
        dataset_sha256,
        expected_rows,
        row_statuses,
    )
    entry = ManifestEntry(
        model_key="m1",
        lineage="lineage1",
        checkpoint_label="ckpt1",
        model_repo_id="org/model",
        ccpc_dir=str(ccpc_dir),
    )
    return entry, dataset_path, dataset_sha256, expected_rows


def test_valid_run_loads_cleanly(tmp_path: Path) -> None:
    entry, _, _, _ = _one_model_setup(tmp_path)
    cell = load_ccpc_cell(entry)
    assert not cell.missing
    assert cell.judged_true + cell.judged_false == 500
    assert len(cell.rows) == 500


def test_missing_directory_is_missing_not_an_error(tmp_path: Path) -> None:
    entry = ManifestEntry(
        model_key="m1",
        lineage="lineage1",
        checkpoint_label="ckpt1",
        model_repo_id="org/model",
        ccpc_dir=str(tmp_path / "does-not-exist"),
    )
    cell = load_ccpc_cell(entry)
    assert cell.missing


def test_duplicate_benchmark_id_raises(tmp_path: Path) -> None:
    entry, dataset_path, dataset_sha256, expected_rows = _one_model_setup(tmp_path)
    responses_path = Path(entry.ccpc_dir) / "responses.json"
    responses = json.loads(responses_path.read_text())
    responses[1]["benchmark_id"] = responses[0]["benchmark_id"]
    responses_path.write_text(json.dumps(responses))
    with pytest.raises(PublicationLoaderError, match="duplicate"):
        load_ccpc_cell(entry)


def test_unknown_status_raises(tmp_path: Path) -> None:
    entry, _, _, _ = _one_model_setup(tmp_path)
    responses_path = Path(entry.ccpc_dir) / "responses.json"
    responses = json.loads(responses_path.read_text())
    responses[0]["status"] = "totally_made_up_status"
    responses_path.write_text(json.dumps(responses))
    with pytest.raises(PublicationLoaderError, match="unknown status"):
        load_ccpc_cell(entry)


def test_wrong_ccpc_denominator_raises(tmp_path: Path) -> None:
    entry, _, _, _ = _one_model_setup(tmp_path)
    run_config_path = Path(entry.ccpc_dir) / "run_config.json"
    run_config = json.loads(run_config_path.read_text())
    run_config["ccpc_benchmark"]["expected_rows"] = 216
    run_config_path.write_text(json.dumps(run_config))
    with pytest.raises(PublicationLoaderError, match="wrong CCPC denominator"):
        load_ccpc_cell(entry)


def test_model_identity_mismatch_raises(tmp_path: Path) -> None:
    entry, _, _, _ = _one_model_setup(tmp_path)
    run_config_path = Path(entry.ccpc_dir) / "run_config.json"
    run_config = json.loads(run_config_path.read_text())
    run_config["evaluation_config"]["model_path_or_repo_id"] = "org/some-other-model"
    run_config_path.write_text(json.dumps(run_config))
    with pytest.raises(PublicationLoaderError, match="model identity mismatch"):
        load_ccpc_cell(entry)


def test_metrics_csv_cross_check_mismatch_raises(tmp_path: Path) -> None:
    entry, _, _, _ = _one_model_setup(tmp_path)
    metrics_path = Path(entry.ccpc_dir) / "metrics.csv"
    metrics_path.write_text(metrics_path.read_text().replace("500", "499"))
    with pytest.raises(PublicationLoaderError, match="disagree"):
        load_ccpc_cell(entry)


def test_missing_responses_file_raises(tmp_path: Path) -> None:
    entry, _, _, _ = _one_model_setup(tmp_path)
    (Path(entry.ccpc_dir) / "responses.json").unlink()
    with pytest.raises(PublicationLoaderError, match="responses.json is missing"):
        load_ccpc_cell(entry)


def test_dataset_identity_mismatch_across_models_raises(tmp_path: Path) -> None:
    entry_a, dataset_path, dataset_sha256, expected_rows = _one_model_setup(tmp_path)
    other_rows = fx.default_ccpc500_dataset_rows()
    other_rows[0] = ("ccpc500-9999", "topic_a", "form_x")
    other_dataset_path = tmp_path / "ccpc500_other.jsonl"
    other_sha256, _ = fx.write_ccpc_dataset(other_dataset_path, other_rows)
    ccpc_dir_b = tmp_path / "model_b" / "ccpc"
    row_statuses = [(row_id, "judged_false", False) for row_id, _, _ in other_rows]
    fx.write_ccpc_run(
        ccpc_dir_b,
        "org/model-b",
        str(other_dataset_path),
        other_sha256,
        expected_rows,
        row_statuses,
    )
    entry_b = ManifestEntry(
        model_key="m2",
        lineage="lineage2",
        checkpoint_label="ckpt2",
        model_repo_id="org/model-b",
        ccpc_dir=str(ccpc_dir_b),
    )
    cells = {"m1": load_ccpc_cell(entry_a), "m2": load_ccpc_cell(entry_b)}
    with pytest.raises(DatasetJoinError, match="disagree"):
        resolve_shared_ccpc_dataset(cells)


def test_manifest_rejects_duplicate_model_key(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "r1",
                "entries": [
                    {
                        "model_key": "dup",
                        "lineage": "l",
                        "checkpoint_label": "c",
                        "model_repo_id": "org/a",
                    },
                    {
                        "model_key": "dup",
                        "lineage": "l",
                        "checkpoint_label": "c",
                        "model_repo_id": "org/b",
                    },
                ],
            }
        )
    )
    with pytest.raises(PublicationManifestError, match="duplicate"):
        load_manifest(manifest_path)


def test_manifest_rejects_unknown_qwen_family_pair_member(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "r1",
                "entries": [
                    {
                        "model_key": "a",
                        "lineage": "l",
                        "checkpoint_label": "c",
                        "model_repo_id": "org/a",
                    }
                ],
                "qwen_family_pair": ["a", "does-not-exist"],
            }
        )
    )
    with pytest.raises(PublicationManifestError, match="unknown model_key"):
        load_manifest(manifest_path)
