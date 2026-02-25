from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from dataset_processing_scripts.upload_cbbq_to_hub import (
    CBBQ_SOURCE_ROOT,
    CBBQ_SOURCE_SPLITS,
    build_conversion_plan,
    build_repo_id,
    normalize_bias_types,
    validate_all,
    validate_and_project_cbbq_dataframe,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_build_repo_id_uses_multi_choice_suffix() -> None:
    assert build_repo_id("gender", "bias") == "hirundo-io/cbbq-gender-bias-multi-choice"
    assert build_repo_id("SES", "unbias") == "hirundo-io/cbbq-SES-unbias-multi-choice"


def test_normalize_bias_types_defaults_and_unknown_types() -> None:
    all_types = normalize_bias_types(None)
    assert "gender" in all_types
    assert "SES" in all_types

    assert normalize_bias_types(["GeNdEr", "SES"]) == ("gender", "SES")
    with pytest.raises(ValueError, match="Unknown CBBQ types"):
        normalize_bias_types(["not-a-type"])


def test_validate_and_project_cbbq_dataframe_accepts_valid_rows() -> None:
    source = pd.DataFrame(
        {
            "context": ["context"],
            "question": ["what next"],
            "ans0": ["a0"],
            "ans1": ["a1"],
            "ans2": ["a2"],
            "label": ["1"],
            "question_polarity": ["neg"],
        }
    )

    projected, errors = validate_and_project_cbbq_dataframe(
        source,
        bias_type="gender",
        source_split="ambiguous",
    )
    assert errors == []
    assert list(projected.columns) == [
        "context",
        "question",
        "ans0",
        "ans1",
        "ans2",
        "label",
        "question_polarity",
    ]
    assert int(projected.loc[0, "label"]) == 1
    assert projected.loc[0, "question_polarity"] == "neg"


def test_validate_and_project_cbbq_dataframe_reports_row_level_errors() -> None:
    source = pd.DataFrame(
        {
            "context": ["context", ""],
            "question": ["q", "q"],
            "ans0": ["a0", "a0"],
            "ans1": ["a1", "a1"],
            "ans2": ["a2", "a2"],
            "label": ["bad", "1"],
            "question_polarity": ["unknown", "non_neg"],
        }
    )

    projected, errors = validate_and_project_cbbq_dataframe(
        source,
        bias_type="gender",
        source_split="ambiguous",
    )

    assert projected.empty
    assert any("invalid label values" in error for error in errors)
    assert any("invalid question_polarity" in error for error in errors)
    assert any("empty required values" in error for error in errors)


def _build_minimal_cbbq_root(tmp_path: Path, bias_type: str) -> Path:
    root = tmp_path / "cbbq"
    base = root / CBBQ_SOURCE_ROOT / bias_type
    for source_split in CBBQ_SOURCE_SPLITS:
        split_dir = base / source_split
        split_dir.mkdir(parents=True, exist_ok=True)
        file_path = split_dir / f"{source_split}.csv"
        file_path.write_text(
            "context,question,ans0,ans1,ans2,label,question_polarity\n"
            "ctx,q,a0,a1,a2,0,neg\n"
            "ctx2,q2,a02,a12,a22,2,non_neg\n"
        )
    return root


def test_build_conversion_plan_targets_expected_repos(tmp_path: Path) -> None:
    root = _build_minimal_cbbq_root(tmp_path, "gender")
    plans = build_conversion_plan(root, ("gender",))
    targets = {entry.target_repo_id for entry in plans}
    assert len(plans) == len(CBBQ_SOURCE_SPLITS)
    assert "hirundo-io/cbbq-gender-bias-multi-choice" in targets
    assert "hirundo-io/cbbq-gender-unbias-multi-choice" in targets


def test_validate_all_requires_all_requested_splits(tmp_path: Path) -> None:
    root = _build_minimal_cbbq_root(tmp_path, "gender")
    validated = validate_all(root, ("gender",))
    assert len(validated) == len(CBBQ_SOURCE_SPLITS)
    repo_ids = {entry.plan.target_repo_id for entry in validated}
    assert "hirundo-io/cbbq-gender-bias-multi-choice" in repo_ids
    assert "hirundo-io/cbbq-gender-unbias-multi-choice" in repo_ids
