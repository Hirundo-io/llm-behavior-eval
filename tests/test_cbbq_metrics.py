import pandas as pd

from llm_behavior_eval.evaluation_utils.cbbq_evaluator import (
    CBBQ_NEUTRAL_LABEL,
    CBBQ_NON_TARGET_LABEL,
    CBBQ_POLARITY_NEG,
    CBBQ_POLARITY_NON_NEG,
    CBBQ_TARGET_LABEL,
    CbbqBiasCounts,
    CbbqEvaluator,
    CbbqUnbiasCounts,
    extract_cbbq_prediction,
)
from llm_behavior_eval.evaluation_utils.eval_config import EvaluationConfig


def test_extract_cbbq_prediction_matches_choices() -> None:
    assert extract_cbbq_prediction("Answer: A") == CBBQ_TARGET_LABEL
    assert extract_cbbq_prediction(" B") == CBBQ_NON_TARGET_LABEL
    assert extract_cbbq_prediction("C") == CBBQ_NEUTRAL_LABEL
    assert extract_cbbq_prediction("the answer is A") is None
    assert extract_cbbq_prediction("No answer here") is None


def test_cbbq_bias_counts_metrics() -> None:
    counts = CbbqBiasCounts()
    counts.record_prediction(CBBQ_TARGET_LABEL, CBBQ_POLARITY_NEG)
    counts.record_prediction(CBBQ_NON_TARGET_LABEL, CBBQ_POLARITY_NEG)
    counts.record_prediction(CBBQ_NEUTRAL_LABEL, CBBQ_POLARITY_NEG)
    counts.record_prediction(None, CBBQ_POLARITY_NEG)

    metrics_result = counts.metrics(total_samples=4)
    assert metrics_result["bias_score"] == 1 / 4
    assert metrics_result["reverse_bias_score"] == 1 / 4
    assert metrics_result["neutrality_rate"] == 1 / 4
    assert metrics_result["neutrality_error_rate"] == 2 / 4
    assert metrics_result["invalid_answer_rate"] == 1 / 4
    assert counts.invalid_responses == 1


def test_cbbq_unbias_counts_metrics() -> None:
    counts = CbbqUnbiasCounts()
    counts.record_prediction(
        CBBQ_TARGET_LABEL,
        gold_label=CBBQ_TARGET_LABEL,
        polarity_value=CBBQ_POLARITY_NEG,
    )
    counts.record_prediction(
        CBBQ_NEUTRAL_LABEL,
        gold_label=CBBQ_NON_TARGET_LABEL,
        polarity_value=CBBQ_POLARITY_NON_NEG,
    )
    counts.record_prediction(
        None,
        gold_label=CBBQ_NON_TARGET_LABEL,
        polarity_value=CBBQ_POLARITY_NON_NEG,
    )

    metrics_result = counts.metrics(total_samples=3)
    assert metrics_result["disambiguated_accuracy"] == 1 / 3
    assert metrics_result["disambiguated_bias_score"] == 1.0
    assert metrics_result["invalid_answer_rate"] == 1 / 3
    assert counts.invalid_responses == 1


def test_cbbq_unbias_counts_disambiguated_bias_zero_when_all_neutral() -> None:
    counts = CbbqUnbiasCounts()
    counts.record_prediction(
        CBBQ_NEUTRAL_LABEL,
        gold_label=CBBQ_NEUTRAL_LABEL,
        polarity_value=CBBQ_POLARITY_NEG,
    )
    counts.record_prediction(
        None,
        gold_label=CBBQ_TARGET_LABEL,
        polarity_value=CBBQ_POLARITY_NEG,
    )

    metrics_result = counts.metrics(total_samples=2)
    assert metrics_result["disambiguated_accuracy"] == 0.5
    assert metrics_result["disambiguated_bias_score"] == 0.0
    assert metrics_result["invalid_answer_rate"] == 0.5


def test_cbbq_finalize_artifacts_writes_summary_contract(tmp_path) -> None:
    evaluator = object.__new__(CbbqEvaluator)
    evaluator.eval_config = EvaluationConfig(
        model_path_or_repo_id="fake-org/fake-model",
        results_dir=tmp_path,
    )

    evaluator.finalize_artifacts(
        {
            "dimension_id": "SES",
            "dimension_label": "Socio-Economic Status (SES)",
            "dataset_type": "bias",
            "bias_score": 0.5,
            "reverse_bias_score": 0.2,
            "neutrality_rate": 0.1,
            "neutrality_error_rate": 0.9,
            "invalid_answer_rate": 0.0,
            "invalid_responses": 0.0,
            "num_samples": 10.0,
            "num_samples_evaluated": 10.0,
        }
    )
    evaluator.finalize_artifacts(
        {
            "dimension_id": "SES",
            "dimension_label": "Socio-Economic Status (SES)",
            "dataset_type": "unbias",
            "disambiguated_accuracy": 0.7,
            "disambiguated_bias_score": 0.4,
            "invalid_answer_rate": 0.1,
            "invalid_responses": 1.0,
            "num_samples": 10.0,
            "num_samples_evaluated": 9.0,
        }
    )

    model_dir = tmp_path / "fake-model"
    summary_path = model_dir / "summary_full.csv"
    overall_summary_path = model_dir / "summary_brief.csv"
    assert summary_path.exists()
    assert overall_summary_path.exists()

    summary_df = pd.read_csv(summary_path)
    assert {"dimension_id", "dimension_label", "dataset_type"} <= set(
        summary_df.columns
    )
    assert "disambiguated_accuracy" in summary_df.columns
    assert "accuracy" not in summary_df.columns
    overall_df = pd.read_csv(overall_summary_path)
    assert {"dataset_type", "num_dimensions"} <= set(overall_df.columns)
    assert "disambiguated_accuracy" in overall_df.columns
    assert "accuracy" not in overall_df.columns
