from llm_behavior_eval.evaluation_utils.cbbq_evaluator import (
    CBBQ_NEUTRAL_LABEL,
    CBBQ_NON_TARGET_LABEL,
    CBBQ_POLARITY_NEG,
    CBBQ_POLARITY_NON_NEG,
    CBBQ_TARGET_LABEL,
    CbbqAmbiguousCounts,
    CbbqDisambiguousCounts,
    extract_cbbq_prediction,
)


def test_extract_cbbq_prediction_matches_choices() -> None:
    assert extract_cbbq_prediction("Answer: A") == CBBQ_TARGET_LABEL
    assert extract_cbbq_prediction("选项：B") == CBBQ_NON_TARGET_LABEL
    assert extract_cbbq_prediction("C") == CBBQ_NEUTRAL_LABEL
    assert extract_cbbq_prediction("No answer here") is None


def test_cbbq_ambiguous_counts_metrics() -> None:
    counts = CbbqAmbiguousCounts()
    counts.record_prediction(CBBQ_TARGET_LABEL, CBBQ_POLARITY_NEG)
    counts.record_prediction(CBBQ_NON_TARGET_LABEL, CBBQ_POLARITY_NEG)
    counts.record_prediction(CBBQ_NEUTRAL_LABEL, CBBQ_POLARITY_NEG)
    counts.record_prediction(None, CBBQ_POLARITY_NEG)

    metrics_result = counts.metrics()
    assert metrics_result["bias_score"] == 1 / 3
    assert metrics_result["reverse_bias_score"] == 1 / 3
    assert metrics_result["neutrality_rate"] == 1 / 3
    assert metrics_result["neutrality_error_rate"] == 2 / 3
    assert counts.empty_responses == 1


def test_cbbq_disambiguous_counts_metrics() -> None:
    counts = CbbqDisambiguousCounts()
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

    metrics_result = counts.metrics()
    assert metrics_result["accuracy"] == 0.5
    assert metrics_result["disamb_bias_score"] == 1.0
    assert counts.empty_responses == 0
