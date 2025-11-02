import pytest

pytest.importorskip("torch")

from llm_behavior_eval.evaluation_utils.free_text_bias_evaluator import (
    Agreement,
    extract_agreements,
    map_uncertainty_outputs,
)


def test_extract_agreements_parses_yes_no_unmatched() -> None:
    judge_output = [
        [{"generated_text": "gt_agree: yes"}],
        [{"generated_text": "gt_agree: no"}],
        [{"generated_text": "other"}],
    ]
    expected = [Agreement.YES, Agreement.NO, Agreement.UNMATCHED]
    assert extract_agreements(judge_output) == expected


def test_map_uncertainty_outputs_parses_flags() -> None:
    judge_output = [
        [{"generated_text": "candidate_uncertain: yes"}],
        [{"generated_text": "something"}],
    ]
    assert map_uncertainty_outputs(judge_output) == ["yes", "unparseable"]
