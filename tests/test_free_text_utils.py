import pytest

pytest.importorskip("torch")

from llm_behavior_eval.evaluation_utils.free_text_bias_evaluator import (
    Agreement,
    extract_agreements,
    map_uncertainty_outputs,
)
from llm_behavior_eval.evaluation_utils.free_text_refusal_evaluator import (
    RefusalJudgment,
    parse_or_bench_judgment,
    parse_xstest_judgment,
    refusal_heuristic,
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


def test_parse_or_bench_judgment_parses_refusal_variants() -> None:
    assert parse_or_bench_judgment("reason [[direct_answer]]") == RefusalJudgment.COMPLY
    assert (
        parse_or_bench_judgment("reason [[direct_refusal]]") == RefusalJudgment.REFUSE
    )
    assert (
        parse_or_bench_judgment("reason [[indirect_refusal]]") == RefusalJudgment.REFUSE
    )


def test_parse_xstest_judgment_parses_compliance_refusal_and_partial() -> None:
    assert parse_xstest_judgment("1_full_compliance") == RefusalJudgment.COMPLY
    assert parse_xstest_judgment("2_full_refusal") == RefusalJudgment.REFUSE
    assert parse_xstest_judgment("3_partial_refusal") == RefusalJudgment.PARTIAL


def test_refusal_judgment_parsers_return_unparseable_for_unknown_output() -> None:
    assert parse_or_bench_judgment("other") == RefusalJudgment.UNPARSEABLE
    assert parse_xstest_judgment("other") == RefusalJudgment.UNPARSEABLE


def test_refusal_heuristic_matches_prefixes_and_markers() -> None:
    assert refusal_heuristic("Sorry, I can't help with that.")
    assert refusal_heuristic("As an AI assistant, I must decline.")
    assert not refusal_heuristic("Here is the answer you requested.")
